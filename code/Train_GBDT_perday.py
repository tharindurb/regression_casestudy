import calendar
import math
import pickle
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

train_dataset='perishable_zerofilled_2016'
test_dataset='perishable_2017'

# loading the training dataset
f = open("../data/train_{}.p".format(train_dataset), 'rb')
train = pickle.load(f)
f.close()

# loading the test dataset
f = open("../data/train_{}.p".format(test_dataset), 'rb')
test = pickle.load(f)
f.close()

# Replacing on promotion NaN values with False(not on promotion)
train['onpromotion'].fillna(False, inplace=True)
test['onpromotion'].fillna(False, inplace=True)

# calculating mean and standard deviation of each item in the training dataset
mean_item = train[['item_nbr','unit_sales']].groupby(['item_nbr'])['unit_sales'].mean().to_frame('mean')
std_item = train[['item_nbr','unit_sales']].groupby(['item_nbr'])['unit_sales'].std().to_frame('std')

# normalising unit_sales in training data by subtracting the mean and dividing by the standard deviation of each item
train = train.merge(right=mean_item, left_on='item_nbr', right_on='item_nbr')
train = train.merge(right=std_item, left_on='item_nbr', right_on='item_nbr')
train['unit_sales'] = train['unit_sales'] - train['mean']
train['unit_sales'] = train['unit_sales'].divide(train['std'],fill_value=0)
train.drop(columns=['mean','std'],inplace=True)

# normalising unit_sales in test data by subtracting the mean and dividing by the standard deviation of each item (mean and std obtained from 2016 data)
test = test.merge(right=mean_item, left_on='item_nbr', right_on='item_nbr')
test = test.merge(right=std_item, left_on='item_nbr', right_on='item_nbr')
test['unit_sales'] = test['unit_sales'] - test['mean']
test['unit_sales'] = test['unit_sales'].divide(test['std'],fill_value=0)
test.drop(columns=['mean','std'],inplace=True)

# Indexing by the combination store id, item id and date
train.set_index(["store_nbr", "item_nbr", "date"],inplace=True)
test.set_index(["store_nbr", "item_nbr", "date"],inplace=True)

# Dropping variables
train.drop(columns=['cluster','city','week_of_year','class','perishable'],inplace=True)
test.drop(columns=['cluster','city','week_of_year','class','perishable'],inplace=True)

# Label encoding of categorical features.
enc_state= LabelEncoder()
enc_family= LabelEncoder()
enc_type= LabelEncoder()

train['state'] = enc_state.fit_transform(train['state'])
train['family'] = enc_family.fit_transform(train['family'])
train['type'] = enc_type.fit_transform(train['type'])

test['state'] = enc_state.transform(test['state'])
test['family'] = enc_family.transform(test['family'])
test['type'] = enc_type.transform(test['type'])

boosting_type = 'gbdt'
lgb_params = {'learning_rate': 0.1,
              'metric': 'rmse',
              'boosting_type': boosting_type,
              'max_depth': 10,
              'num_leaves': 60,
              'objective': 'regression',
              'min_data_per_leaf': 250,
              'num_threads': 4}

Y_true=np.array([])
Y_pred=np.array([])

# Training models for each day of week
for day in range(0,7):
    result_name = '{}_day_{}_gaussian_normalised_lgb_lr_1'.format(train_dataset, calendar.day_name[day])

    # Extracting data subsets for each day of week
    train_day = train[train.day_of_week == day].copy()
    test_day = test[test.day_of_week== day].copy()

    # Separating features and target
    X_test = test_day.drop(columns=['unit_sales'])
    Y_test = test_day['unit_sales'].values

    # Splitting the training data to train and validation 80%/20% split
    X_train, X_val, Y_train, Y_val = train_test_split(train_day.drop(columns=['unit_sales','day_of_week']),
                                                      train_day['unit_sales'].values,
                                                      test_size=0.2, random_state=42)

    print('Training dataset for day {}: {}'.format(calendar.day_name[day], X_train.shape))
    print('Validation dataset for day {}: {}'.format(calendar.day_name[day], X_val.shape))
    print('Test dataset for day {}: {}'.format(calendar.day_name[day], X_test.shape))

    # converting datasets into lgb format, list of names of categorical variable has been provided to conduct One-hot encoding
    lgb_train = lgb.Dataset(data=X_train, label=Y_train, categorical_feature=['state', 'type', 'family'])
    lgb_val = lgb.Dataset(data=X_val, label=Y_val, categorical_feature=['state', 'type', 'family'], reference=lgb_train)

    evals_result = {}
    # training the model using 100 iterations with early stopping if validation RMSE decreases
    gbm = lgb.train(lgb_params,
                num_boost_round=100,
                train_set=lgb_train,
                valid_sets=[lgb_train, lgb_val],
                verbose_eval=True,
                evals_result=evals_result,
                early_stopping_rounds=10,
                )

    # dump the trained model
    with open('../models/model_{}.pkl'.format(result_name), 'wb') as fout:
        pickle.dump(gbm, fout)

    # predictions from the model for the test dataset of the corresponding day
    Y_test_pred = gbm.predict(X_test)


    Y_true = np.append(Y_true, Y_test)
    Y_pred = np.append(Y_pred, Y_test_pred)

    # calculating the RMSE for the test dataset of the corresponding day
    test_rmse=math.sqrt(mean_squared_error(Y_test, Y_test_pred))
    print("Test RMSE for {}: {}".format(result_name,test_rmse))
    with open("../results/test_rmse.csv", "a+") as test_result:
        test_result.write('{},{}\r\n'.format(result_name,test_rmse))

    # calculating the MAPE for the test dataset of the corresponding day
    test_mape = mean_absolute_percentage_error(Y_test, Y_test_pred)
    with open("../results/test_mape.csv", "a+") as test_result:
        test_result.write('{},{}\r\n'.format(result_name, test_mape))

    # Saving the training and validation RMSE at each iteration
    pd_results = pd.DataFrame()
    for dataset in evals_result.keys():
        dataset_result = evals_result[dataset]
        for metric in dataset_result.keys():
            metric_result = dataset_result[metric]
            pd_results['{}_{}'.format(dataset,metric)]=metric_result
    pd_results.to_csv("../results/{}_metrics_boosting_type_{}.csv".format(result_name,boosting_type))

    # Saving the feature importance results
    pd_feature_importance = pd.concat([pd.Series(gbm.feature_name(), name='Feature'),
                                       pd.Series(gbm.feature_importance(importance_type='split'), name='Split'),
                                       pd.Series(gbm.feature_importance(importance_type='gain'), name='Gain')],
                                      axis=1)
    pd_feature_importance.set_index('Feature',inplace=True)
    pd_feature_importance.to_csv("../results/{}_feature_importance_boosting_type_{}.csv".format(result_name,boosting_type))

    # Plotting the training and validation RMSE at each iteration
    ax = lgb.plot_metric(evals_result, metric='rmse')
    plt.title('Gradient Boosting, RMSE')
    plt.savefig(
        "../results/images/{}_metrics_boosting_type_{}.png".format(result_name,boosting_type),
        bbox_inches='tight', dpi=500)
    # plt.show()
    plt.close()

    # Plotting the feature importance results
    ax = lgb.plot_importance(gbm, max_num_features=10)
    plt.title('Gradient Boosting feature importance')
    plt.savefig(
        "../results/images/{}_feature_importance_boosting_type_{}.png".format(result_name,boosting_type),
        bbox_inches='tight', dpi=500)
    # plt.show()
    plt.close()

result_name='{}_day_{}_gaussian_normalised_lgb_lr_1'.format(train_dataset, 'all')

# calculating the RMSE for the entire test dataset
test_rmse = math.sqrt(mean_squared_error(Y_true, Y_pred))
with open("../results/test_rmse.csv", "a+") as test_result:
    test_result.write('{},{}\r\n'.format(result_name, test_rmse))

# calculating the MAPE for the entire test dataset
test_mape = mean_absolute_percentage_error(Y_true, Y_pred)
with open("../results/test_mape.csv", "a+") as test_result:
    test_result.write('{},{}\r\n'.format(result_name, test_mape))