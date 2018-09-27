import pickle

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# loading the training dataset
train_dataset='perishable_zerofilled_2016'
f = open("../data/train_{}.p".format(train_dataset), 'rb')
train = pickle.load(f)
f.close()

result_name='{}_gaussian_normalised_lgb'.format(train_dataset)

# Replacing on promotion NaN values with False(not on promotion)
train['onpromotion'].fillna(False, inplace=True)

# calculating mean and standard deviation of each item in the training dataset
mean_item = train[['item_nbr','unit_sales']].groupby(['item_nbr'])['unit_sales'].mean().to_frame('mean')
std_item = train[['item_nbr','unit_sales']].groupby(['item_nbr'])['unit_sales'].std().to_frame('std')

# normalising unit_sales by subtracting the mean and dividing by the standard deviation of each item
train = train.merge(right=mean_item, left_on='item_nbr', right_on='item_nbr')
train = train.merge(right=std_item, left_on='item_nbr', right_on='item_nbr')
train['unit_sales'] = train['unit_sales'] - train['mean']
train['unit_sales'] = train['unit_sales'].divide(train['std'],fill_value=0)
train.drop(columns=['mean','std'],inplace=True)

# Indexing by the combination store id, item id and date
train.set_index(["store_nbr", "item_nbr", "date"],inplace=True)

# Dropping variables
train.drop(columns=['cluster','city','week_of_year','class','perishable'],inplace=True)

# Label encoding of categorical features.
enc_state= LabelEncoder()
enc_family= LabelEncoder()
enc_type= LabelEncoder()

train['state'] = enc_state.fit_transform(train['state'])
train['family'] = enc_family.fit_transform(train['family'])
train['type'] = enc_type.fit_transform(train['type'])

# Separating features and target
X_train = train.drop(columns=['unit_sales'])
Y_train = train['unit_sales'].values

print('Training dataset: {}'.format(X_train.shape))

# Iterating across the parameter grid learning rate x minimum data in the leaf of tree
for learning_rate in [0.01, 0.05, 0.1]:
    for min_data_per_leaf in [50, 100, 250, 500]:
        lgb_params = {'learning_rate'    : learning_rate,
                      'metric'           : 'rmse',
                      'boosting_type'   : 'gbdt',
                      'max_depth': 10,
                      'num_leaves'       : 60,
                      'objective'        : 'regression',
                      'min_data_per_leaf': min_data_per_leaf,
                      'num_threads':4}

        # converting dataset into lgb format, list of names of categorical variable has been provided to conduct One-hot encoding
        lgb_train = lgb.Dataset(data=X_train, label=Y_train, categorical_feature=['state', 'type', 'family'])

        evals_result = {}

        # cross validation is handled by the lgb package itself
        cv_results = lgb.cv(lgb_params,
                    num_boost_round=100,
                    train_set=lgb_train,
                    verbose_eval=True,
                    categorical_feature=['state', 'type', 'family'],
                    early_stopping_rounds=10,
                    stratified=False
                    )
        print('learning_rate: {}, min_data_per_leaf: {}, rmse-mean: {}, rmse-stdv: {}'.format(learning_rate,min_data_per_leaf,cv_results['rmse-mean'][-1],cv_results['rmse-stdv'][-1]))
