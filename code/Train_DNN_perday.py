import calendar
import math
import pickle

import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

# One-hot encoding of categorical features.
train=pd.get_dummies(train, columns=['state', 'type', 'family'])
test=pd.get_dummies(test, columns=['state', 'type', 'family'])
dropout=0.2

Y_true=np.array([])
Y_pred=np.array([])

# Training models for each day of week
for day in range(0,7):
    model_name='DNN_256_128'
    result_name='{}_day_{}_gaussian_normalised_dnn'.format(train_dataset,calendar.day_name[day])

    # Extracting data subsets for each day of week
    train_day = train[train.day_of_week == day].copy()
    test_day = test[test.day_of_week== day].copy()

    # Separating features and target
    X = train_day.drop(columns=['unit_sales','day_of_week']).values
    Y = train_day['unit_sales'].values
    X_test = test_day.drop(columns=['unit_sales','day_of_week']).values
    Y_test = test_day['unit_sales'].values

    # Splitting the training data to train and validation 80%/20% split
    X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size=0.2, random_state=42)

    print('Training dataset for day {}: {}'.format(calendar.day_name[day],X_train.shape))
    print('Validation dataset for day {}: {}'.format(calendar.day_name[day],X_val.shape))
    print('Test dataset for day {}: {}'.format(calendar.day_name[day],X_test.shape))

    # configuring the deep neural network
    # setting number of neurones in each hidden layer and activation function
    # adding a dropout layer in between each hidden layer to reduce overfitting
    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mean_squared_error'])

    # setting early stopping trigger to monitor validation MSE and to call off training if validation MSE drops in 5 consecutive training iterations
    early_stopping = EarlyStopping(monitor='val_mean_squared_error', min_delta=0, patience=5, verbose=0, mode='min')
    history = History()

    model.fit(X_train, Y_train,
              validation_data=(X_val, Y_val),
              epochs=10,
              verbose=2,
              batch_size=1024,
              callbacks=[early_stopping, history])
    model.save("../models_new/{}_model_{}_final.h5".format(result_name,model_name))

    # predictions from the model for the test dataset of the corresponding day
    Y_test_pred= np.ravel(model.predict(X_test))

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
    rmse_train=[math.sqrt(y) for y in history.history['mean_squared_error']]
    rmse_val=[math.sqrt(y) for y in history.history['val_mean_squared_error']]
    with open("../results/{}_model_{}.csv".format(result_name,model_name), mode='w') as statSaveFile:
        statSaveFile.write("Iter,Train_RMSE,Test_RMSE\n")
        for i in range(0, len(rmse_train)):
            statSaveFile.write("{},{},{}\n".format(i+1, rmse_train[i],rmse_val[i]))

    print("Completed: {}_model_{}".format(result_name,model_name))
    K.clear_session()

# calculating the RMSE for the entire test dataset
result_name='{}_day_{}_gaussian_normalised_dnn'.format(train_dataset, 'all')
test_rmse = math.sqrt(mean_squared_error(Y_true, Y_pred))
with open("../results/test_rmse.csv", "a+") as test_result:
    test_result.write('{},{}\r\n'.format(result_name, test_rmse))

# calculating the MAPE for the entire test dataset
test_mape = mean_absolute_percentage_error(Y_true, Y_pred)
with open("../results/test_mape.csv", "a+") as test_result:
    test_result.write('{},{}\r\n'.format(result_name, test_mape))
