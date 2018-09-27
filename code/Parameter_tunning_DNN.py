import gc
import math
import pickle

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, History
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import KFold

train_dataset='perishable_zerofilled_2016'

# loading the training dataset
f = open("../data/train_{}.p".format(train_dataset), 'rb')
train = pickle.load(f)
f.close()

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

# One-hot encoding of categorical features.
X=pd.get_dummies(train.drop(columns=['unit_sales']), columns=['state', 'type', 'family']).values
Y = train['unit_sales'].values
gc.collect()

# Dividing dataset into 5 folds for 5 fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cvscores_train = []
cvscores_test = []
dropout=0.2

# Iterating across different layer settings
for layers in [[256],[256,128],[256,128,64]]:
    model_name = "DNN_{}_layers".format(train_dataset)
    for layer in layers:
        model_name = "{}_{}".format(model_name, layer)

    # Iterating across different k-fold splits of the dataset
    for train_index, validation_index in kfold.split(X, Y):

        # configuring the deep neural network
        model = Sequential()
        i = 1
        for layer in layers:
            if i == 1:
                # setting number of neurones in the first hidden layer, input datset size and activation function
                model.add(Dense(layer, input_dim=X.shape[1], name="l_{}".format(i), kernel_initializer='normal', activation='relu'))
            else:
                # setting number of neurones in the hidden layers, and activation function
                model.add(Dense(layer, name="l_{}".format(i), kernel_initializer='normal', activation='relu'))
            if dropout > 0:
                # adding a dropout layer in between each hidden layer to reduce overfitting
                model.add(Dropout(dropout))
            i += 1
        model.add(Dense(1, kernel_initializer='normal'))
        # setting the optimisation metric to MSE and optimiser to ADAM
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mean_squared_error'])

        # setting early stopping trigger to monitor validation MSE and to call off training if validation MSE drops in 5 consecutive training iterations
        early_stopping = EarlyStopping(monitor='val_mean_squared_error', min_delta=0, patience=5, verbose=0, mode='min')
        history = History()

        # training the DNN model with 25 iterations
        model.fit(X[train_index], Y[train_index],
                  validation_data=(X[validation_index], Y[validation_index]),
                  epochs=25,
                  verbose=2,
                  batch_size=1024,
                  callbacks=[early_stopping, history])

        # calculating training and validation RMSE
        rmse_train = math.sqrt(history.history['mean_squared_error'][-1])
        rmse_test = math.sqrt(history.history['val_mean_squared_error'][-1])
        print("rmse train: {}, rmse test: {}".format(rmse_train, rmse_test))
        cvscores_train.append(rmse_train)
        cvscores_test.append(rmse_test)

    # calculating the mean and std of RMSEs across the cross validation of each parameter setting
    print("Model {}:  mean rmse train: {} ({})".format(model_name,np.mean(cvscores_train), np.std(cvscores_train)))
    print("Model {}:  mean rmse test: {} ({})".format(model_name,np.mean(cvscores_test), np.std(cvscores_test)))
