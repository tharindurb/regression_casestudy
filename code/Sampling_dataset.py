import pickle
import pandas as pd

# Reading the sales data file
file_name='../data/train.csv'
train = pd.read_csv(
    file_name, usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    parse_dates=["date"],
)

# Sampling the 2016 records of 'perishable' items
train= train[(train.date.isin(pd.date_range(start='2016-01-01', end='2016-12-31'))) & (train.perishable==1)].copy()

# Unique dates, stores, and items
u_dates = train.date.unique()
u_stores = train.store_nbr.unique()
u_items = train.item_nbr.unique()

print('Training datasets size before adding unavailable records: {}'.format(train.shape[0]))
print('Unique dates fill: {}'.format(u_dates.shape[0]))
print('Unique stores: {}'.format(u_stores.shape[0]))
print('Unique items: {}'.format(u_items.shape[0]))

train.set_index(['date', 'store_nbr', 'item_nbr'], inplace=True)

# Adding the unavailable records. Note: only on the training dataset
train = train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=['date','store_nbr','item_nbr']
    )
).reset_index()
train['unit_sales'].fillna(0, inplace=True)
train['onpromotion'].fillna(False, inplace=True)

print('Training datasets size after adding unavailable records:: {}'.format(train.shape[0]))

# Reading the store and item information data files
stores = pd.read_csv("../data/stores.csv")
items = pd.read_csv("../data/items.csv")

# Merging the store information to the training dataset using 'store_nbr' as the key
train = train.merge(right=stores, left_on='store_nbr', right_on='store_nbr')

# Merging the item information to the training dataset using 'item_nbr' as the key
train = train.merge(right=items, left_on='item_nbr', right_on='item_nbr')

# Adding date related features to the training data
train['month'] = train.date.dt.month
train['week_of_year'] = train.date.dt.weekofyear
train['day_of_week'] = train.date.dt.dayofweek

print(train.tail(n=10))

# Saving dataset as a Pickle dump
pickle.dump(train, open( "../data/train_perishable_zerofilled_2016.p", "wb"))