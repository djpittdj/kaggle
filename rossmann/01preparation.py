import pandas as pd
from pandas import DataFrame
import numpy as np
import xgboost as xgb

def proc_df(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df.Date.dt.year.astype(float)
    df['Month'] = df.Date.dt.month.astype(float)
    df['Day'] = df.Date.dt.day.astype(float)
    df['WeekOfYear'] = df.Date.dt.weekofyear.astype(float)
    df['StateHoliday'] = df['StateHoliday'].replace({'0':0, 'a':1, 'b':2, 'c':3})
    df['CompetitionOpen'] = 12 * (df.Year - df.CompetitionOpenSinceYear) + \
                            (df.Month - df.CompetitionOpenSinceMonth)
    df['Promo2Open'] = 12 * (df.Year - df.Promo2SinceYear) + \
                            (df.WeekOfYear - df.Promo2SinceWeek) / 4.0
    df['Promo2Open'] = df.Promo2Open.apply(lambda x: x if x<15000 else 0.0)
    df['OpenShift'] = df.Open.shift(-1)
    df['SchoolHolidayShift'] = df.SchoolHoliday.shift(-1)
    df['WeekStart'] = (df.Date.dt.dayofweek == 1).astype(int)
    df['WeekEnd'] = (df.Date.dt.dayofweek == 6).astype(int)
    df['MonthStart'] = df.Date.dt.is_month_start.astype(int)
    df['MonthEnd'] = df.Date.dt.is_month_end.astype(int)
    df['QuarterStart'] = df.Date.dt.is_quarter_start.astype(int)
    df['QuarterEnd'] = df.Date.dt.is_quarter_end.astype(int)

    return df

train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')
test.fillna(0, inplace=True)
store = pd.read_csv('../store.csv')
stores_states = pd.read_csv('../stores_states.csv')
stores_states = pd.concat((stores_states, pd.get_dummies(stores_states['State'], prefix='State')), axis=1)
store = pd.merge(store, stores_states)

store = pd.concat((store, pd.get_dummies(store['StoreType'], prefix='StoreType')), axis=1)
store['StoreType'] = store['StoreType'].replace({'a':1, 'b':2, 'c':3, 'd':4})

store = pd.concat((store, pd.get_dummies(store['Assortment'], prefix='Assortment')), axis=1)
store['Assortment'] = store['Assortment'].replace({'a':1, 'b':2, 'c':3})

store['PromoInterval'] = store['PromoInterval'].replace({'Mar,Jun,Sept,Dec':1, 'Feb,May,Aug,Nov':2, 'Jan,Apr,Jul,Oct':3})

# deal with CompetitionDistance
comp_dist = store.CompetitionDistance
comp_dist.fillna(comp_dist.median(), inplace=True)
comp_dist = np.log(comp_dist)
store['CompetitionDistance'] = comp_dist

competition_since_year = store.CompetitionOpenSinceYear
competition_since_year = competition_since_year.fillna(competition_since_year.median())
competition_since_year = competition_since_year.astype(int)
store['CompetitionOpenSinceYear'] = competition_since_year
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(1)

# map store to integer, based on mean sales per store
sales_store = train[['Sales', 'Store']].groupby('Store').mean().sort_values(by='Sales')
sales_store['StoreID'] = range(sales_store.shape[0])
sales_store.drop('Sales', inplace=True, axis=1)
mapping_dict = sales_store.to_dict()['StoreID']
store['StoreID'] = store['Store'].replace(mapping_dict)

# the only NAN values are for Promo2SinceWeek, Promo2SinceYear & PromoInterval
store.fillna(0, inplace=True)

train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

# map state to integer, based on mean sales per store
sales_state = train[['Sales', 'State']].groupby('State').mean().sort_values(by='Sales')
sales_state['StateID'] = range(sales_state.shape[0])
sales_state.drop('Sales', inplace=True, axis=1)
mapping_dict = sales_state.to_dict()['StateID']
train['StateID'] = train['State'].replace(mapping_dict)
test['StateID'] = test['State'].replace(mapping_dict)

train = proc_df(train)
test = proc_df(test)
test['OpenShift'] = test.OpenShift.fillna(1)
test['SchoolHolidayShift'] = test.SchoolHolidayShift.fillna(0)

sales_store_weekly13 = train.loc[train.Year==2013, ['Sales', 'Store', 'WeekOfYear']].groupby(['Store', 'WeekOfYear']).mean()
sales_store_weekly13.columns = ['WeeklySales13']
sales_store_weekly13['LogWeeklySales13'] = sales_store_weekly13['WeeklySales13'].apply(np.log1p)

for (i, j) in zip(['Store', 'WeekOfYear'], [0,1]):
    sales_store_weekly13[i] = sales_store_weekly13.index.get_level_values(j)

train = pd.merge(train, sales_store_weekly13, on=['Store', 'WeekOfYear'])
test = pd.merge(test, sales_store_weekly13, on=['Store', 'WeekOfYear'])

google_trend = pd.read_csv('../Wolfanger_mod/Rossmann_DE.csv', header=None, names=['Duration', 'Trend'])
google_trend['BeginDate'] = pd.to_datetime(google_trend.Duration.apply(lambda x: x.split(' - ')[0]))
google_trend['Year'] = google_trend.BeginDate.dt.year
google_trend['WeekOfYear'] = google_trend.BeginDate.dt.weekofyear
google_trend = google_trend.drop(['Duration', 'BeginDate'], axis=1)
train = pd.merge(train, google_trend, on=['Year', 'WeekOfYear'])
test = pd.merge(test, google_trend, on=['Year', 'WeekOfYear'])

for state_name in ['BE', 'BW', 'BY', 'HE', 'HH', 'NI', 'NW', 'RP', 'SH', 'SL', 'SN', 'ST', 'TH']:
    google_trend = pd.read_csv('../Wolfanger_mod/Rossmann_DE_%s.csv' % state_name, header=None, names=['Duration', 'Trend_%s' % state_name])
    google_trend['BeginDate'] = pd.to_datetime(google_trend.Duration.apply(lambda x: x.split(' - ')[0]))
    google_trend['Year'] = google_trend.BeginDate.dt.year
    google_trend['WeekOfYear'] = google_trend.BeginDate.dt.weekofyear
    google_trend = google_trend.drop(['Duration', 'BeginDate'], axis=1)
    train = pd.merge(train, google_trend, on=['Year', 'WeekOfYear'])
    test = pd.merge(test, google_trend, on=['Year', 'WeekOfYear'])

train = train[train.Open == 1]
train = train[train.Sales != 0]
train = train.reset_index(drop=True)
train['LogSales'] = train.Sales.apply(np.log1p)

train.to_csv('train01.csv', index=False)
test.to_csv('test01.csv', index=False)
