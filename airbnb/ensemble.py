import pandas as pd
import numpy as np
import xgboost as xgb

train01 = pd.read_csv('../py14/train10.csv')
country_destination_int = train01['country_destination_int'].values
del train01

train01_xgb_14_10 = pd.read_csv('oof_train_xgb_py14_10.csv')
train01_xgb_14_10.columns = map(lambda x: x if x=='id' else x+'_01', train01_xgb_14_10.columns.values)
train02_et_14_10 = pd.read_csv('oof_train_extratrees_py14_10.csv')
train02_et_14_10.columns = map(lambda x: x if x=='id' else x+'_02', train02_et_14_10.columns.values)
train07_xgb_14_26 = pd.read_csv('oof_train_xgb_py14_26.csv')
train07_xgb_14_26.columns = map(lambda x: x if x=='id' else x+'_07', train07_xgb_14_26.columns.values)
train08_xgb_14_19 = pd.read_csv('oof_train_xgb_py14_19.csv')
train08_xgb_14_19.columns = map(lambda x: x if x=='id' else x+'_08', train08_xgb_14_19.columns.values)
train09_xgb_14_27 = pd.read_csv('oof_train_xgb_py14_27.csv')
train09_xgb_14_27.columns = map(lambda x: x if x=='id' else x+'_09', train09_xgb_14_27.columns.values)
train11_h2o_DL_14_10 = pd.read_csv('oof_train_h2o_deeplearning_py14_10.csv')
train11_h2o_DL_14_10.columns = map(lambda x: x if x=='id' else x+'_11', train11_h2o_DL_14_10.columns.values)
train12_h2o_DL_14_26 = pd.read_csv('oof_train_h2o_deeplearning_py14_26.csv')
train12_h2o_DL_14_26.columns = map(lambda x: x if x=='id' else x+'_12', train12_h2o_DL_14_26.columns.values)
train13_h2o_DL_14_27 = pd.read_csv('oof_train_h2o_deeplearning_py14_27.csv')
train13_h2o_DL_14_27.columns = map(lambda x: x if x=='id' else x+'_13', train13_h2o_DL_14_27.columns.values)

test01_xgb_14_10 = pd.read_csv('oof_test_xgb_py14_10.csv')
test01_xgb_14_10.columns = map(lambda x: x if x=='id' else x+'_01', test01_xgb_14_10.columns.values)
test02_et_14_10 = pd.read_csv('oof_test_extratrees_py14_10.csv')
test02_et_14_10.columns = map(lambda x: x if x=='id' else x+'_02', test02_et_14_10.columns.values)
test07_xgb_14_26 = pd.read_csv('oof_test_xgb_py14_26.csv')
test07_xgb_14_26.columns = map(lambda x: x if x=='id' else x+'_07', test07_xgb_14_26.columns.values)
test08_xgb_14_19 = pd.read_csv('oof_test_xgb_py14_19.csv')
test08_xgb_14_19.columns = map(lambda x: x if x=='id' else x+'_08', test08_xgb_14_19.columns.values)
test09_xgb_14_27 = pd.read_csv('oof_test_xgb_py14_27.csv')
test09_xgb_14_27.columns = map(lambda x: x if x=='id' else x+'_09', test09_xgb_14_27.columns.values)
test11_h2o_DL_14_10 = pd.read_csv('oof_test_h2o_deeplearning_py14_10.csv')
test11_h2o_DL_14_10.columns = map(lambda x: x if x=='id' else x+'_11', test11_h2o_DL_14_10.columns.values)
test12_h2o_DL_14_26 = pd.read_csv('oof_test_h2o_deeplearning_py14_26.csv')
test12_h2o_DL_14_26.columns = map(lambda x: x if x=='id' else x+'_12', test12_h2o_DL_14_26.columns.values)
test13_h2o_DL_14_27 = pd.read_csv('oof_test_h2o_deeplearning_py14_27.csv')
test13_h2o_DL_14_27.columns = map(lambda x: x if x=='id' else x+'_13', test13_h2o_DL_14_27.columns.values)

train = pd.merge(train01_xgb_14_10, train02_et_14_10, on='id')
train = pd.merge(train, train07_xgb_14_26, on='id')
train = pd.merge(train, train08_xgb_14_19, on='id')
train = pd.merge(train, train09_xgb_14_27, on='id')
train = pd.merge(train, train11_h2o_DL_14_10, on='id')
train = pd.merge(train, train12_h2o_DL_14_26, on='id')
train = pd.merge(train, train13_h2o_DL_14_27, on='id')

test = pd.merge(test01_xgb_14_10, test02_et_14_10, on='id')
test = pd.merge(test, test07_xgb_14_26, on='id')
test = pd.merge(test, test08_xgb_14_19, on='id')
test = pd.merge(test, test09_xgb_14_27, on='id')
test = pd.merge(test, test11_h2o_DL_14_10, on='id')
test = pd.merge(test, test12_h2o_DL_14_26, on='id')
test = pd.merge(test, test13_h2o_DL_14_27, on='id')

params = {'eta': 0.01,
          'objective':'multi:softprob',
          'max_depth':6,
          'subsample':1.0,
          'colsample_bytree':1.0,
          'min_child_weight':1,
          'num_class':12,
          'eval_metric':'airbnbNDCG'}
nrounds = 100

features = train.columns
features = features.drop('id')

dtrain = xgb.DMatrix(data=train[features].values, label=country_destination_int)
watchlist = [(dtrain, 'train')]
xgb.cv(params=params, dtrain=dtrain, num_boost_round=nrounds, nfold=10)

