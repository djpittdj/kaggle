from __future__ import division
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
import pandas as pd
import utils
import sys

np.random.seed(0) # seed to shuffle the train set

train = pd.read_csv('../py14/train10.csv')
test = pd.read_csv('../py14/test10.csv')

country_mapping = pd.read_csv('../py14/country_mapping.csv')
country_mapping.index = country_mapping.country_int
country_mapping = country_mapping.drop('country_int', axis=1)
country_mapping_inverse_dict = country_mapping.to_dict()['country']
features = train.columns
features = features.drop(['id', 'num_actions', 'country_destination', 'country_destination_int'])

X = train[features].values
y = train['country_destination_int'].values

X_test = test[features].values
mat_test = xgb.DMatrix(data=X_test)

n_folds = 10
n_classes = np.unique(y).shape[0]

params = {'objective': "multi:softprob",
          'num_class': 12,
          'eta': 0.01,
          'max_depth': 6,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'min_child_weight': 1,
          'eval_metric': 'airbnbNDCG',
          'silent': 1}
nrounds = 1010

data_blend_train = np.zeros((X.shape[0], n_classes))
data_blend_test_mat = np.zeros((X_test.shape[0], n_classes))

skf = list(StratifiedKFold(y, n_folds, shuffle=True))
NDCG_scores = []
for ifold, (train_index, val_index) in enumerate(skf):
    X_train = X[train_index]
    y_train = y[train_index]
    X_val = X[val_index]
    y_val = y[val_index]
    mat_train = xgb.DMatrix(data=X_train, label=y_train)
    mat_val = xgb.DMatrix(data=X_val, label=y_val)
    watchlist = [(mat_train, 'train'), (mat_val, 'val')]
    model = xgb.train(params=params, dtrain=mat_train, num_boost_round=nrounds, evals=watchlist, verbose_eval=10)
    y_hat_val = model.predict(mat_val)
    this_NDCG_score = utils.calc_NDCG(y_val, y_hat_val)
    print 'Fold %i: %8.5f' % (ifold+1, this_NDCG_score)
    sys.stdout.flush()
    NDCG_scores.append(this_NDCG_score)
    data_blend_train[val_index, :] = y_hat_val
    data_blend_test_mat += model.predict(mat_test)
data_blend_test = data_blend_test_mat/n_folds
print 'Mean NDCG score:', np.array(NDCG_scores).mean()

df_blend_train = pd.DataFrame(data_blend_train, index=train['id'])
df_blend_test = pd.DataFrame(data_blend_test, index=test['id'])
df_blend_train.to_csv('01oof_train_xgb_py14_10.csv')
df_blend_test.to_csv('01oof_test_xgb_py14_10.csv')

