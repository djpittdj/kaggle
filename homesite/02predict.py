import pandas as pd
import numpy as np
import xgboost as xgb
import operator

# from cast42 script
def create_feature_map(features):
    outfile = open('xgb_features.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

train = pd.read_csv('train01.csv')
test = pd.read_csv('test01.csv')

features = train.columns
features = features.drop(['Original_Quote_Date', 'QuoteConversion_Flag', 'QuoteNumber'])

X_train = train[features].values
y_train = train.QuoteConversion_Flag.values
X_test = test[features].values

dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test)

watchlist = [(dtrain, 'train')]
params = {'objective' : "binary:logistic", 
          'eta' : 0.05,
          'max_depth' : 6,
          'subsample' : 1.0,
          'colsample_bytree' : 0.4,
          'min_child_weight': 5,
          'eval_metric':'auc',
          'silent' : 1,
          }

model = xgb.train(num_boost_round = 600, params = params, dtrain = dtrain, evals=watchlist, verbose_eval=10)

submission = pd.DataFrame({'QuoteNumber':test.QuoteNumber})
submission['QuoteConversion_Flag'] = model.predict(dtest)
submission.to_csv('sub_pred_py7.csv', index=False)

# feature importance
create_feature_map(features)
feature_importance = model.get_fscore(fmap='xgb_features.fmap')
feature_importance = sorted(feature_importance.items(), key=operator.itemgetter(1))
feature_importance = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
feature_importance = feature_importance.sort_values(by='importance', ascending=False)
feature_importance.index = feature_importance.feature
feature_importance = feature_importance.drop('feature',axis=1)
feature_importance.to_csv('feature_importance.csv')

