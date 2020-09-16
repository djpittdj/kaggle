from __future__ import division
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

np.random.seed(0) # seed to shuffle the train set

train = pd.read_csv('train01.csv')
test = pd.read_csv('test01.csv')

features = train.columns
features = features.drop(['Original_Quote_Date', 'QuoteConversion_Flag', 'QuoteNumber'])
print "features: ", features

X = train[features].values
y = train.QuoteConversion_Flag.values
X_test = test[features].values

clf =  XGBClassifier(n_estimators=4920,
                      learning_rate=0.01,
                      max_depth=6, 
                      subsample=1.0, 
                      colsample_bytree=0.4,
                      min_child_weight=5,
                      nthread=-1)

print "Creating train and test sets for blending."

n = 10
result_arr = np.zeros((X_test.shape[0], n))
for i in range(n):
    rand = np.random.randint(0,10000)
    print "i", i, rand
    clf.set_params(seed=rand)
    clf.fit(X, y)
    result_arr[:, i] = clf.predict_proba(X_test)[:,1]
result = result_arr.mean(1)

print "Saving Results"
submission = pd.DataFrame({'QuoteNumber':test.QuoteNumber})
submission['QuoteConversion_Flag'] = result
submission.to_csv('sub_ave_py7.csv', index=False)

