from __future__ import division
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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

n_folds = 10
skf = list(StratifiedKFold(y, n_folds))
clfs = [XGBClassifier(n_estimators=4920,
                      learning_rate=0.01,
                      max_depth=6, 
                      subsample=1.0, 
                      colsample_bytree=0.4,
                      min_child_weight=5,
                      nthread=-1),
        ExtraTreesClassifier(n_estimators=2000, n_jobs=-1)]

dataset_train = np.zeros((X.shape[0], len(clfs)))
dataset_test = np.zeros((X_test.shape[0], len(clfs)))

n = 10
result_arr = np.zeros((X_test.shape[0], n))
for i in range(n):
    rand = np.random.randint(0,10000)
    print "i", i, rand
    
    for j, clf in enumerate(clfs):
        print j, clf
        if j == 0:
            clf.set_params(seed=rand)
        else:
            clf.set_params(random_state=rand)
        dataset_test_j = np.zeros((X_test.shape[0], len(skf)))
        for f, (train_index, val_index) in enumerate(skf):
            print "Fold", f
            X_train = X[train_index]
            y_train = y[train_index]
            X_val = X[val_index]
            y_val = y[val_index]
            clf.fit(X_train, y_train)
            dataset_train[val_index, j] = clf.predict_proba(X_val)[:,1]
            dataset_test_j[:, f] = clf.predict_proba(X_test)[:,1]
        dataset_test[:,j] = dataset_test_j.mean(1)

    clf = LogisticRegression()
    clf.fit(dataset_train, y)
    y_submission = clf.predict_proba(dataset_test)[:,1]

    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    result_arr[:, i] = y_submission
result = result_arr.mean(1)

print "Saving Results"
submission = pd.DataFrame({'QuoteNumber':test.QuoteNumber})
submission['QuoteConversion_Flag'] = result
submission.to_csv('sub_blend_ave_py7.csv', index=False)

