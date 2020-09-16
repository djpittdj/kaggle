from __future__ import division
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

np.random.seed(0) # seed to shuffle the train set

train = pd.read_csv('train01.csv')
test = pd.read_csv('test01.csv')

features = train.columns
features = features.drop(['Original_Quote_Date', 'QuoteConversion_Flag', 'QuoteNumber'])
print("features: ", features)

X = train[features].values
y = train.QuoteConversion_Flag.values
X_test = test[features].values

# blend results from multiple models
n_folds = 10
skf = list(StratifiedKFold(y, n_folds))
clfs = [XGBClassifier(n_estimators=4920,
                      learning_rate=0.01,
                      max_depth=6, 
                      subsample=1.0, 
                      colsample_bytree=0.4,
                      min_child_weight=5,
                      nthread=-1),
        ExtraTreesClassifier(n_estimators=2000, n_jobs=-1)
        ]

print("Creating train and test sets for blending.")

dataset_train = np.zeros((X.shape[0], len(clfs)))
dataset_test = np.zeros((X_test.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    print(j, clf)
    dataset_test_j = np.zeros((X_test.shape[0], len(skf)))
    for i, (train_index, val_index) in enumerate(skf):
        print("Fold", i)
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]
        clf.fit(X_train, y_train)
        dataset_train[val_index, j] = clf.predict_proba(X_val)[:,1]
        dataset_test_j[:, i] = clf.predict_proba(X_test)[:,1]
    dataset_test[:,j] = dataset_test_j.mean(1)

pd.DataFrame(dataset_train).to_csv('blend_train.csv', index=False, header=False)
pd.DataFrame(dataset_test).to_csv('blend_test.csv', index=False, header=False)
print()
print("Blending.")
clf = LogisticRegression()
clf.fit(dataset_train, y)
y_submission = clf.predict_proba(dataset_test)[:,1]

print("Linear stretch of predictions to [0,1]")
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

print("Saving Results")
# np.savetxt(fname='sub.csv', X=y_submission, fmt='%0.9f')
submission = pd.DataFrame({'QuoteNumber':test.QuoteNumber})
submission['QuoteConversion_Flag'] = y_submission
submission.to_csv('sub_blend_py7.csv', index=False)

