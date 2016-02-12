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

train = pd.read_csv('train10.csv')
test = pd.read_csv('test10.csv')
country_mapping = pd.read_csv('country_mapping.csv')
country_mapping.index = country_mapping.country_int
country_mapping = country_mapping.drop('country_int', axis=1)
country_mapping_inverse_dict = country_mapping.to_dict()['country']
features = train.columns
features = features.drop(['id', 'num_actions', 'country_destination', 'country_destination_int'])
print 'features:', features.values
X_train = train[features].values
y_train = train['country_destination_int'].values
mat_train = xgb.DMatrix(data=X_train, label=y_train)
X_test = test[features].values
mat_test = xgb.DMatrix(data=X_test)

params = {'objective': "multi:softprob",
          'num_class': 12,
          'eta': 0.005,
          'max_depth': 6,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'min_child_weight': 1,
          'eval_metric': 'airbnbNDCG',
          'silent': 1}
nrounds = 1200

watchlist = [(mat_train, 'train')]
model = xgb.train(params=params, dtrain=mat_train, num_boost_round=nrounds, evals=watchlist, verbose_eval=10)
y_hat = model.predict(mat_test)
y_hat_top = np.argsort(y_hat)[:,[-1,-2,-3,-4,-5]]

id_test = test['id']
df_y_hat = pd.DataFrame(y_hat, index=id_test)
df_y_hat.to_csv('y_hat_pred19.csv')
ids = []
cts = []
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    for j in y_hat_top[i]:
        cts.append(country_mapping_inverse_dict[j])

sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
# sub.to_csv('sub_pred_py14.csv',index=False)

# direct append py10E predictions
sub1 = pd.read_csv('../py10E/sub_py10E_pred.csv')
sub1_id_lst = sub1.id.unique()
sub_id_lst = sub.id.unique()
diff_id_lst = np.setdiff1d(sub1_id_lst, sub_id_lst)
merged = sub.copy()
for id in diff_id_lst:
    merged = pd.concat((merged, sub1.loc[sub1.id==id]))
merged = merged.reset_index(drop=True)
merged.to_csv('sub_py14_pred19_with_py10E.csv', index=False)

# feature importance
create_feature_map(features)
feature_importance = model.get_fscore(fmap='xgb_features.fmap')
feature_importance = sorted(feature_importance.items(), key=operator.itemgetter(1))
feature_importance = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
feature_importance = feature_importance.sort_values(by='importance', ascending=False)
feature_importance.index = feature_importance.feature
feature_importance = feature_importance.drop('feature',axis=1)
feature_importance .to_csv('feature_importance.csv')
