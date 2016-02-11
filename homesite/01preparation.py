import pandas as pd
import numpy as np
import xgboost as xgb

def proc_df(df):
    df = df.drop(['PropertyField6', 'GeographicField10A'], axis=1)
    df['Field10'] = df['Field10'].apply(lambda x: x.replace(',', ''))
    df['GeographicField63'] = df['GeographicField63'].replace({' ':'N'})
    df['Original_Quote_Date'] = pd.to_datetime(df['Original_Quote_Date'])
    df['Year'] = df.Original_Quote_Date.dt.year
    df['Month'] = df.Original_Quote_Date.dt.month
    df['DayOfWeek'] = df.Original_Quote_Date.dt.dayofweek
    df['Day'] = df.Original_Quote_Date.dt.day

    df['PropertyField37'] = df['PropertyField37'].replace({' ':'N'})

    for field in ['PersonalField84', 'PropertyField29']:
        df[field] = df[field].fillna(-1)
    for field in ['PersonalField7', 'PropertyField3', 'PropertyField4', 'PropertyField30', 'PropertyField36', 'PropertyField38']:
        df[field] = df[field].fillna('N')
    for field in ['PropertyField5', 'PropertyField32', 'PropertyField34']:
        df[field] = df[field].fillna('Y')

    df['Field9'] = df['Field9']*1000.0

    return df

train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

train = proc_df(train)
test = proc_df(test)

combined = pd.concat((train, test), ignore_index=True)
combined['QuoteConversion_Flag'] = combined['QuoteConversion_Flag'].fillna(0)
combined = combined[train.columns]

for feature in combined.columns:
    if combined[feature].dtype == 'object':
        print feature
        vc = combined[feature].value_counts()
        combined[feature+'_c'] = combined[feature].replace(vc)
        combined[feature] = combined[feature].astype('category').cat.codes.astype('int')

for feature in ['Year', 'Month', 'DayOfWeek', 'Day']:
    print feature
    vc = combined[feature].value_counts()
    combined[feature+'_c'] = combined[feature].replace(vc)
    combined[feature] = combined[feature].astype('category').cat.codes.astype('int')

important_features = pd.read_csv('../py5/feature_importance.csv')
top_features = important_features['feature'].head(n=20).values
for i in range(top_features.size-1):
    feature_i = top_features[i]
    for j in range(i+1, top_features.size):
        feature_j = top_features[j]
        if combined[feature_i].unique().size<50 and combined[feature_j].unique().size<50:
            new_feature_name = feature_i + str(':') + feature_j
            print new_feature_name
            combined[new_feature_name] = combined[feature_i].astype('str') + str(':') + combined[feature_j].astype('str')
            vc = combined[new_feature_name].value_counts()
            combined[new_feature_name+'_c'] = combined[new_feature_name].replace(vc)
            combined[new_feature_name] = combined[new_feature_name].astype('category').cat.codes.astype('int')

train = combined.loc[train.index]
test_index = combined.index.drop(train.index)
test = combined.loc[test_index]

train.to_csv('train01.csv', index=False)
test.to_csv('test01.csv', index=False)
features = train.columns
features = features.drop(['Original_Quote_Date', 'QuoteConversion_Flag', 'QuoteNumber'])
mat_train = xgb.DMatrix(data=train[features], label=train.QuoteConversion_Flag)
mat_train.save_binary('train01.DMatrix')
