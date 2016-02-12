import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import sys

# process sessions information
sessions = pd.read_csv('../sessions.csv')
sessions = sessions.loc[sessions.user_id.notnull()]
secs_elapsed = sessions['secs_elapsed']
sessions['secs_elapsed'] = secs_elapsed.fillna(secs_elapsed.median())
sessions['action'] = sessions['action'].fillna('show')
sessions['action_type'] = sessions['action_type'].fillna('-unknown-')
sessions['action_detail'] = sessions['action_detail'].fillna('-unknown-')
sessions = sessions.drop(['device_type'], axis=1)

# denoise session data
features = ['action', 'action_detail']
for feature in features:
    vc_perc = sessions[feature].value_counts(normalize=True)
    vc_others = vc_perc[vc_perc.cumsum()>0.999].index
    sessions[feature] = sessions[feature].apply(lambda x: 'Other_'+feature if x in vc_others else x)

# action
action_group = sessions[['user_id', 'action', 'secs_elapsed']].groupby(['user_id', 'action'], sort=False)

action_total_time_tab = action_group.sum().apply(np.log1p)
action_total_time_tab_unstack1 = action_total_time_tab.unstack(1)
action_total_time_tab_unstack1 = action_total_time_tab_unstack1.fillna(0)
action_total_time_tab_unstack1.columns.name = None
action_total_time_tab_unstack1.columns = action_total_time_tab_unstack1.columns.droplevel(0)
action_total_time_tab_unstack1.columns = action_total_time_tab_unstack1.columns.name + str(':') + action_total_time_tab_unstack1.columns + '_total_t'

action_median_time_tab = action_group.median().apply(np.log1p)
action_median_time_tab_unstack1 = action_median_time_tab.unstack(1)
action_median_time_tab_unstack1 = action_median_time_tab_unstack1.fillna(0)
action_median_time_tab_unstack1.columns.name = None
action_median_time_tab_unstack1.columns = action_median_time_tab_unstack1.columns.droplevel(0)
action_median_time_tab_unstack1.columns = action_median_time_tab_unstack1.columns.name + str(':') + action_median_time_tab_unstack1.columns + str('_median_t')

# action_type
action_type_group = sessions[['user_id', 'action_type', 'secs_elapsed']].groupby(['user_id', 'action_type'], sort=False)

action_type_total_time_tab = action_type_group.sum().apply(np.log1p)
action_type_total_time_tab_unstack1 = action_type_total_time_tab.unstack(1)
action_type_total_time_tab_unstack1 = action_type_total_time_tab_unstack1.fillna(0)
action_type_total_time_tab_unstack1.columns.name = None
action_type_total_time_tab_unstack1.columns = action_type_total_time_tab_unstack1.columns.droplevel(0)
action_type_total_time_tab_unstack1.columns = action_type_total_time_tab_unstack1.columns.name + str(':') + action_type_total_time_tab_unstack1.columns + '_total_t'

action_type_median_time_tab = action_type_group.median().apply(np.log1p)
action_type_median_time_tab_unstack1 = action_type_median_time_tab.unstack(1)
action_type_median_time_tab_unstack1 = action_type_median_time_tab_unstack1.fillna(0)
action_type_median_time_tab_unstack1.columns.name = None
action_type_median_time_tab_unstack1.columns = action_type_median_time_tab_unstack1.columns.droplevel(0)
action_type_median_time_tab_unstack1.columns = action_type_median_time_tab_unstack1.columns.name + str(':') + action_type_median_time_tab_unstack1.columns + str('_median_t')

# action_detail
action_detail_group = sessions[['user_id', 'action_detail', 'secs_elapsed']].groupby(['user_id', 'action_detail'], sort=False)

action_detail_total_time_tab = action_detail_group.sum().apply(np.log1p)
action_detail_total_time_tab_unstack1 = action_detail_total_time_tab.unstack(1)
action_detail_total_time_tab_unstack1 = action_detail_total_time_tab_unstack1.fillna(0)
action_detail_total_time_tab_unstack1.columns.name = None
action_detail_total_time_tab_unstack1.columns = action_detail_total_time_tab_unstack1.columns.droplevel(0)
action_detail_total_time_tab_unstack1.columns = action_detail_total_time_tab_unstack1.columns.name + str(':') + action_detail_total_time_tab_unstack1.columns + '_total_t'

action_detail_median_time_tab = action_detail_group.median().apply(np.log1p)
action_detail_median_time_tab_unstack1 = action_detail_median_time_tab.unstack(1)
action_detail_median_time_tab_unstack1 = action_detail_median_time_tab_unstack1.fillna(0)
action_detail_median_time_tab_unstack1.columns.name = None
action_detail_median_time_tab_unstack1.columns = action_detail_median_time_tab_unstack1.columns.droplevel(0)
action_detail_median_time_tab_unstack1.columns = action_detail_median_time_tab_unstack1.columns.name + str(':') + action_detail_median_time_tab_unstack1.columns + str('_median_t')

# merge with number of actions
sessions_num_actions = sessions[['user_id', 'action']].groupby('user_id', sort=False).count()
sessions_num_actions.columns = ['num_actions']
sessions_num_actions['num_actions_log'] = sessions_num_actions['num_actions'].apply(np.log)

sessions_agg = pd.concat((action_total_time_tab_unstack1, action_median_time_tab_unstack1, 
    action_type_total_time_tab_unstack1, action_type_median_time_tab_unstack1, 
    action_detail_total_time_tab_unstack1, action_detail_median_time_tab_unstack1, 
    sessions_num_actions), axis=1)

# user information
train_users = pd.read_csv('../train_users_2.csv')
test_users = pd.read_csv('../test_users.csv')
test_users['country_destination'] = 'NDF'
all_users = pd.concat((train_users, test_users), ignore_index=True)
all_users = all_users.drop(['date_first_booking'], axis=1)
all_users = all_users.fillna(-1)
label_encoder = LabelEncoder()
all_users['country_destination_int'] = label_encoder.fit_transform(all_users['country_destination'].values)
mapping = pd.DataFrame({'country': label_encoder.classes_, 'country_int': range(label_encoder.classes_.size)})
mapping.to_csv('country_mapping.csv', index=False)

all_users['date_account_created'] = pd.to_datetime(all_users['date_account_created'])
all_users['date_account_created_year'] = all_users['date_account_created'].dt.year
all_users['date_account_created_month'] = all_users['date_account_created'].dt.month
all_users['date_account_created_day'] = all_users['date_account_created'].dt.day
all_users = all_users.drop('date_account_created', axis=1)

all_users['timestamp_first_active'] = all_users['timestamp_first_active'].apply(lambda x: str(x)[0:8])
all_users['timestamp_first_active'] = pd.to_datetime(all_users['timestamp_first_active'], format='%Y%m%d')
all_users['timestamp_first_active_year'] = all_users['timestamp_first_active'].dt.year
all_users['timestamp_first_active_month'] = all_users['timestamp_first_active'].dt.month
all_users['timestamp_first_active_day'] = all_users['timestamp_first_active'].dt.day
all_users = all_users.drop('timestamp_first_active', axis=1)

age = all_users.age.values
all_users['age'] = np.where(np.logical_or(age<14, age>100), -1, age)

obj_features = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for feature in obj_features:
    print feature
    all_users = pd.concat((all_users, pd.get_dummies(all_users[feature], prefix=feature)), axis=1)
    all_users = all_users.drop(feature, axis=1)

train_users = all_users.loc[train_users.index]
test_index = all_users.index.drop(train_users.index)
test_users = all_users.loc[test_index]
test_users = test_users.reset_index(drop=True)

# merge sessions and user information
train = pd.merge(sessions_agg, train_users, left_index='user_id', right_on='id')
test = pd.merge(sessions_agg, test_users, left_index='user_id', right_on='id')

features = train.columns
features = features.drop(['num_actions', 'id', 'country_destination', 'country_destination_int'])
mat_train = xgb.DMatrix(data=train[features].values, label=train.country_destination_int.values)
name = sys.argv[0][0:2]
mat_train.save_binary('train%s.DMatrix' % name)
train.to_csv('train%s.csv' % name, index=False)
test.to_csv('test%s.csv' % name, index=False)
