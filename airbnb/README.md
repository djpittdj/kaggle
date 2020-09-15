[Kaggle Airbnb New User Bookings](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/overview)

# Preparation of data
## "airbnbNDCG" evaluation function
* it's written in C++ for xgboost source code
## for session log data
* calculate the total and median time a user spent on the Airbnb website by action, action type and action detail
* transform the time by log1p
## for user data
* transform date-related features
* create dummy variables for categorical features
## combine data
* combine session data with user data to generate a dataset with over 700 features

# Model fitting
* use several algorithms including XGBoost,  ExtraTrees from Scikit-Learn and Deep learning module from H2O to create a first-layer predictions, then use these predictions as input to a XGBoost model to generate the final output

