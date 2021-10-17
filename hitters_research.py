
################################################
# End-to-End Hitters Machine Learning Pipeline I
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function

################################################
# Libraries
################################################
# ignore warnings

import warnings

warnings.simplefilter(action='ignore', category=Warning)
import joblib
from helpers.data_prep import *
from helpers.eda import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

# linear models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# non-linear models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

############################################
# Let's get the dataset
############################################

df = pd.read_csv("hafta_8/hitters.csv")
df.head()

############################################
# Data Understanding
############################################

df.info()
df.shape  # There are 322 observation and 20 variables.
df.isnull().sum()  # There are missing values in Salary columns.
check_df(df)

# Descriptive Analysis
# When I examine the dataset, we see that there is
# a difference between the mean and median values of the Assist variable.
# This difference is also supported by the standard deviation.
df.describe([0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T

# Specifying variable types
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
cat_cols, num_cols, cat_but_car

# Visualization of categorical variables
# plot shows there are around 140 count for Player's of League A and around 125 for League

for col in cat_cols:
    cat_summary(df, col, plot=True)

# Examination of numerical variables
df[num_cols].describe().T

for col in num_cols:
    num_summary(df, col, plot=True)

###########################################
# Corr Between Variables
###########################################
# Correlation of numerical variables with each other
correlation_matrix(df, num_cols)
# We can make observations such as, as AtBat have high correlation with Runs

# Examination of categorical variables with Target
for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)

###########################################
# Missing Value Understanding
###########################################
# Random Forests and LightGbm has an effective method for estimating
# missing data and maintains accuracy when a large proportion of the
# data are missing.I will still throw out the small amount of outlier in the data.
# We removed in the missing observations with dropna method:
msno.bar(df)


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    freq_na = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([freq_na, np.round(ratio, 2)], axis=1, keys=['freq_na', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


missing_values_table(df, True)

df.dropna(inplace=True)


###########################################
# OUTLIER ANALYSIS
###########################################
def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name, q1=0.10, q3=0.90):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# Outlier' value
outlier_thresholds(df, df.columns)

# There seems to be no outliers compared to 0.1 and 0.9 on target
check_outlier(df, "Salary")

# Here, 3 outlier observations in all variables in quartiles of 0.1 and 0.9 are accessed
# As we can see below, there are outliers in the dataset
for col in num_cols:
    print(col, check_outlier(df, col))

# We replace with thresholds
for col in num_cols:
    replace_with_thresholds(df, col)

# control for outliers
for col in num_cols:
    print(col, check_outlier(df, col))

# Examined detailed outlier analysis for the target variable
df["Salary"].describe([0.05, 0.25, 0.45, 0.50, 0.65, 0.85, 0.95, 0.99]).T

sns.boxplot(x=df["Salary"])
plt.show()

# remove salary bigger than up limit
q3 = 0.90
salary_up = int(df["Salary"].quantile(q3))
df = df[(df["Salary"] < salary_up)]

###########################################
# another method I tried for salary but not works

# low, up = outlier_thresholds(df, "Salary", 0.25, 0.75)
# df[~((df["Salary"] < low) | (df["Salary"] > up))].shape (263, 20)
# df.loc[(df["Salary"] > up), "Salary"] = up
# df = df[(df["Salary"] < 900) | (df["Salary"].isnull())]
###########################################

for col in cat_cols:
    cat_summary(df, col)

###########################################
# Feature Engineering
###########################################
# New variables were created with the most appropriate variables according to their proportions.

df["new_Hits/CHits"] = df["Hits"] / df["CHits"]
df["new_OrtCHits"] = df["CHits"] / df["Years"]
df["new_OrtCHmRun"] = df["CHmRun"] / df["Years"]
df["new_OrtCruns"] = df["CRuns"] / df["Years"]
df["new_OrtCRBI"] = df["CRBI"] / df["Years"]
df["new_OrtCWalks"] = df["CWalks"] / df["Years"]


df["New_Average"] = df["Hits"] / df["AtBat"]
df['new_PutOutsYears'] = df['PutOuts'] * df['Years']
df["new_RBIWalksRatio"] = df["RBI"] / df["Walks"]
df["New_CHmRunCAtBatRatio"] = df["CHmRun"] / df["CAtBat"]
df["New_BattingAverage"] = df["CHits"] / df["CAtBat"]
df.dropna(inplace=True)

###########################################
# Variables that I added but have low feature importance

# df["new_CWalks*CRuns"] = df["CWalks"] * df["CRuns"]
# df["new_RBIWalks"] = df["RBI"] * df["Walks"]
# df["new_CRunsCHmRun"] = df["CHmRun"] / df["CRuns"]
# df["New_CRunsWalks"] = df["CRuns"] * df["Walks"]
# df['New_PutOutsAssists'] = df['PutOuts'] * df['Assists']
###########################################

###########################################
# LABEL ENCODING
###########################################
# Binary Encoding
# label encoding of categorical features (League, Division, NewLeague) with two class
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
for col in binary_cols:
    labelencoder = LabelEncoder()
    df[col] = labelencoder.fit_transform(df[col])

# One-Hot Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(df)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=False)
    return dataframe


df = one_hot_encoder(df, cat_cols)

df.head()
df.dropna(inplace=True)

####################################################
# Feature importances and Scaler Transform
####################################################

y = df["Salary"]
X = df.drop(["Salary"], axis=1)
df.shape

# list feature importances for a regressor model like LGBM
pre_model = LGBMRegressor().fit(X, y)
feature_imp = pd.DataFrame({'Feature': X.columns, 'Value': pre_model.feature_importances_})
feature_imp.sort_values("Value", ascending=False)

# Scaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

######################################################
# Base Models
######################################################
models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          # ("CatBoost", CatBoostRegressor(verbose=False))
          ]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

"""
RMSE: 208.2738 (LR) 
RMSE: 202.011 (Ridge) 
RMSE: 199.6219 (Lasso) 
RMSE: 220.5961 (ElasticNet) 
RMSE: 188.5584 (KNN) 
RMSE: 209.4031 (CART) 
RMSE: 155.0623 (RF) 
RMSE: 268.7786 (SVR) 
RMSE: 158.4295 (GBM) 
RMSE: 163.5608 (XGBoost) 
RMSE: 162.2267 (LightGBM) 
 """
######################################################
# Automated Hyperparameter Optimization
######################################################

cart_params = {'max_depth': range(1, 20),  # ne kadar dallanacak
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [3, 5, 8, 15, 20],
             "n_estimators": [600, 650, 1000]}

xgboost_params = {"learning_rate": [0.1, 0.01, 0.01],
                  "max_depth": [5, 8, 12, 20],
                  "n_estimators": [100, 200, 300, 500],
                  "colsample_bytree": [0.5, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.001, 0.01, 0.1, 0.001],
                   "n_estimators": [250, 300, 500, 1500, 2500,3000],
                   "colsample_bytree": [0.1, 0.3, 0.5, 0.7, 1]}

regressors = [("CART", DecisionTreeRegressor(), cart_params),
              ("RF", RandomForestRegressor(), rf_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params)]

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

"""
########## CART ##########
RMSE: 200.3846 (CART) 
RMSE (After): 178.1955 (CART) 
CART best params: {'max_depth': 3, 'min_samples_split': 21}

########## RF ##########
RMSE: 156.2336 (RF) 
RMSE (After): 152.2343 (RF) 
RF best params: {'max_depth': 15, 'max_features': 7, 'min_samples_split': 5, 'n_estimators': 1000}

########## XGBoost ##########
RMSE: 163.5608 (XGBoost) 
RMSE (After): 160.6244 (XGBoost) 
XGBoost best params: {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500}

########## LightGBM ##########
RMSE: 162.2267 (LightGBM) 
RMSE (After): 156.5638 (LightGBM) 
LightGBM best params: {'colsample_bytree': 0.1, 'learning_rate': 0.01, 'n_estimators': 500}
"""

######################################################
# Stacking & Ensemble Learning
######################################################

voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('LightGBM', best_models["LightGBM"])])

voting_reg.fit(X, y)

np.mean(np.sqrt(-cross_val_score(voting_reg,
                                 X, y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

"""
RMSE = 153.54219398316238
"""

######################################################
# Prediction for a New Observation
######################################################

X.columns
random_user = X.sample(1, random_state=45)  # 247 index y[y.index == 247] --> 560
voting_reg.predict(random_user)


###########################################
# Functionalization
###########################################

def hitters_data_prep(dataframe):

    ############ Specifying variable types ############

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)

    ############ We replace with thresholds ############

    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    ############ remove salary bigger than up limit ############

    q3 = 0.90
    salary_up = int(dataframe["Salary"].quantile(q3))
    dataframe = dataframe[(dataframe["Salary"] < salary_up)]

    ############ Feature engineering  ############

    # New variables were created with the most appropriate variables according to their proportions.
    dataframe["new_Hits/CHits"] = dataframe["Hits"] / dataframe["CHits"]
    dataframe["new_OrtCHits"] = dataframe["CHits"] / dataframe["Years"]
    dataframe["new_OrtCHmRun"] = dataframe["CHmRun"] / dataframe["Years"]
    dataframe["new_OrtCruns"] = dataframe["CRuns"] / dataframe["Years"]
    dataframe["new_OrtCRBI"] = dataframe["CRBI"] / dataframe["Years"]
    dataframe["new_OrtCWalks"] = dataframe["CWalks"] / dataframe["Years"]

    dataframe["New_Average"] = dataframe["Hits"] / dataframe["AtBat"]
    dataframe['new_PutOutsYears'] = dataframe['PutOuts'] * dataframe['Years']
    dataframe["new_RBIWalksRatio"] = dataframe["RBI"] / dataframe["Walks"]
    dataframe["New_CHmRunCAtBatRatio"] = dataframe["CHmRun"] / dataframe["CAtBat"]
    dataframe["New_BattingAverage"] = dataframe["CHits"] / dataframe["CAtBat"]
    dataframe.dropna(inplace=True)

    ############ Binary Encoding ############
    # label encoding of categorical features (League, Division, NewLeague) with two class
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in
                   [int, float] and dataframe[col].nunique() == 2]

    for col in binary_cols:
        labelencoder = LabelEncoder()
        dataframe[col] = labelencoder.fit_transform(dataframe[col])

    ############ One-Hot Encoding ############
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    dataframe = one_hot_encoder(dataframe, cat_cols)

    ############ MODEL ############

    y = dataframe["Salary"]
    X = dataframe.drop(["Salary"], axis=1)


    ############ Scaler ############
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

X, y = hitters_data_prep(df)
