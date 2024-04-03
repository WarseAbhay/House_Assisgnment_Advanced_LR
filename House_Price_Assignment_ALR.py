#Importing necessary packages for modelling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error
from sklearn import metrics

#Reading the csv file
data = pd.read_csv(r"C:\Users\infin\Downloads\train.csv")

#Checking data heads
pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 1500)
data.head()

#Checkin information of data and null entries
data.info()
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)
data.isnull().sum()
data["GarageYrBlt"].value_counts()

#Dropping columns with maximum missing values.Also drop id it being serial number
data=data.drop(["Id","Alley","PoolQC","Fence","MiscFeature"], axis=1)

#Treating null values of categorical columns related to garage
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        data[col]= data[col].fillna('No_Garage') 

#Treating null values of categorical columns related to basement
for col in ('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):
        data[col]= data[col].fillna('No_Basement') 

#Treating other categorical null values
data['FireplaceQu']=data['FireplaceQu'].fillna('No_Fireplace')
data['MasVnrType']=data['MasVnrType'].fillna('None')
data['MasVnrArea']=data['MasVnrArea'].fillna(0)
data['Electrical']=data['Electrical'].fillna('No_ES')
data['GarageYrBlt']=data['GarageYrBlt'].fillna(0)

data['LotFrontage'].describe()

#Filling null in Lot frontage with median 
data['LotFrontage']=data['LotFrontage'].fillna(data['LotFrontage'].median())

# Adding property age colum and dropping built and sold years
data["Property_age"] = data["YrSold"]-data["YearBuilt"]
data = data.drop(["YrSold","YearBuilt","MoSold"],axis=1)
data.info()
data.head()
data.describe()

# Plotting data
plt.figure(figsize=(20,12))
plt.subplot(4,4,1)
sns.regplot(data=data,y="SalePrice",x="LotFrontage")
plt.subplot(4,4,2)
sns.regplot(data=data,y="SalePrice",x="LotArea")
plt.subplot(4,4,3)
sns.regplot(data=data,y="SalePrice",x="MasVnrArea")
plt.subplot(4,4,4)
sns.regplot(data=data,y="SalePrice",x="BsmtFinSF1")
plt.subplot(4,4,5)
sns.regplot(data=data,y="SalePrice",x="BsmtFinSF2")
plt.subplot(4,4,6)
sns.regplot(data=data,y="SalePrice",x="BsmtUnfSF")
plt.subplot(4,4,7)
sns.regplot(data=data,y="SalePrice",x="TotalBsmtSF")
plt.subplot(4,4,8)
sns.regplot(data=data,y="SalePrice",x="TotalBsmtSF")
plt.subplot(4,4,9)
sns.regplot(data=data,y="SalePrice",x="2ndFlrSF")
plt.subplot(4,4,10)
sns.regplot(data=data,y="SalePrice",x="LowQualFinSF")
plt.subplot(4,4,11)
sns.regplot(data=data,y="SalePrice",x="GrLivArea")
plt.subplot(4,4,12)
sns.regplot(data=data,y="SalePrice",x="GarageArea")
plt.subplot(4,4,13)
sns.regplot(data=data,y="SalePrice",x="WoodDeckSF")
plt.subplot(4,4,14)
sns.regplot(data=data,y="SalePrice",x="ScreenPorch")
plt.subplot(4,4,15)
sns.regplot(data=data,y="SalePrice",x="3SsnPorch")
plt.subplot(4,4,16)
sns.regplot(data=data,y="SalePrice",x="PoolArea")
plt.show()

data.columns

plt.figure(figsize=(50,30))
plt.subplot(17,3,1)
sns.boxplot(x='SalePrice', y='MSSubClass', data=data)
plt.subplot(17,3,2)
sns.boxplot(x='SalePrice', y='MSZoning', data=data)
plt.subplot(17,3,3)
sns.boxplot(x='SalePrice', y='Street', data=data)
plt.subplot(17,3,4)
sns.boxplot(x='SalePrice', y='LotConfig', data=data)
plt.subplot(17,3,5)
sns.boxplot(x='SalePrice', y='LotShape', data=data)
plt.subplot(17,3,6)
sns.boxplot(x='SalePrice', y='LandContour', data=data)
plt.subplot(17,3,7)
sns.boxplot(x='SalePrice', y='Utilities', data=data)
plt.subplot(17,3,8)
sns.boxplot(x='SalePrice', y='LandSlope', data=data)
plt.subplot(17,3,9)
sns.boxplot(x='SalePrice', y='Neighborhood', data=data)
plt.subplot(17,3,10)
sns.boxplot(x='SalePrice', y='Condition1', data=data)
plt.subplot(17,3,11)
sns.boxplot(x='SalePrice', y='Condition2', data=data)
plt.subplot(17,3,12)
sns.boxplot(x='SalePrice', y='BldgType', data=data)
plt.subplot(17,3,13)
sns.boxplot(x='SalePrice', y='HouseStyle', data=data)
plt.subplot(17,3,14)
sns.boxplot(x='SalePrice', y='OverallQual', data=data)
plt.subplot(17,3,15)
sns.boxplot(x='SalePrice', y='OverallCond', data=data)
plt.subplot(17,3,16)
sns.boxplot(x='SalePrice', y='RoofStyle', data=data)
plt.subplot(17,3,17)
sns.boxplot(x='SalePrice', y='RoofMatl', data=data)
plt.subplot(17,3,18)
sns.boxplot(x='SalePrice', y='Exterior1st', data=data)
plt.subplot(17,3,19)
sns.boxplot(x='SalePrice', y='Exterior2nd', data=data)
plt.subplot(17,3,20)
sns.boxplot(x='SalePrice', y='MasVnrType', data=data)
plt.subplot(17,3,21)
sns.boxplot(x='SalePrice', y='ExterQual', data=data)
plt.subplot(17,3,22)
sns.boxplot(x='SalePrice', y='ExterCond', data=data)
plt.subplot(17,3,23)
sns.boxplot(x='SalePrice', y='BsmtCond', data=data)
plt.subplot(17,3,24)
sns.boxplot(x='SalePrice', y='BsmtExposure', data=data)
plt.subplot(17,3,25)
sns.boxplot(x='SalePrice', y='BsmtFinType1', data=data)
plt.subplot(17,3,26)
sns.boxplot(x='SalePrice', y='BsmtFinType2', data=data)
plt.subplot(17,3,27)
sns.boxplot(x='SalePrice', y='Heating', data=data)
plt.subplot(17,3,28)
sns.boxplot(x='SalePrice', y='HeatingQC', data=data)
plt.subplot(17,3,29)
sns.boxplot(x='SalePrice', y='CentralAir', data=data)
plt.subplot(17,3,30)
sns.boxplot(x='SalePrice', y='Electrical', data=data)
plt.subplot(17,3,31)
sns.boxplot(x='SalePrice', y='BsmtFullBath', data=data)
plt.subplot(17,3,32)
sns.boxplot(x='SalePrice', y='BsmtHalfBath', data=data)
plt.subplot(17,3,33)
sns.boxplot(x='SalePrice', y='FullBath', data=data)
plt.subplot(17,3,34)
sns.boxplot(x='SalePrice', y='HalfBath', data=data)
plt.subplot(17,3,35)
sns.boxplot(x='SalePrice', y='BedroomAbvGr', data=data)
plt.subplot(17,3,36)
sns.boxplot(x='SalePrice', y='KitchenAbvGr', data=data)
plt.subplot(17,3,37)
sns.boxplot(x='SalePrice', y='KitchenQual', data=data)
plt.subplot(17,3,38)
sns.boxplot(x='SalePrice', y='TotRmsAbvGrd', data=data)
plt.subplot(17,3,39)
sns.boxplot(x='SalePrice', y='Functional', data=data)
plt.subplot(17,3,40)
sns.boxplot(x='SalePrice', y='Fireplaces', data=data)
plt.subplot(17,3,41)
sns.boxplot(x='SalePrice', y='FireplaceQu', data=data)
plt.subplot(17,3,42)
sns.boxplot(x='SalePrice', y='GarageType', data=data)
plt.subplot(17,3,43)
sns.boxplot(x='SalePrice', y='GarageFinish', data=data)
plt.subplot(17,3,44)
sns.boxplot(x='SalePrice', y='GarageQual', data=data)
plt.subplot(17,3,45)
sns.boxplot(x='SalePrice', y='GarageCond', data=data)
plt.subplot(17,3,46)
sns.boxplot(x='SalePrice', y='PavedDrive', data=data)
plt.subplot(17,3,47)
sns.boxplot(x='SalePrice', y='Exterior2nd', data=data)
plt.subplot(17,3,48)
sns.boxplot(x='SalePrice', y='SaleType', data=data)
plt.subplot(17,3,49)
sns.boxplot(x='SalePrice', y='SaleCondition', data=data)
plt.show()

#Plotting heatmap to check the correlation 
plt.figure(figsize=(50,30))
sns.heatmap(data.corr(), annot=True, cmap="YlGnBu")
plt.show()
data.columns

# creating dummy variables
Dcol = ['MSSubClass', 'MSZoning', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl',
       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
       'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
       'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
       'SaleType', 'SaleCondition',]

data_dummy = pd.get_dummies(data[Dcol], drop_first=True)

# Merging the dummy colums with actual data
data = pd.concat([data,data_dummy], axis=1)
data.shape

#Since dummies are added, we will drop original colums
data = data.drop(Dcol, axis=1)
data.head()
data.shape

# Splitting data into train and test
data_train, data_test = train_test_split(data, train_size=0.70, random_state=100)
print(data_train.shape)
print(data_test.shape)

# X_train and y_train
y_train = data_train.pop('SalePrice')
X_train = data_train

# X_test and y_test
y_test = data_test.pop('SalePrice')
X_test = data_test

# Rescaling of numerical columns
scaler = MinMaxScaler()
num_var =['LotFrontage','LotArea','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF',
          '2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
         'TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
         'ScreenPorch','PoolArea','MiscVal','Property_age']

data_train[num_var]= scaler.fit_transform(data_train[num_var])
data_test[num_var]= scaler.transform(data_test[num_var])
data_train.describe()

#Training the model
lr= LinearRegression()
lr.fit(X_train,y_train)

#Extract top 50 features using RFE
rfe = RFE(estimator=lr,n_features_to_select=50)
rfe.fit(X_train,y_train)
X_train.columns[rfe.support_]
X_train_50 = X_train[['LotArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BedroomAbvGr',
       'KitchenAbvGr', 'GarageCars', 'PoolArea', 'Property_age', 'MSZoning_FV',
       'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'LandSlope_Sev',
       'Neighborhood_Crawfor', 'Neighborhood_NoRidge', 'Condition2_PosA',
       'Condition2_PosN', 'RoofMatl_CompShg', 'RoofMatl_Membran',
       'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv',
       'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'Exterior1st_CBlock',
       'Exterior2nd_CBlock', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA',
       'BsmtCond_No_Basement', 'Heating_OthW', 'KitchenQual_Fa',
       'KitchenQual_Gd', 'KitchenQual_TA', 'Functional_Sev', 'Functional_Typ',
       'GarageQual_Gd', 'GarageQual_Po', 'GarageCond_Fa', 'GarageCond_Gd',
       'GarageCond_TA', 'SaleType_Con', 'SaleType_New',
       'SaleCondition_Partial']]
X_test_50 = X_test[['LotArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BedroomAbvGr',
       'KitchenAbvGr', 'GarageCars', 'PoolArea', 'Property_age', 'MSZoning_FV',
       'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'LandSlope_Sev',
       'Neighborhood_Crawfor', 'Neighborhood_NoRidge', 'Condition2_PosA',
       'Condition2_PosN', 'RoofMatl_CompShg', 'RoofMatl_Membran',
       'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv',
       'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'Exterior1st_CBlock',
       'Exterior2nd_CBlock', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA',
       'BsmtCond_No_Basement', 'Heating_OthW', 'KitchenQual_Fa',
       'KitchenQual_Gd', 'KitchenQual_TA', 'Functional_Sev', 'Functional_Typ',
       'GarageQual_Gd', 'GarageQual_Po', 'GarageCond_Fa', 'GarageCond_Gd',
       'GarageCond_TA', 'SaleType_Con', 'SaleType_New',
       'SaleCondition_Partial']]

lr1 = lr.fit(X_train, y_train)
# Print the coefficients and intercept
print(lr1.intercept_)
print(lr1.coef_)

#r2score,RSS and RMSE
y_pred_train = rfe.predict(X_train)
y_pred_test = rfe.predict(X_test)

metric = []
r2_train_lr = r2_score(y_train, y_pred_train)
print(r2_train_lr)
metric.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print(r2_test_lr)
metric.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print(rss1_lr)
metric.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print(rss2_lr)
metric.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print(mse_train_lr)
metric.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print(mse_test_lr)
metric.append(mse_test_lr**0.5)


# r2 for train is 0.90 and for test is 0.78 which indicates overfitting of model
# We will use Ridge and Lasso regression to fit the model

#RIDGE regression

#alphas to tune 

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}

ridge = Ridge()

#cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge,
                       param_grid = params,
                       scoring = 'neg_mean_absolute_error',
                       cv = folds,
                       return_train_score=True,
                       verbose = 1)
model_cv.fit(X_train_50, y_train)

# printing best hyperparameter alpha
print(model_cv.best_params_)

# Fitting ridge model for alpha = 0.05 and printing penalised coefficients
alpha = 0.05
ridge = Ridge(alpha=alpha)
ridge.fit(X_train_50, y_train)
ridge.coef_

#  R2 score, RSS and RMSE after Ridge regression

y_pred_train = ridge.predict(X_train_50)
y_pred_test = ridge.predict(X_test_50)

metric2 = []
r2_train_lr = r2_score(y_train, y_pred_train)
print(r2_train_lr)
metric2.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print(r2_test_lr)
metric2.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print(rss1_lr)
metric2.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print(rss2_lr)
metric2.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print(mse_train_lr)
metric2.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print(mse_test_lr)
metric2.append(mse_test_lr**0.5)


#Lasso Regression

lasso = Lasso()
# cross validation

lasso_model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

lasso_model_cv.fit(X_train_50, y_train)

# printing best hyperparameter alpha
print(lasso_model_cv.best_params_)

# Fitting ridge model for alpha = 20 and printing penalised coefficients
alpha2 =20
lasso = Lasso(alpha=alpha2)    
lasso.fit(X_train_50, y_train) 

lasso.coef_

#  R2 score, RSS and RMSE after Lasso regression

y_pred_train = lasso.predict(X_train_50)
y_pred_test = lasso.predict(X_test_50)

metric3 = []
r2_train_lr = r2_score(y_train, y_pred_train)
print(r2_train_lr)
metric3.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print(r2_test_lr)
metric3.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print(rss1_lr)
metric3.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print(rss2_lr)
metric3.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print(mse_train_lr)
metric3.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print(mse_test_lr)
metric3.append(mse_test_lr**0.5)


# Creating a table which contain all the metrics

lr_table = {'Metric': ['R2 Score (Train)','R2 Score (Test)','RSS (Train)','RSS (Test)',
                       'MSE (Train)','MSE (Test)'], 
        'Linear Regression': metric
        }

lr_metric = pd.DataFrame(lr_table ,columns = ['Metric', 'Linear Regression'] )

rg_metric = pd.Series(metric2, name = 'Ridge Regression')
ls_metric = pd.Series(metric3, name = 'Lasso Regression')

final_metric = pd.concat([lr_metric, rg_metric, ls_metric], axis = 1)

final_metric

# Evaluation of model 

ridge_pred = ridge.predict(X_test_50)

# Plotting y_test and y_pred to understand the spread for ridge regression.
fig = plt.figure(dpi=100)
plt.scatter(y_test,ridge_pred)
fig.suptitle('y_test vs ridge_pred', fontsize=20)           
plt.xlabel('y_test', fontsize=18)                         
plt.ylabel('ridge_pred', fontsize=16)  
plt.show()

lasso_pred = lasso.predict(X_test_50)

# Plotting y_test and y_pred to understand the spread for lasso regression.
fig = plt.figure(dpi=100)
plt.scatter(y_test,lasso_pred)
fig.suptitle('y_test vs lasso_pred', fontsize=20)              
plt.xlabel('y_test', fontsize=18)                         
plt.ylabel('lasso_pred', fontsize=16)  
plt.show()

betas = pd.DataFrame(index=X_train_50.columns)
betas.rows = X_train_50.columns
betas['Ridge'] = ridge.coef_
betas['Lasso'] = lasso.coef_
pd.set_option('display.max_rows', None)
betas.head(300)