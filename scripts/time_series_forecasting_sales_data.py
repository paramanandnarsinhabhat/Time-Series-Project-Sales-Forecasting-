import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
# If you are in a Jupyter notebook, use the following line to display plots inline:
from sklearn.metrics import mean_squared_log_error
from math import sqrt
from statistics import mean, stdev
import warnings
# To ignore warnings in your script or notebook, you can use:
warnings.filterwarnings('ignore')



# loading the data
data = pd.read_csv("/Users/paramanandbhat/Downloads/Final_Project/data/Train_KQyJ5eh.csv")

print(data.shape)

#Check the first few number of rows 
print(data.head())

#Check the last few no of rows
print(data.tail())

'''
- We have daily sales data for two years, starting from 1-jan-07 to 24-dec-08.
- Using this historical data, we need to forecast the demand expected in the next 6 months. 

Let us do some basic exploration and find out if the given data has any trend or seasonal patterns.
'''

# 3. Preprocessing the Data
### Plotting Time Series

data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
data.index = data['Date']

plt.figure(figsize=(12,8))

plt.plot(data.index, data['Number_SKU_Sold'], label='Train')
plt.legend(loc='best')
plt.show()

'''
- Clearly there are some very high values in the data. 
- Could this be around holiday season like new year's? Let us find out
'''

### Outliers in Data
print(data['Number_SKU_Sold'].describe())

#Print
print('Value at 95th percentile:', (np.percentile(data['Number_SKU_Sold'], 95)))
print('Value at 97th percentile:', (np.percentile(data['Number_SKU_Sold'], 97)))
print('Value at 99th percentile:', (np.percentile(data['Number_SKU_Sold'], 99)))

sns.boxplot(data['Number_SKU_Sold'])
plt.show()  # This line will display the plot

#IQR for outliers
IQR = (np.percentile(data['Number_SKU_Sold'], 75)) - (np.percentile(data['Number_SKU_Sold'], 25))
whisker_val = (np.percentile(data['Number_SKU_Sold'], 75)) + (1.5*(IQR))

print(whisker_val)

# number of values greater than whisker value
print(data.loc[data['Number_SKU_Sold']>whisker_val].shape)

#get values
print(data.loc[data['Number_SKU_Sold']>whisker_val])

data_original = data['Number_SKU_Sold'] 

data['Number_SKU_Sold'] = data['Number_SKU_Sold'].apply(lambda x: np.nan if x > whisker_val else x)

data['Number_SKU_Sold'].isnull().sum()

# removing outliers using ffill
data['Number_SKU_Sold'] = data['Number_SKU_Sold'].fillna(method ='ffill')

data['Number_SKU_Sold'].isnull().sum()

fig, axs = plt.subplots(2, 1,  sharex=True)

axs[0].plot(data_original,) 
axs[1].plot(data['Number_SKU_Sold'])
  
plt.show() 


### Missing Date Values in Data
print(data['Date'].min(), data['Date'].max())

print('Total days between 01-jan-07 to 24-Dec-08:', (data['Date'].max() - data['Date'].min()).days)
print('Number of rows present in the data are:', data.shape[0])

print(pd.date_range(start = '2007-01-01', end = '2008-12-24' ).difference(data.index))

'''
- There are 137 days missing over the span of two years. 
- Are these weekends? Or Holidays? 
- Are these days missing at random?
'''

start_date = '2007-01-01'
end_date = '2008-12-24'

missing_dates = pd.DataFrame(data = pd.date_range(start = start_date, end = end_date).difference(data.index), 
                             columns= ['Date'])

print(missing_dates)

#Extract day, month, and year from misssing_dates
missing_dates['Day'] = missing_dates['Date'].dt.strftime("%A")
missing_dates['Month'] = missing_dates['Date'].dt.month
missing_dates['year'] = missing_dates['Date'].dt.year

missing_dates.head(5)

print(missing_dates.head(5))

#Let us check which days and which months have 
# maximum number of missing days
print(missing_dates['Day'].value_counts())
#Sunday has highest number of missed days
print(missing_dates['Month'].value_counts())

print(pd.crosstab(missing_dates['year'], missing_dates['Day']))

'''
- Most of the missing days are Sundays
- Remaining 51 missing days are from Mon-Fri
- January has seen most of the missing days
'''

'''
Conclusions:
1. We will not make predictions for Sunday
2. We will make predictions for all weekdays
3. Discuss with stakeholders for manual adjustments on Sunday and holiday
'''

### Dealing with Missing Values
# add rows for missing days
data_ = pd.DataFrame(data['Number_SKU_Sold'])
data_.index = pd.DatetimeIndex(data.Date)

print(data_.head(10))

# add missing dates to the data
idx = pd.date_range('2007-01-01', '2008-12-24')
data_ = data_.reindex(idx, fill_value=0)

print(data_.head(9))

# extract weekday from the dates
data_['Date'] = data_.index
data_['weekday_name'] = data_['Date'].dt.strftime("%A")
data_.shape

print(data_.shape)
print(data_.head())

# remove sundays from data
data_ = data_.loc[data_['weekday_name']!= 'Sunday']
data_.shape

print(data_.shape)

data_original = data['Number_SKU_Sold']
data_['Number_SKU_Sold'] = data_['Number_SKU_Sold'].apply(lambda x: np.nan if x == 0.0 else x)
data_.isnull().sum()

# impute missing values
data_['Number_SKU_Sold'] = data_['Number_SKU_Sold'].fillna(method ='ffill')

data_.isnull().sum()

# 4. Feature Extraction and Exploration
### Decompose Series
from statsmodels.tsa.seasonal import seasonal_decompose

decomposed_series = seasonal_decompose(data_['Number_SKU_Sold'], period=6)

decomposed_series.plot()
plt.show()

# considering 26 days a month 
decomposed_series.seasonal[0:26].plot()

'''
- Pattern repeats 4 times a month
- This suggests weekly seasonality in the data
'''

data_feat = pd.DataFrame({
    "year": data_['Date'].dt.year,
    "month": data_['Date'].dt.month,
    "day": data_['Date'].dt.day,
    "weekday": data_['Date'].dt.dayofweek,
    "weekday_name": data_['Date'].dt.strftime("%A"),
    "dayofyear": data_['Date'].dt.dayofyear,
    "week": data_['Date'].dt.isocalendar().week,
    "quarter": data_['Date'].dt.quarter,
})


complete_data = pd.concat([data_feat, data_['Number_SKU_Sold']], axis=1)
complete_data.head()

print(complete_data.head())

# boxplot for yearly sale
plt.figure(figsize=(10,6))

sns.boxplot(x=complete_data['year'], y=complete_data['Number_SKU_Sold'], )
plt.title('Yearly Sales')
plt.show()

# boxplot for week's sales
plt.figure(figsize=(10,6))

sns.boxplot(x=complete_data['weekday_name'], y=complete_data['Number_SKU_Sold'], )
plt.title('Weekly Sales Trend')
plt.show()

# boxplot for monthly sales
plt.figure(figsize=(10,6))

sns.boxplot(x=complete_data['month'], y=complete_data['Number_SKU_Sold'], )
plt.title('Montly Sales Trend')
plt.show()

'''
- Except for a few high sales in 2007, sales were comparatively higher in 2008
- All weekdays have similar trend on sales
- Average sales are higher towards the end of the year
'''

# 5. Holdout Validation
print(data_.head())

plt.figure(figsize=(12,8))

plt.plot(data_.index, data_['Number_SKU_Sold'], label = 'Data')
plt.legend(loc='best')
plt.show()

#Divide into train and validation sets

train_data = data_[:469]
valid_data = data_[469:]

valid_data.tail()

plt.figure(figsize=(12,8))

plt.plot(train_data.index, train_data['Number_SKU_Sold'], label='Train')
plt.plot(valid_data.index, valid_data['Number_SKU_Sold'], label='Validation')
plt.legend(loc='best')
plt.show()

def rmsle(actual, preds):
    for i in range(0,len(preds)):
        if preds[i]<0:
            preds[i] = 0
        else:
            pass
    
    error = (sqrt(mean_squared_log_error(actual, preds)))*100
    return error

## 6. Time Series Forecasting Models
### Holt's Winters (aka triple enponential smoothing)
#importing module
from statsmodels.tsa.api import ExponentialSmoothing
#training the model
model = ExponentialSmoothing(np.asarray(train_data['Number_SKU_Sold']), seasonal_periods=6, trend='add', seasonal='add')
model = model.fit(smoothing_level=0.2, smoothing_slope=0.001, smoothing_seasonal=0.2)
    
# predictions and evaluation
preds = model.forecast(len(valid_data)) 
score = rmsle(valid_data['Number_SKU_Sold'], preds)

# results
print('RMSLE for Holt Winter is:', score)

## Grid search
from itertools import product
from tqdm import tqdm_notebook

# setting initial values and some bounds for them
level = [0.1, 0.3, 0.5, 0.8]
smoothing_slope = [0.0001, 0.001, 0.05] 
smoothing_seasonal = [0.2, 0.4, 0.6]


# creating list with all the possible combinations of parameters
parameters = product(level, smoothing_slope, smoothing_seasonal)
parameters_list = list(parameters)
len(parameters_list)

print(parameters_list)

def grid_search(parameters_list):
    
    results = []
    best_error_ = float("inf")

    for param in tqdm_notebook(parameters_list):
        #training the model
        model = ExponentialSmoothing(np.asarray(train_data['Number_SKU_Sold']), seasonal_periods=6, trend='add', seasonal='add')
        model = model.fit(smoothing_level=param[0], smoothing_slope=param[1], smoothing_seasonal=param[2])

        # predictions and evaluation
        preds = model.forecast(len(valid_data)) 
        score = rmsle(valid_data['Number_SKU_Sold'], preds)
        
        # saving best model, rmse and parameters
        if score < best_error_:
            best_model = model
            best_error_ = score
            best_param = param
        results.append([param, score])
        
    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'RMSLE']
    
    
    # sorting in ascending order, the lower rmse is - the better
    result_table = result_table.sort_values(by='RMSLE', ascending=True).reset_index(drop=True)
    
    return result_table

result_table = grid_search(parameters_list)

print(result_table.parameters[0])

#training the model
model = ExponentialSmoothing(np.asarray(train_data['Number_SKU_Sold']), seasonal_periods=6, trend='add', seasonal='add')
model = model.fit(smoothing_level=0.1, smoothing_slope=0.0001, smoothing_seasonal=0.2)
    
# predictions and evaluation
preds = model.forecast(len(valid_data)) 
score = rmsle(valid_data['Number_SKU_Sold'], preds)

# results
print('RMSLE for Holt Winter is:', score)

plt.figure(figsize = (12,8))

plt.plot(train_data.index , train_data['Number_SKU_Sold'], label = 'train')
plt.plot(valid_data.index , valid_data['Number_SKU_Sold'], label = 'valid')
plt.plot(valid_data.index , preds, label = 'forecast')
plt.legend(loc='best')

plt.show()


# SARIMA Model
### Stationarity Test
# dickey fuller
from statsmodels.tsa.stattools import adfuller

def adf_test(timeseries):
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

print(adf_test(data_['Number_SKU_Sold']))

'''
If the test statistic is less than the
critical value, we can reject the null 
hypothesis (aka the series is stationary)
. When the test statistic is greater than
the critical value, we fail to reject
the null hypothesis (which means the series is not stationary). 
**Here test statistic is > than critical. Hence series is not stationary**
'''
### Making Series Stationary

def inverse_boxcox(y, lambda_):
    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)

from scipy import stats
train_data['Number_SKU_Sold_log'], lambda_ar = stats.boxcox(train_data['Number_SKU_Sold'])
lambda_ar

train_data['Number_SKU_Sold_log_diff']=train_data['Number_SKU_Sold_log']-train_data['Number_SKU_Sold_log'].shift(6)


plt.figure(figsize=(12, 8))
plt.plot(train_data.index, train_data['Number_SKU_Sold_log_diff'], label='Number SKU Sold Log Diff')
plt.legend(loc='best')
plt.title("Stationary Series")
plt.show()

train_data['Number_SKU_Sold_log_diff_diff'] = train_data['Number_SKU_Sold_log_diff'] - train_data['Number_SKU_Sold_log_diff'].shift(1)
plt.figure(figsize=(12, 8))
plt.plot(train_data.index, train_data['Number_SKU_Sold_log_diff_diff'], label='Number SKU Sold Log Diff Diff')
plt.legend(loc='best')
plt.title("Stationary Series")
plt.show()

### Building Sarima Model

from statsmodels.tsa.statespace import sarimax
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train_data['Number_SKU_Sold_log_diff_diff'].dropna(), lags=25)
plot_pacf(train_data['Number_SKU_Sold_log_diff_diff'].dropna(), lags=25)
plt.show()

#training the model
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train_data['Number_SKU_Sold_log'], seasonal_order=(1, 0, 1, 6), order=(1, 0, 1))
results = model.fit(maxiter=500)

# predictions and evaluation
def rmsle(actual, preds):
    # Assuming actual and preds are Pandas Series of the same length
    # Ensure no negative values
    preds = preds.clip(lower=0)
    return sqrt(mean_squared_log_error(actual, preds)) * 100





# Predictions and evaluation
end = len(train_data) + len(valid_data) - 1  # Adjusted to correct the range
preds = results.predict(start=0, end=end)  # Use the results object for prediction
preds_transformed = inverse_boxcox(preds[len(train_data):], lambda_ar)
print(len(valid_data['Number_SKU_Sold']), len(preds_transformed))
#preds_transformed = inverse_boxcox(preds[len(train_data):len(train_data) + len(valid_data)], lambda_ar)
score = rmsle(valid_data['Number_SKU_Sold'], preds_transformed)
print('RMSLE for SARIMA model Forecasts is', score)
# Ensure that preds is sliced correctly to match the length of valid_data
# Assuming preds was obtained from a model prediction and transformed if necessary
preds_for_valid = preds[-len(valid_data):]

# Now, plot the data
plt.figure(figsize=(12, 8))
plt.plot(train_data.index, train_data['Number_SKU_Sold'], label='train')
plt.plot(valid_data.index, valid_data['Number_SKU_Sold'], label='valid')
plt.plot(valid_data.index, preds_for_valid, label='preds')
plt.legend()
plt.show()

# 6. Building ML Models

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# separating features and target variable
x_train = train_data.drop(['Number_SKU_Sold','weekday_name'], axis=1)
y_train = train_data['Number_SKU_Sold']

x_valid = valid_data.drop(['Number_SKU_Sold', 'weekday_name'], axis=1)
y_valid = valid_data['Number_SKU_Sold']

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Normalize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)

# Training the model
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# Making predictions
preds = model.predict(x_valid_scaled)

# Calculating RMSLE and other results
score = rmsle(y_valid, preds)
print('RMSLE for Linear Regression is', score)

feature_coeff = pd.DataFrame(zip(x_train.columns, model.coef_), columns=['Feature', 'coeff'])
feature_coeff

plt.bar(feature_coeff['Feature'], feature_coeff['coeff'])


