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

