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


