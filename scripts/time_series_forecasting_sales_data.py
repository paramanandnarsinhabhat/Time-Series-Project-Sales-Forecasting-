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



