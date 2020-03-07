#!/usr/bin/env python
# coding: utf-8

# In[32]:


#Assignment_1 for CS 4375.0U2 
#using import from_future To make sure that print function to make sure the sytax of print functions is still functional in future builds.
from __future__ import print_function
#now using %matplotlib inline; this is for line-oriented magic, this will be allowed results be used in the right hand side.
get_ipython().run_line_magic('matplotlib', 'inline')
#import the Scientific Package that proides support for multi-dimensional Array Object
import numpy as np
#import Pandas libary to support pandas for data manipulation and analysis with structures/tables.
import pandas as pd
#import the statsmodel, since linear regression OLS is involved.
import statsmodels
#import the dataset loading
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
#import matplotlib.pyplot as plt is for the ability to save the namespace
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
#now importing the Libraries of sklearn to allow to facilitate Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
fires = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv")
fires.columns = [ 'X',
 'Y',
 'month',
 'day',
 'FFMC',
 'DMC',
 'DC',
 'ISI',
 'temp',
   'RH',
    'wind', 'rain','area']
#delete the day and The Month
#Here is Using the is Null function to see if there are any missing values from out data sets.
fires.isnull().sum()
fires_features = fires.columns.values.tolist()
#listing the attributes
fires_features
#delete the day and The Month
del fires['day']
del fires['month']
#since the Target is the area, the features gets reduced to 10 since we are testing the other possible candidates
X = fires.iloc[:, 0:10].values    
y= fires.iloc[:, 10].values
#Get number of Columns
col_nums=len(fires_features)

s_basics = pd.DataFrame(index=range(0, col_nums - 2), columns=('min', 'max', 'mean', 'median', 'std'))
#here show that the stats are normal before ops
print(stats_basics.T)
#have the minn, max, other stats variables free.
index=0
for attr in [0, 1] + list(range(4, col_nums)):
    stats_basics.loc[index] = {'min':    min(fires[fires_features[attr]]), 
                           'max':    max(fires[fires_features[attr]]),
                           'mean':   fires[fires_features[attr]].mean(),
                           'median': fires[fires_features[attr]].median(),
                           'std':    fires[fires_features[attr]].std()}
print(stats_basics.T)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print((smf.ols(formula = "area ~ temp + wind + rain", data = fires).fit()).summary())


# In[29]:




