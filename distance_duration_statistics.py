#!/usr/bin/env python
# coding: utf-8

# # General statistics 
# @Liubov
# 
# Function to calculate distance vs.duration statistics. 
# We estimate the distribution of distance vs. duration for dataframes from Move in Saclay module before and after strike of 2019 December.

# In[3]:


#import plotly.plotly as py
#import plotly.graph_objs as go
import pandas as pd
import csv


# load data on trips, it may be heavy 

df_full = pd.read_csv('C:/Users/lyubo/Documents/DATA_networks/mobilitydata/cityBrain/trips_updated.csv')

print(df_full.shape)
print(df_full.columns)
df_full.head(10)


# # Data file before strike
# Open data before the strike. We store it in data file "my_trips.csv"

# In[15]:


df_before_strike = pd.read_csv('C:/Users/lyubo/Documents/DATA_networks/mobilitydata/cityBrain/my_trips.csv')

#print(df_before_strike.shape)
#print(df_before_strike.columns)
df_before_strike.head(5) 


# In[20]:


# data before and after strike should be comparable, e.g. the same time-length of the sample
before_strike_date = "2019-11-04"# 1 month before strike

#strike_date = "2019-12-04" #strike date
#greater than the start date and smaller than the end date
#mask_before =  (df_full['datetimestart'] < strike_date)
#Select the sub-DataFrame:
#df_before_strike = df_full.loc[mask_before]


mask_after = (df_before_strike['datetimestart'] > before_strike_date)
df_before_strike_new = df_before_strike.loc[mask_after]


df_before_strike_new.head(20)


print(df_before_strike.shape)
print(df_before_strike_new.shape)


# # Filtering data
# We remove small data points and remove the short trips, which are not counted.

# In[7]:


import numpy as np
import pandas as pd 

df_new = df_before_strike # copy dataframe 

#filter rows in this dataframe 
shape = df.shape
size = shape[0]

for ind in range(0,2580):#range(size): 
    if (df.distance.iloc[ind]<2) or (df.durationsec.iloc[ind]<600): # remove outliers
        #print('dropping', ind)
        df_new = df_new.drop(df.index[ind]) # drop the raw
        
print('size of new cleaned dataframe ', df_new.shape ) 


# # Distance vs. duration statistics
# We plot distance vs. duration statistics estimated from data from dataframe.

# In[9]:


from matplotlib import pyplot as plt

duration = df_before_strike.durationsec
distance = df_before_strike.distance

plt.plot(duration, distance,'bo', alpha = 0.5)
plt.xlabel('duration')
plt.ylabel('distance')

plt.plot(df_new.durationsec, df_new.distance,'ro', alpha = 0.5)
plt.xlabel('duration')
plt.ylabel('distance')

plt.legend(['Values before strike','Values before strike, filtered'])
plt.show()


# In[32]:


df['datetimestart'] = pd.to_datetime(df['datetimestart'])
df.head()


# # Data after strike
# Here we first get the data after strike. In total the sample of data before and after strike should be comparable, e.g. the same time-length of the sample.

# In[21]:


# Make a boolean mask. start_date and end_date can be datetime.datetimes, np.datetime64s, pd.Timestamps, or even datetime strings:
start_date = "2019-12-04"# start date strike

#greater than the start date and smaller than the end date
mask = (df_full['datetimestart'] > start_date) #& (df['date'] <= end_date)

#Select the sub-DataFrame:
df_strike = df.loc[mask]

print(df_strike.shape)


# In[22]:


df_strike.head()


# In[16]:


from matplotlib import pyplot as plt

duration_strike = df_strike.durationsec
distance_strike = df_strike.distance

plt.plot(duration_strike, distance_strike,'bo', alpha = 0.1)
plt.xlabel('duration')
plt.ylabel('distance')

plt.plot(df_new.durationsec, df_new.distance,'ro', alpha = 0.1)
plt.xlabel('duration')
plt.ylabel('distance')

plt.legend(['Values during strike','Values before strike, filtered'])
plt.show()


# # Linear regression fitting
# We first apply simple linear regression algorithm.

# In[10]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[27]:


#fitting data before strike 

x_bef = df_before_strike.durationsec
y_bef = df_before_strike.distance

fit_bef = np.polyfit(x_bef,y_bef,1)


#fitting data during strike
x_str = np.array(duration_strike)#.reshape((-1, 1))
y_str = np.array(distance_strike)

fit = np.polyfit(x_str,y_str,1)

#plotting
plt.plot(x_bef,y_bef,'bo',alpha =0.1)
plt.plot(x_bef, fit_bef[0]*x_bef + fit_bef[1], color='darkblue', linewidth=2)

plt.plot(x_str,y_str,'ro',alpha =0.1)
plt.plot(x_str, fit[0]*x_str + fit[1], color='darkred', linewidth=2)


plt.xlabel('duration')
plt.ylabel('distance')

plt.legend(['Values before strike','Lin.regression','Values during strike','Lin.regression'])
plt.show()


# In[17]:



x = np.array(duration_strike).reshape((-1, 1))
y = np.array(distance_strike)
print(np.shape(x))
print(type(x))

model = LinearRegression()

model = LinearRegression().fit(x, y)


# In[ ]:





# In[54]:


print('intercept:', model.intercept_)
#intercept: 5.633333333333329
print('slope:', model.coef_)


# In[ ]:


#plotting 

plt.plot(X.T[1], y, '.', markersize=30, label='points')
plt.plot(X.T[1], np.dot(X, weights.flatten()), '-',         label='y = %2.2f + %2.2f x' % tuple(weights.flatten()))
plt.xlabel('X')
plt.ylabel('y')

