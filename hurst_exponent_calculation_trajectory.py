#!/usr/bin/env python
# coding: utf-8

# # Temporal analysis
# ### Memory effects and Hurst exponent
# 
# One can also estimate Hurst exponent, which is used as a measure of long-term memory of time series
# https://en.wikipedia.org/wiki/Hurst_exponent#Estimating_the_exponent.
# Hurst exponent relates to the autocorrelations of the time series, and the rate at which these decrease as the lag between pairs of values increases. 
# For any trajectory we can write $Var(τ) \propto \tau^{(2H)}$, where H is the Hurst exponent.
# Hence $(Var(z(t)-z(t-\tau))) \propto \tau^{(2H)}$. 
# Then 
# [log (Var(z(t)-z(t-τ))) / log τ ] / 2 ∝ H (gives the Hurst exponent) where we know the term in square brackets on far left is the slope of a log-log plot of tau and a corresponding set of variances.
# *Range of Hurst exponent.*
# A value H in the range 0.5–1 indicates a time series with long-term positive autocorrelation, meaning both that a high value in the series will probably be followed by another high value and that the values a long time into the future will also tend to be high. A value in the range 0 – 0.5 indicates a time series with long-term switching between high and low values in adjacent pairs, meaning that a single high value will probably be followed by a low value and that the value after that will tend to be high, with this tendency to switch between high and low values lasting a long time into the future. A value of H=0.5 can indicate a completely uncorrelated series.
# 
# Basically, the idea of Hurst exponent is to characterize trajectory in terms of self-repetition. If Hurst exponent is between [0,0.5], then it indicates that trajectory has switches between different regimes. 
# While if Hurst exponent  [0.5, 1], then it means that there are less switches between long and short jumps and trajectory has long-term positive autocorrelation. If Hurst exponent is 0.5, then it indicates completely uncorrelated series (although it depends on scales). 
# 
# 

# In[6]:


import numpy as np
import seaborn
from matplotlib import pyplot as plt



'''
input: 

data1 is an trajectory, we get array of sequences X_1(t), X_2(t),... X_n(t), 
where n is number of dimensions, t is time. 

'''

size = 100
data1 = np.random.random((size, size)) # for example we take random sample

lags = range(2,100)
def hurst_exponen_data(p):
    '''
    given series p(t), where t is time 
    p(t) is format of zip(list) of arrays from X and Y
    '''    
    variancetau = []; tau = []

    for lag in lags: 
        #  Write the different lags into a vector to compute a set of tau or lags
        tau.append(lag)
        # Compute the log returns on all days, then compute the variance on the difference in log returns
        # call this pp or the price difference
        pp = np.subtract(p[lag:], p[:-lag])
        variancetau.append(np.var(pp))

    # we now have a set of tau or lags and a corresponding set of variances.
    #print tau
    #print variancetau

    # plot the log of those variance against the log of tau and get the slope
    m = np.polyfit(np.log10(tau),np.log10(variancetau),1)

    hurst = m[0] / 2

    return hurst

print('hurst exponent')
print(hurst_exponen_data(data1))




# ### STD of trajectory
# 

# In[7]:


import numpy as np
import seaborn
from matplotlib import pyplot as plt


'''
input:
we get input trajectory from another file 
or 
we get input trajectory from X(t) and Y(t)
'''

print("Calculating STD of trajectory ")  
#data1 = list(zip(X_tr1, Y_tr1))#zip together x and y coordinates
#data1 = np.asarray(latlon1) # if imported from dataframe
print("std of trajectory : ", np.std(data1)) 

