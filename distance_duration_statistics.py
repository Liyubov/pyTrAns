# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:02:46 2020

@author: liubov for Move in Saclay
"""
import pandas as pd
import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression



def mask_df_after_date(df, date):
    '''function 
    input: dataframe, date in the format "2019-10-04"# e.g. 1 month before strike

    output: dataframe values after date '''
    
    #format of the column with date should be of type "to_datetime"
    df['datetimestart'] = pd.to_datetime(df['datetimestart'])
    
    print('we cut 1 month before strike and after given date') #to compare it with 1 month after strike

    mask_after = (df['datetimestart'] > date)
    df_new = df.loc[mask_after]


    df_new.head(5)
    return df_new


#function to estimate distance vs. duration statistics
#We plot distance vs. duration statistics estimated from data from dataframe.

def distance_duration(df_strike, df_before_strike):

    duration_strike = df_strike.durationsec
    distance_strike = df_strike.distance

    plt.plot(duration_strike, distance_strike,'bo', alpha = 0.1)
    plt.xlabel('duration')
    plt.ylabel('distance')

    plt.plot(df_before_strike.durationsec, df_before_strike.distance,'ro', alpha = 0.1)
    plt.xlabel('duration')
    plt.ylabel('distance')

    plt.legend(['Values during strike','Values before strike, filtered'])
    plt.show()


def linear_regression(df_strike, df_before_strike):
    
    
    duration_strike = df_strike.durationsec
    distance_strike = df_strike.distance
    
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


########################################
# Main programs body
########################################
    

# load data on total trips, it may be heavy 

filepath_before = ''#'C:/Users/lyubo/Documents/DATA_networks/mobilitydata/cityBrain/my_trips.csv'
filepath_full = ''#'C:/Users/lyubo/Documents/DATA_networks/mobilitydata/cityBrain/trips_updated.csv'

df_full = pd.read_csv(filepath_full)
print(df_full.shape)
print(df_full.columns)
df_full.head(5)


df_before_strike = pd.read_csv(filepath_before)
print(df_before_strike.shape)
print(df_before_strike.columns)
df_before_strike.head(5) 


date = "2019-10-04"# e.g. 1 month before strike, strike date "2019-12-04"
date_strike = "2019-12-04"

df_before = mask_df_after_date(df_before_strike, date)
df_strike = mask_df_after_date(df_full, date_strike)

# plot distance vs. duration distributions before and after strike
distance_duration(df_strike, df_before)

#make fitting linear regression (see notebook)

