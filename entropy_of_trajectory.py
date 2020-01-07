# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:03:38 2020

@author: lyubo
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
%matplotlib inline
from sklearn.cluster import KMeans 



def k_mean_from_data(df, k):
    ''' 
    input: real data of coordinates 
    output: 
    K_mean result of clustering; 
    Clusters - (2 x N) array of number of clusters of type
    
    [cl_1_start, cl_1_stop
     cl_2_start, cl_2_stop
     ...
    ]
    '''
#    df = pd.read_csv('C:/Users/lyubo/Documents/DATA_networks/mobilitydata/cityBrain/my_trips.csv')

    Y = df.latitudestart.values # in the format of np.random.rand(100,2)
    X = df.longitudestart.values
    Y = np.append(Y, df.latitudestop.values)
    X = np.append(X, df.longitudestop.values)
    
    #plt.scatter(X, Y, s = 50, c = 'b')
    #plt.show()

    size_origin = np.shape(df.latitudestop.values)
    #print('size destinations ', size_origin)

    size = X.shape

    X_data = np.zeros(( int(size[0]),2))
    #print(X_data.shape)              
                  
    X_data[:,0] = X 
    X_data[:,1] = Y
          
    #print('plotting data on a map with centres')
    
    Kmean = KMeans(n_clusters=k) #In this case, we arbitrarily gave k (n_clusters) an arbitrary value of two
    Kmean.fit(X_data)

    #print('centres of clustering ', Kmean.cluster_centers_)
    #array_cent = Kmean.cluster_centers_    
    X_new = np.zeros((size_origin[0],2)) 

    X_new[:,0] = Kmean.labels_[0:size_origin[0]]
    X_new[:,1] = Kmean.labels_[size_origin[0]:size[0]]
    
    #print('done with clustering')
    
    return Kmean, X_new


########################################
# Main programs body
########################################
    

# load data on total trips, it may be heavy 

filepath_before = 'C:/Users/lyubo/Documents/DATA_networks/mobilitydata/cityBrain/my_trips.csv'
filepath_full = 'C:/Users/lyubo/Documents/DATA_networks/mobilitydata/cityBrain/trips_updated.csv'

df_full = pd.read_csv(filepath_full)
print(df_full.shape)
print(df_full.columns)
df_full.head(5)


