# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:03:38 2020

@author: lyubo
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#%matplotlib inline
from sklearn.cluster import KMeans 


from scipy.stats import entropy
from math import log, e
#import timeit


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
    X_labels = np.zeros((size_origin[0],2)) 

    X_labels[:,0] = Kmean.labels_[0:size_origin[0]]
    X_labels[:,1] = Kmean.labels_[size_origin[0]:size[0]]
    
    #print('done with clustering')
    
    return Kmean, X_labels





# function to estimate simple entropy     
def simple_entrop_data(Kmean_res, array_size):
    '''
    input: Kmean_res results of k-means clustering on k clusters,
    output: function returns entropy array as function of time'''
    
    size_origin = np.shape(Kmean_res.labels_)
    #print('size of array', size_origin)
    entrop_array_time = np.zeros(2*int(size_origin[0])) # size of entrop_array is (2*size) since each trip has 2 entrances

    for ind in range(1, 2*int(size_origin[0])): 
        #each entrance of X_new[i, j] corresponds to trip 
        entrop_shape = np.shape(np.unique(Kmean_res.labels_[0:ind]))
        entrop_array_time[ind] = int(entrop_shape[0])
        
    return entrop_array_time

 
# function to estimate shannon entropy    
def entropy_shannon(labels, base=None):
    '''
    input: labels 1D array of set of points
    '''
    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)
    
# different way of calculation of entropy
def entropy2(labels, base=None):
    '''
    input: labels 1D array of set of points
    output:
    computes entropy of label distribution. 
    '''
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

  # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent
    
    


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


#########################################
# run  clustering code functions on data
k = 30
K_mean_df, X_labels = k_mean_from_data(df_full, k) 
print(X_labels)

# run entropy code on data 
  

print('plotting Shannon entropy as a function of time')
k = 50 #number of clusters
N_data_points = df_full.shape[0] #df.shape() # number of datapoints

Kmean_res, X_labels = k_mean_from_data(df_full, k) # gives function which 
shannon_entrop_array = np.zeros(2*N_data_points) # entropy after each move of the traveler (for each time step there are two points)

print('shape of labels ', np.shape(Kmean_res.labels_[:]))


for time in range(10, 2*N_data_points): 
    shannon_entrop_array[time] = entropy_shannon(Kmean_res.labels_[0:time]) #entropy_shannon(labels)
    
#test and plot entropy array for different k-means clusters    
plt.plot(shannon_entrop_array)
#print('calculating entropy for k clusters ', k_ind)
plt.ylabel('number of unique locations')
plt.xlabel('rescaled time - trips ') #trips
plt.title('shannon entropy for one user')
plt.show()




