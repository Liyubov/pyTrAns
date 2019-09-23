#!/usr/bin/env python
# coding: utf-8

# ## Trajectories analysis: distributions of jumps
# 
# Calculate distribution of jumps from a given trajectory by its coordinates X(t), Y(t). 
# We consider cases for 2-dimensions and n-dimensions.

# In[ ]:


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


# In[3]:


def calc_dist_highdim(point1,point2): 
    '''
    input:
    point1, point2 are points in N-dimensional space, array format
    
    output:
    Euclidean distance between two points
    '''

    sum_dist = 0
    dim = len(point1) #dimension of coordinates of an array
    for ind in range(0,dim):
        sum_dist = sum_dist + (point1[ind] - point2[ind])*(point1[ind] - point2[ind])
    distance = np.sqrt(sum_dist)    
    return distance #gives distance in 175 dimensional space


def calc_distrib_highdim(vector_array):
    '''
    input:      array of vectors
    vector1: [0, 1, 0 ....]
    vector2: [0, 0, 0 ....]
    
    output:     array of distances of jumps between consequent points
                # one can also calculate distribution from initial point x0
    '''  
    arr_shape = vector_array.shape()
    dim = arr_shape[1] # dimension of each vector
    n = arr_shape[0] #number of vectors in array
    
    distrib_dist = np.zeros((n,n))    # if n is large we may get #memory error
    #x0 = Xarray[0]
    #y0 = Yarray[0]

    for i in range(0, n ):
        for k in range(i, n):    #since distance is symmetric we need to calculate it only between i and k, where k>i
            
            distrib_dist[i, k] = calc_dist_highdim(vector_array[i,:], vector_array[k,:]) # calculate distances in dim-dimensional space
        
    return distrib_dist


def plot_dist_from_traj(dist):
    #plt.hist(jumps_lengths, bins=20, alpha=0.2) # alpha is transparency parameter    # now we are plotting the histogram 
    plt.xlabel('distance')
    plt.ylabel('frequency')
    #ax.set_xscale('log')
#    plt.show()
    seaborn.distplot(dist, bins=20)
#    plt.title('distribution of jumps ')
    plt.show()    


# In[ ]:




