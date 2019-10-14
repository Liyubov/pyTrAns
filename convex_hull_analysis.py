#!/usr/bin/env python
# coding: utf-8

# # Convex hull analysis of trajectories 
# 
# Given trajectory in 2D (in general in N dimensions) we can characterize some of its properties by the applying convex hull algorithms. 
# Definition of convex hull for a given trajectory in any dimensional space can be explained also by 
# https://en.wikipedia.org/wiki/Convex_hull
# 
# 
# We draw the polygon around trajectory points and estimate its volume, which allows us to characterize the trajectory qualitatively .
# 
# 
# 
# ### Convex hull sliding window analysis of trajectories
# We also analyze trajectory using sliding window analysis, where we measure the volume of convex hull $V(t,t+ \Delta t)$ for trajectory $Tr(t)=(X(t),Y(t))$ during the period of ($\Delta t$). 
# To read more about convex hull algorithm applied to particle trajectories 
# [1] https://arxiv.org/pdf/1708.06517.pdf
# 
# Example for the convex hull sliding window analysis for Levy flight with distributions of steps following various distributions can be found in the article [1].
# 
# Why do we care about convex hull? Convex hull analysis allows to give quantitative characteristics of the trajectory in time. 
# This sliding window analysis however depends on the size of the window. Therefore we need some other complementary characteristics for measures the long-term trends of trajectories.
# 
# 

# In[1]:



import matplotlib.pyplot as plt
import numpy as np
import os
#from RDP import rdp #import rdp
import networkx as nx
from scipy.spatial import ConvexHull

                       
def convex_hull_window(data):
    hull = ConvexHull(data)
    print(hull.volume)
    if hull.volume == 0:
        return 0
    #if 2dimensional then one cannot calculate volume
    else:
        return hull.volume
    

def convex_hull_sliding_window(data, steps, size_window):
    '''
    data - trajectory in format np.ASARRAY(data1) 
    steps - number of time steps in trajectory 
    size_window - time size of the sliding window
    '''
    volume_array = np.zeros(steps)
    
    for itime in range(0, int((steps-size_window)/2)): #steps-1):
        #print(itime)
        time_max = (itime +size_window)%(steps+1) #calculate maximum time for sliding window
        data_i = data[itime: time_max] #cut data_i  from origianal data by cutting trajectory 
        volume_array[itime] = convex_hull_window(data_i)#apply convex_hull function
        #make exception for non-convex areas
    return volume_array

def  convex_hull_window_dataframe(dataframe, size_window):
    '''
    data - trajectory as DATAFRAME
    steps - number of time steps in trajectory 
    size_window - time size of the sliding window
    '''
    #extract array of trajectory from dataframe
    X_tr = dataframe.x.values
    Y_tr = dataframe.y.values
    datanew = list(zip(X_tr, Y_tr))#zip together x and y coordinates
    data = np.asarray(datanew)

    #steps calculation 
    steps = np.size(data) 
    print('steps should be more than time-window ', steps)
    
    volume_array = np.zeros(steps)
    print('applying window analysis to number of steps ', int((steps-size_window)/2))
    for itime in range(0, int((steps-size_window)/2)): #steps-1):
        #print('window itime ',itime)
        time_max = (itime +size_window)%(steps+1) #calculate maximum time for sliding window
        data_i = data[itime: time_max] #cut data_i  from origianal data by cutting trajectory 
        volume_array[itime] = convex_hull_window(data_i)#apply convex_hull function
        #make exception for non-convex areas
    plt.plot(volume_array)
    plt.title("Convex hull volume")
    plt.xlabel("time (number of articles)")
    plt.ylabel("Convex volume")
    
    #return volume_array

# TODO: 
# calculate average convex hull volume for trajectory 
# show that in the middle of career people are more active


# 
# Apply sliding window analysis to example trajectory in 2D. 
# 

# In[ ]:




