#!/usr/bin/env python
# coding: utf-8
'''
# ## Generating random walks
# 
# Here we generate random walks in N-dimensional space. We take N=2, easier to visualize.
# Alternatively we can also create or load dataframe with trajectories.
# We consider several distributions: 
# 1. Weibul distribution 
# 2. Pareto distribution 
# 3. Normal distribution
'''


import matplotlib.image as mpimg
import numpy as np
from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
#import pandas
import math


def CTRW(alpha, time, scale):
    x= np.zeros((time))
    y= np.zeros((time))
    xs = 0
    ys = 0
    ts = 0
    alpha=0.5
    
    for t in range(time):
    
        while ts < t:
            alpha = 0.5
            #Here I simulate the waiting time according to 1/t**(1+alpha)
            dt=0.01*np.power(1-np.random.rand(),-1/alpha)  
    
            theta=np.random.rand()*2*np.pi
    
            ts += dt         
            xs += np.cos(theta)
            ys += np.sin(theta)
        x[t] = xs
        y[t] = ys
    coord = np.array([x,y]).T
    return scale*coord, scale*x, scale*y #coord


'''
Parameters of RW setting

'''
n= 500 #length of random walk
mu = 0.5 #normal distribution
sigma =20
beta = 5 #exponential parameters
a = 1 # pareto distribution
weib = 1 #weibul parameter

'''
Simple RW motion with random steps
'''

x = np.cumsum(np.random.randn(n))
y = np.cumsum(np.random.randn(n)) 

'''
Now we introduce some CTRW motion in between the steps driven from 
 Weibul distribution 
 Pareto distribution 
 Random normal distribution
'''

x =  np.cumsum(np.random.exponential(1./beta, n))
y =  np.cumsum(np.random.exponential(1./beta, n))

x_w = np.cumsum(np.random.weibull(weib, n))
y_w = np.cumsum(np.random.weibull(weib, n))

x_p =  np.cumsum(np.random.pareto(a, n))
y_p =  np.cumsum(np.random.pareto(a, n))

x_n =  np.cumsum(np.random.normal(mu, sigma, n))
y_n =  np.cumsum(np.random.normal(mu, sigma, n))


# We add 10 intermediary points between two
# successive points. We interpolate x and y.


'''
Now the trajectory is recorded in two arrays x2, y2
'''
k = 10
X_tr = np.interp(np.arange(n * k), np.arange(n) * k, x)
Y_tr = np.interp(np.arange(n * k), np.arange(n) * k, y)
#print('x2 rw', x2)


X_tr2 = np.interp(np.arange(n * k), np.arange(n) * k, x_n)
Y_tr2 = np.interp(np.arange(n * k), np.arange(n) * k, y_n)


# ## Plotting random walks

# In[4]:



'''
plotting one RW
'''

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(X_tr, Y_tr, c=range(n * k), linewidths=0,
           marker='o', s=3, cmap=plt.cm.jet,) # We draw our points with a gradient of colors.
ax.axis('equal')
ax.set_axis_off()
#fig.suptitle('Distribution of steps for RW a='+str(a), fontsize=16)
fig.suptitle('Distribution of steps for RW mu='+str(mu)+' sigma= '+str(sigma), fontsize=16)
#plt.savefig('RW_motion_steps_normal_mu'+str(mu)+'sigma'+str(mu)+'.png')


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(X_tr2, Y_tr2, c=range(n * k), linewidths=0,
           marker='o', s=3, cmap=plt.cm.jet,) # We draw our points with a gradient of colors.
ax.axis('equal')
ax.set_axis_off()
#fig.suptitle('Distribution of steps for RW a='+str(a), fontsize=16)
fig.suptitle('Distribution of steps for RW mu='+str(mu)+' sigma= '+str(sigma), fontsize=16)
#plt.savefig('RW_motion_steps_normal_mu'+str(mu)+'sigma'+str(mu)+'.png')


# -*- coding: utf-8 -*-
"""
CTRW generation
"""

alpha, time, scale = 0.5, 100, 1
scalecoord, X_tr, Y_tr = CTRW(alpha, time, scale)
print(X_tr, Y_tr)

np.savetxt('CTRW'+ str(alpha)+'_time_'+str(time)+'.txt',scalecoord)

'''
plotting CTRW
'''

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(X_tr, Y_tr, #c=range(n * k), linewidths=0,
           marker='o', s=30, cmap=plt.cm.jet,) # We draw our points with a gradient of colors.
ax.axis('equal')
ax.set_axis_off()
plt.savefig('CTRW_alpha_'+ str(alpha)+'_time_'+str(time)+'.png')
plt.show()




