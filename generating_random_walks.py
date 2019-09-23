#!/usr/bin/env python
# coding: utf-8

# ## Generating random walks
# 
# Here we generate random walks in N-dimensional space. We take N=2, easier to visualize.
# Alternatively we can also create or load dataframe with trajectories.
# We consider several distributions: 
# 1. Weibul distribution 
# 2. Pareto distribution 
# 3. Normal distribution
# 

# In[3]:




import matplotlib.image as mpimg
import numpy as np
from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


'''
Parameters of RW setting

'''
n= 500 #length of random walk
mu = 0.5 #normal distribution
sigma =20
beta = 5 #exponential parameters
a = 1 # pareto distribution
weib = 1 #weibul parameter
k = 10


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
X_tr = np.interp(np.arange(n * k), np.arange(n) * k, x)
Y_tr = np.interp(np.arange(n * k), np.arange(n) * k, y)
#print('x2 rw', x2)


X_tr2 = np.interp(np.arange(n * k), np.arange(n) * k, x_n)
Y_tr2 = np.interp(np.arange(n * k), np.arange(n) * k, y_n)


def generate_rand_walk(n, k, mu, sigma):
    '''
    Parameters of RW setting (we should be able to set any distribution)
    
    n= 500 #length of random walk
    mu = 0.5 #normal distribution
    sigma =20
    
    
    beta = 5 #exponential parameters
    a = 1 # pareto distribution
    weib = 1 #weibul parameter
    k - scaling parameter for trajectory coloring
    '''
    #distribution sets the distribution for steps
    
    x =  np.cumsum(np.random.normal(mu, sigma, n))
    y =  np.cumsum(np.random.normal(mu, sigma, n))

    X_tr = np.interp(np.arange(n * k), np.arange(n) * k, x)
    Y_tr = np.interp(np.arange(n * k), np.arange(n) * k, y)

    
    return X_tr, Y_tr


# ## Plotting random walks

# In[4]:



'''
plotting first RW
'''

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(X_tr, Y_tr, c=range(n * k), linewidths=0,
           marker='o', s=3, cmap=plt.cm.jet,) # We draw our points with a gradient of colors.
ax.axis('equal')
ax.set_axis_off()
fig.suptitle('Distribution of steps for RW mu='+str(mu)+' sigma= '+str(sigma), fontsize=16)
#plt.savefig('RW_motion_steps_normal_mu'+str(mu)+'sigma'+str(mu)+'.png')

'''
plotting second RW
'''


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(X_tr2, Y_tr2, c=range(n * k), linewidths=0,
           marker='o', s=3, cmap=plt.cm.jet,) # We draw our points with a gradient of colors.
ax.axis('equal')
ax.set_axis_off()
fig.suptitle('Distribution of steps for RW mu='+str(mu)+' sigma= '+str(sigma), fontsize=16)
#plt.savefig('RW_motion_steps_normal_mu'+str(mu)+'sigma'+str(mu)+'.png')


# In[ ]:




