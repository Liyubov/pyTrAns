#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# ### Load trajectories
# 
# 
# @Liubov Tupikina 
# 
# We first load real trajectories from real data.
# We start with sample data from Brownian motion. 
# Loading big csv files may take time.
# 
# 1. Mobility trajectory (open data) from openhumans as we analyzed them here https://github.com/Liyubov/mobility_analysis
# 2. D. S. Grebenkov, Dataset created for the project INADILIC. A Brownian motion trajectory is generated as a cumulative sum of independent Gaussian variables with mean 0 and  variance 2D, with the diffusion coefficient D set to 1. The full set contains M = 1000 one-dimensional trajectories of N = 10000 points (steps). 
# 3. Experimental Data from (Golding & Cox) mRNA in E. coli cell with N=140-1600, M=27, d=2
# 	
# 
# Data on real trajectories is take from the research project, which I was part of http://inadilic.fr/data/ (credits to D.Grebenkov, the leader of the project).
#     
# 

# In[23]:


import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt



# load data, it is very heavy 
print('loading the data')
file_name = "C:/Users/lyubo/Documents/DATA_networks/trajectories_data/BM_sample.txt"
file_name_Nd = "C:/Users/lyubo/Documents/DATA_networks/trajectories_data/GC_sample_cell_data.txt"


traj = np.loadtxt(file_name) #, delimiter = ',')
print('data loaded in format ', type(traj),np.shape(traj))

traj_Nd =  np.loadtxt(file_name_Nd) #, delimiter = ',')
print('data loaded in format ', type(traj_Nd), np.shape(traj_Nd))

shape = np.shape(traj)
size = shape[0]

shape = np.shape(traj_Nd)
size_Nd =  shape[0]

# we can now save trajectory as dataframe for convenience 
# for 1D trajectory
df = pd.DataFrame({'Column1': traj[:]})

# for ND trajectory
df_Nd =  pd.DataFrame({'Column1': traj_Nd[:,0], 'Column2': traj_Nd[:, 1], 'Column3': traj_Nd[:, 2]})


# we can also load trajectories from csv file
#traj = pd.read_csv('C:/Users/lyubo/Documents/DATA_networks/mobilitydata/bikes_sharing_data_technologiestiftung_berlin/pseudonomysed_raw.csv')
#traj.head()


# ## Displaying trajectories

# In[8]:


import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import csv
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
timeaxis =  np.linspace(start = 0, stop = size, num = size)
shape = np.shape(traj)
size = shape[0]
ax.scatter(timeaxis, traj, c=range(size), linewidths=0,marker='o', s=3, cmap=plt.cm.jet,) # We draw our points with a gradient of colors.
ax.axis('equal')
ax.set_axis_off()
#fig.suptitle('Distribution of steps for RW a='+str(a), fontsize=16)
fig.suptitle('Trajectory from real data', fontsize=16)


#  Displaying trajectories in N-dimensions.

# In[24]:


import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import csv
import numpy as np



from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for ind in range(0, size_Nd):
    xs = traj_Nd[ind, 0]#randrange(n, 23, 32)
    ys = traj_Nd[ind, 1]#randrange(n, 0, 100)
    zs = traj_Nd[ind, 2]#randrange(n, zlow, zhigh)
    
    ax.scatter(xs, ys, zs, marker=m)    
plt.show()


# In[ ]:




