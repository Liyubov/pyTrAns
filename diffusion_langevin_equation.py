#!/usr/bin/env python
# coding: utf-8

# # Analysis of single trajectories 
# 
# Consider a trajectory of free Brownian particle.
# <img src="images/brown.png" alt="Drawing" style="width: 350px;"/>
# 
# ## Langevin equation
# Consider a free particle of mass m with equation of motion described by
# $ m \frac{d p}{dt}$ =$-\frac{p}{\mu }+ a{W}(t),$
# 
# where $p$ is the particle velocity, μ is the particle mobility, $a$ is some parameter of the random force,   $W(t)$ is a rapidly fluctuating force whose time-average vanishes over a characteristic timescale $t_c$ of particle collisions. 
# 
# 
# A solution of a Langevin equation for a particular realization of the fluctuating force is of no interest by itself. 
# 
# The Langevin equation describes the dynamics of a particle that moves according to Newton’s second law and is in contact with a thermal reservoir that is at equilibrium.

# In[12]:


## simulation of Langevin equation
# we modify code from Langevin equation repository and add our part on diffusion simulations
# thanks to Pavliotis for explanations and to https://github.com/DelSquared/Partial-Differential-Equations/blob/master/Addendum%20To%20The%20Tensorflow%20PDE%20Documentation/Tensorflow%20PDE%20(Heat%20Equation).ipynb


import numpy as np 
from scipy.integrate import odeint
from matplotlib import pyplot as plt

k,m=0.000001,1.5 
#the differential equation y''= -my + kR is very sensitive on these parameters k and m.
#The odeint() functon may not work with some values because it thinks that there may not be convergence.

def W(t): #F is the "random force" term
    # if the Force is random we get Brownian motion
    return np.random.normal(0,1) #np.random.pareto(0.4, 2)#
    

def Langevin (xv,t): #Defining the ODE plus stochastic force = SDE
    # xv 
    x,p=xv
    dxv=[p,-m*p+k*W(t)]
    return dxv

t=np.linspace(0,1000,1000000)
XV=np.array([0,1]) #Initial conditions

xv=odeint(Langevin,XV,t)

fig=plt.figure(figsize=((2**13)/100,(2**10)/100))
plt.plot(t,xv[:,0],label="velocity")
plt.plot(t,xv[:,1],label="position")
plt.xlabel('time')
plt.ylabel('')
plt.show()
fig.savefig('soln.png')


# ## Diffusion equation 
# 
# In order to find corresponding diffusion equation we can write
# the Langevin equation as a system of first-order stochastic differential equations in phase space $(q, p)$. 
# For this we introduce the momentum $pt = ˙qt$ (we write $qt = q(t)$, $pt = p(t)$).
# 
# Then we get 
# 
# $dq_t = p_t dt,$
# 
# $dp_t = −∇V(q_t )dt −γp_t dt + \sqrt{2γ^{β−1}}dW_t .$
# 
# From this we can get the generator for Markov process, as shown in [Pavliotis]. 
# Then the evolution of the probability distribution func- tion of the Markov process with coordinate and momentum ${q_t , p_t}$ is governed by the Fokker–Planck equation
# $$ \frac{d \rho}{ dt} = - p \nabla \rho + \nabla V \nabla \rho + \gamma (\nabla_p (p \rho)  + \beta^{-1} \Delta_p \rho) .$$
# 
# We can start with simpler example of diffusion equation and solve fist simpler version 
# $$ \frac{d \rho}{ dt} = \nabla^2 \rho .$$
# 
# 

# In[ ]:


#Import libraries for simulation
import tensorflow as tf
import numpy as np

#Imports for visualization
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display


# In[2]:


# adapted code

def DisplayArray(a, fmt='jpeg', rng=[0,1]):
    """Display an array as a picture."""
    a = (a - rng[0])/float(rng[1] - rng[0])*255
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    clear_output(wait = True)
    display(Image(data=f.getvalue()))

## functions for Laplace operator 
def make_kernel(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1])
    return tf.constant(a, dtype=1)



#sess = tf.InteractiveSession()

def simple_conv(x, k):
    """A simplified 2D convolution operation"""
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
    return y[0, :, :, 0]

def laplace(x):
    """Compute the 2D laplacian of an array"""
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                           [1.0, -6., 1.0],
                           [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)


# In[ ]:



## simulation of corresponding diffusion equation
# tnx to  DelSquared 


''' initial array'''

N=500
# Initial Conditions -- some high temperature spots on material

# Set everything to zero
u_init = np.zeros([N, N], dtype=np.float32)

# Some high temp spots hit at random
for n in range(40):
  a,b = np.random.randint(0, N, 2)
  u_init[a,b] = np.random.uniform()

DisplayArray(u_init, rng=[-0.1, 0.1])

''' array after diffusion'''
# Parameters:
# eps -- time resolution
# D -- diffusion/conductivity coefficient
dt = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())
D = 2

# Create variables for simulation state
Rho  = tf.Variable(u_init)

# Discretized PDE update rules
Rho_ = Rho + dt * (D*laplace(Rho)) 

# Operation to update the state
step = tf.group(
  Rho.assign(Rho_))


# In[14]:


## TODO: 
## implement diffusion with parameters from Langevin equation with random force W and parameter a


# In[ ]:




