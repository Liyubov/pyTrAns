# Calculation of gyration radius of trajectories 

import numpy as np
import seaborn
from matplotlib import pyplot as plt



# Calculate centre of masses for X_tr and Y_tr 
# First we split them into pairs (X_tr1, Y_tr1), ... 
Tr_array = np.zeros((2,np.size(X_tr)))
print(np.size(X_tr))

for i in range(0,np.size(X_tr)):
    Tr_array[0,i] = X_tr[i]
    Tr_array[1,i] = Y_tr[i]


nonZeroMasses = Tr_array[np.nonzero(Tr_array[:,2])] # Not really necessary, can just use masses because 0 mass used as weight will work just fine.
CM = np.average(nonZeroMasses[:,:2], axis=0, weights=nonZeroMasses[:,2])
print('centre of masses', CM)

# Now we make summation through trajectory elements 

rad_gyr = 0
for i in range(0, np.size(X_tr)):
    rad_gyr = rad_gyr + (X_tr[i] - CM[0])*(X_tr[i] - CM[0]) + (Y_tr[i] - CM[1])*(Y_tr[i] - CM[1])
rad_gyr = np.sqrt(rad_gyr*1./np.size(X_tr))
print('radius gyration', rad_gyr)


# Now we make summation through trajectory elements in Time

rad_gyr = np.zeros(np.size(X_tr))
sum_gyr = 0
#rad_gyr[0] = (X_tr[0] - CM[0])*(X_tr[0] - CM[0]) + (Y_tr[0] - CM[1])*(Y_tr[0] - CM[1])
for i in range(0, np.size(X_tr)):
    sum_gyr = sum_gyr + (X_tr[i] - CM[0])*(X_tr[i] - CM[0]) + (Y_tr[i] - CM[1])*(Y_tr[i] - CM[1])
    rad_gyr[i] = np.sqrt(sum_gyr*1./i)
print('radius gyration', rad_gyr)

plt.plot(rad_gyr) # % tuple(popt))
plt.xlabel('time')
plt.ylabel('radius of gyration')
plt.legend()
plt.show()


