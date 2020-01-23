
import pandas as pd
import numpy as np

# load the data of trips 

# example data
filepath_full = 'mypath'

df_full = pd.read_csv(filepath_full)
print(df_full.shape)
print(df_full.columns)
df_full.head()



import seaborn as sns, numpy as np
import matplotlib.pyplot as plt

# estimate distance between two sequent trips, plot them in the form of CTRW graph X(t)


f, ax = plt.subplots(figsize=(7, 7))
#ax.set( yscale="log")
plt.plot( cumulative_duration, cumulative_distance)
plt.title('Cumulative plot of distance vs. time')
plt.xlabel('delta time')
plt.ylabel('delta distance')
plt.show()

# Average speed calculation 

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#fitting data before strike 

#duration = df_full.durationsec
#distance = df_full.distance
x_bef = cumulative_duration #df_before_strike.durationsec
y_bef = cumulative_distance #df_before_strike.distance

fit_bef = np.polyfit(x_bef,y_bef,1)



#plotting
plt.plot(x_bef,y_bef,'bo',alpha =0.1)
plt.plot(x_bef, fit_bef[0]*x_bef + fit_bef[1], color='darkblue', linewidth=2)


plt.xlabel('duration')
plt.ylabel('distance')

plt.legend(['Values','Lin.regression fit']) #,'Values during strike','Lin.regression'])
plt.show()





