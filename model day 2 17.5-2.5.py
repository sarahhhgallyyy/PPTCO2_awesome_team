#PFR ideal
#ode 
#mass balance, heat balance
#missing moisture sensor, in the future make moisture model
#GC at exit to fnd saturation of CO2

#importing stuff
import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

path = r""
data = pd.read_csv(path, sep=',', decimal=',', parse_dates=[0])
data = data.rename(columns={data.columns[0]: 'DateTime'}).set_index('DateTime')

# ensure all remaining columns are numeric
data = data.apply(pd.to_numeric, errors='coerce')

window_size = 16
T_rolling = data.iloc[:, 7:-1].rolling(window=window_size).mean() #7 to remove all flow and pressure meters

# plot vs time index
for col in T_rolling.columns:
    plt.plot(T_rolling.index, T_rolling[col], label=col)

plt.xlabel("Time")
plt.ylabel("Rolling Mean")
plt.title("test 1")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') #so the legend is not on the figure itself
plt.show()