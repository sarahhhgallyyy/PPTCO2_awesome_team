#importing stuff
#PFR ideal
#ode 
#mass balance, heat balance
#missing moisture sensor, in the future make moisture model
#GC at exit to fnd saturation of CO2

import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

path = r"C:\Users\20210996\OneDrive - TU Eindhoven\Desktop\TUe\Year 4 2025-26\PPT Lab\Day 2 2.5CO2 17.5N2 attempt 2.csv"

# read correctly: EU numbers + parse first col as datetime
data = pd.read_csv(path, sep=',', decimal=',', parse_dates=[0])
data = data.rename(columns={data.columns[1]: 'DateTime'}).set_index('DateTime')

# ensure all remaining columns are numeric
data = data.apply(pd.to_numeric, errors='coerce')

window_size = 16
T_rolling = data.rolling(window=window_size).mean()

# plot vs time index
for col in T_rolling.columns:
    plt.plot(T_rolling.index, T_rolling[col], label=col)

plt.xlabel("Time")
plt.ylabel("Rolling Mean")
plt.title("test 1")
plt.legend()
plt.show()
