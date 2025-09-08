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

#import data
path = r"C:\Users\20210996\OneDrive - TU Eindhoven\Desktop\TUe\Year 4 2025-26\PPT Lab\trial day 2.csv"
data = pd.read_csv(path, sep=';', decimal=',')  #, delimiter="\t")

#select data: time and T data (for now)
header_names = [f"TT{i}" for i in range(16)]
t = data.iloc[0, :]#.values
T = data.iloc[:, 1:17]#.values

plt.plot(t, T)
plt.show()
exit()

#apply rolling average
window_size = 16
print(type(T))
T_rolling = T.rolling(window_size).mean()
T_rolling = T_rolling.values[window_size:, :]
t= t[window_size:].values

print(data.columns)
plt.plot(t, T_rolling)
plt.show()


