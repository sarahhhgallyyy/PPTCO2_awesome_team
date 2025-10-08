# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 11:23:54 2025

@author: 20223544
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import least_squares
#from Model1 import temperature_steady_state
path = r"Data\free cooling (5 hrs).csv"

# read correctly: EU numbers + parse first col as datetime
data = pd.read_csv(path, sep=',', decimal=',', parse_dates=[0])
data = data.rename(columns={data.columns[0]: 'DateTime'}).set_index('DateTime')

# ensure all remaining columns are numeric
data = data.apply(pd.to_numeric, errors='coerce')
window_size = 32
T_rolling = data.rolling(window=window_size).mean()
start = pd.Timestamp("2025-09-22 15:57:33")
end = pd.Timestamp("2025-09-22 20:30:00")
T_window = T_rolling.loc[start:end]


# plot vs time index
for col in T_rolling.loc[:, "T210_PV":"T225_PV"].columns:
    plt.plot(T_rolling.index, T_rolling[col], label=col)

plt.xlabel("Time")
plt.ylabel("Rolling Mean")
plt.title("test 1")
plt.legend()
plt.show()

# some constants

heat_capacity_glass = 0.840 #J/gK
density_glass = 2.2 #g/mL

# getting the right volume per tank
tank_void = 10 #mL (void part)
void_fraction = 0.1 # guessed for now
tank_volume = tank_void/void_fraction #mL
tank_surface = 0.0065 #m^2
number_of_tanks = 17
Boltzman_constant = 5.67*10**-8 #W/m^2K^-4

#T_outside = 20 # °C
T_in = 77 #K

def heatbalance(T, t_span, params):
    alpha, T_outside, epsilon = params
    dTdt = np.zeros(T.size)
    #first tank
    first_tank_volume = 10 * 10**-3 # L; measured as the volume of the zone before the beads
    first_tank_surface = 0.005 #m^2
    #dTdt[0] = tank_surface*alpha*(T_outside - T[0])/(first_tank_volume*density_glass*heat_capacity_glass)
    
    #remaining tanks
    for i in range(T.size):
        dTdt[i] = (tank_surface*alpha*(T_outside - T[i]) + Boltzman_constant*epsilon*(T_outside**4 - T[i]**4))/(tank_volume*density_glass*heat_capacity_glass)
    return dTdt


# setting a time span
total_time = T_window["T210_PV"].size
time_increments = total_time
t_span = np.linspace(0, total_time, time_increments) #seconds
dt = total_time/(time_increments-1)

#T_steady_state = np.linspace(-178, -140, number_of_tanks)
T_steady_state = T_window.loc[start]["T210_PV":"T225_PV"]+273

alpha_dummy = 20 # W/m^2K

def model_T_flat(params, t_eval, T0):
    """Return flattened temperatures (len(t_eval)*n_tanks) for least_squares."""
    # odeint expects args as tuple; make sure alpha is a scalar
    sol = odeint(heatbalance, T0, t_eval, args=(params,))
    # sol shape = (len(t_eval), number_of_tanks)
    return sol.flatten()

def residuals(params_array, t_eval, T0, T_meas_flat):
    
   # if alpha < 0:
        # Give bigger residues when alpha is negative
   #     return 1e6 * np.ones_like(T_meas_flat)
    sim_flat = model_T_flat(params_array, t_eval, T0)
    return sim_flat - T_meas_flat

T_measured = T_window.loc[:, "T210_PV":"T225_PV"].values +273

T_meas_flat = T_measured.flatten()

# initial guess and bounds (alpha > 0)
alpha0 = 2000   # play with this
T_outside0 = 263
epsilon0 = 0
x0 = np.array([alpha0, T_outside0, epsilon0])
lower = [0,        253,    0.0]
upper = [20000.0,  323,    1.0]

res = least_squares(
    residuals,
    x0=x0,
    bounds=(lower, upper),
    args=(t_span, T_steady_state, T_meas_flat),
    method='trf',   # trust-region reflective, werkt goed met bounds
    xtol=1e-10,
    ftol=1e-10,
    gtol=1e-10,
    verbose=2
)

params_fit = res.x
print("Fitted parameters alpha, T_outside, epsilon =", params_fit)

# Residual variance estimate
n = len(res.fun)
p = len(res.x)
dof = max(1, n - p)
residual_variance = np.sum(res.fun**2) / dof

# Covariance matrix and standard deviations
J = res.jac
cov = np.linalg.inv(J.T @ J) * residual_variance
param_std = np.sqrt(np.diag(cov))

print("Fitted parameters:", params_fit)
print("Standard deviations:", param_std)

# genereer model met gefitte alpha
T_sim = model_T_flat(params_fit, t_span, T_steady_state).reshape(len(t_span), number_of_tanks-1)

#plotting the CSTR-in-series
plt.subplots(4,4,figsize=(14,8))
for i in range(number_of_tanks-1):
    plt.subplot(4,4,i+1)
    plt.plot(t_span, T_sim[:,i], label="model", marker=',', color='b')
    plt.plot(t_span, T_measured[:,i], label='measured', marker=',', color='r')
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [°C]")
    plt.title(f"Tank {i+1}")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
#plt.text(600, 600, f'alpha = {params_fit[0]:.2f}', fontsize=22)
alpha_fit, T_outside_fit, epsilon_fit = params_fit
