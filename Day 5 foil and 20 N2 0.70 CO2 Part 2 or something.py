# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 10:36:30 2025

@author: 20223544
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 09:41:36 2025

@author: 20223544
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from Heating_up import alpha_fit as alpha 
from Heating_up import T_outside_fit as T_outside
from Heating_up import epsilon_fit as epsilon
from scipy.interpolate import PchipInterpolator


# for i in [1,7,11]:
#     plt.plot(t_span, T_sim_meas[:, [i]], color='b', alpha=(1-i/22))
#     plt.plot(t_span, T_interpol[:, [i]], color='c', alpha=(1-i/22))
# plt.xlabel("Time [s]")
# plt.ylabel("Temperature [K]")
# plt.title("Sublimation model for 12 tanks")
# plt.legend(["Model", "Measured"], loc='upper right', bbox_to_anchor = (1.15, 1))
# plt.show()

# df

# def reconstruct_flowrate(t, m_sim, rho_CO2):
#     dm_dt = np.gradient(m_sim, t, axis=0)
#     F = np.zeros_like(m_sim)
#     F[:, 0] = volume_flow_CO2_sec
#     for i in range(1, m_sim.shape[1]):
#         F[:, i] = F[:, i-1] - dm_dt[:, i] / rho_CO2
#     return F

# F_sim = reconstruct_flowrate(t_span, m_sim_meas, density_CO2)

# plt.plot(t_span[:int(t_span.size*.65)], F_sim[:int(t_span.size*.65), [1,4,8,12]], label="CO₂ flow per tank")
# plt.xlabel("Time [s]")
# plt.ylabel("CO₂ flowrate [Ln/s]")
# plt.title("0.017 Ln/s inflow, 12 tanks")
# plt.legend(["tank 1", "tank 4", "tank 8", "tank 12"])
# plt.show()
# kl

#from Model1 import temperature_steady_state
path = r"Data\Day 5 foil 6.09 CO2 12.5 N2 finished, foil on.csv"

# read correctly: EU numbers + parse first col as datetime
data = pd.read_csv(path, sep=',', decimal=',', parse_dates=[0])
data = data.rename(columns={data.columns[0]: 'DateTime'}).set_index('DateTime')

# ensure all remaining columns are numeric
data = data.apply(pd.to_numeric, errors='coerce')
window_size = 32
T_rolling = data.rolling(window=window_size).mean()
start = pd.Timestamp("2025-09-17 09:52:33")
end = pd.Timestamp("2025-09-17 10:30:10")
T_window = T_rolling.loc[start:end]


# plot vs time index
for col in T_rolling.columns:
    plt.plot(T_rolling.index, T_rolling[col], label=col)
    
plt.xlabel("Time")
plt.ylabel("Rolling Mean")
plt.title("test 1")
plt.legend()
plt.show()

# some constants

enthalpy_of_sublimation_CO2_mol = 26.3e3 #J/mol @ 167K source: https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=4
molar_mass_CO2 = 44e-3 #kg/mol source: chemists know this
enthalpy_of_sublimation_CO2 = enthalpy_of_sublimation_CO2_mol / molar_mass_CO2 #J/kg @ 167 K 
heat_capacity_glass = 880 #J/kgK source: https://en.wikipedia.org/wiki/Table_of_specific_heat_capacities
density_glass = 2.5e3 #kg/m^3 assuming fused silica. source: https://en.wikipedia.org/wiki/List_of_physical_properties_of_glass

# getting the right thermal mass
heat_capacity_N2 = 1039 # J/kgK source: https://www.engineeringtoolbox.com/nitrogen-d_977.html
density_N2 = 1.132 * 10**-3 #kg/Ln source: https://www.engineeringtoolbox.com/nitrogen-N2-density-specific-weight-temperature-pressure-d_2039.html
volume_flow_N2_mins = T_window.values[100][0] #Ln/min 
volume_flow_N2_sec = volume_flow_N2_mins / 60 #Ln/sec
thermal_mass_flow_N2 = heat_capacity_N2 * density_N2 * volume_flow_N2_sec #W/K

heat_capacity_CO2 = 735 # J/kgK @200K (near sublimation point)
density_CO2 = 1.784 * 10**-3 #kg/Ln source: https://www.engineeringtoolbox.com/carbon-dioxide-density-specific-weight-temperature-pressure-d_2018.html
volume_flow_CO2_mins = T_window.values[100][2] #Ln/min
volume_flow_CO2_sec = volume_flow_CO2_mins / 60 #Ln/sec
thermal_mass_flow_CO2 = heat_capacity_CO2 * density_CO2 * volume_flow_CO2_sec #W/K

thermal_mass_flow_combined = thermal_mass_flow_CO2 + thermal_mass_flow_N2 #W/K

def sublimation_heat_CO2(volume_flow_CO2):
    sub_heat = enthalpy_of_sublimation_CO2 * volume_flow_CO2_sec * density_CO2 #W
    return sub_heat

# getting the right volume per tank
tank_surface = 0.0065 #m^2
trunk = 3
N_sim = 12
number_of_tanks = N_sim + trunk
tank_void = 160/N_sim/10**6 #m^3 (void part)
tank_diameter = 0.03490 #m
tank_thickness = 3.26*10**-3 #m
wall_volume_total = ((tank_diameter / 2 + tank_thickness)**2 - (tank_diameter /2)**2) * math.pi * 0.03*17
tank_volume_total = tank_diameter**2*math.pi/4*(.030*17)
tank_volume = (tank_volume_total + wall_volume_total)/N_sim
#void_fraction = 0.1 # guessed for now
Boltzman_constant = 5.67*10**-8 #W/m^2K^-4
Factor_packing = 1 #taken up space of the beads

partial_pressure_CO2 = volume_flow_CO2_sec/(volume_flow_CO2_sec + volume_flow_N2_sec)
#sublimation_point_CO2 = 1301.679 / (6.81228 - math.log10(partial_pressure_CO2)) - 3.494 #source: https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=4&Type=ANTOINE&Plot=on
sublimation_point_CO2 = 163 #K
#T_outside = 20 # °C
#T_in = 0 #°C


def heat_and_massbalance(t, y, params):
    n = number_of_tanks
    T = y[:n]
    m = y[n:2*n]
    F = np.zeros_like(T)   # CO2_flowrate per tank [Ln/s]
    alpha_conv, T_in, first_tank_volume = params
    dTdt = np.zeros_like(T)
    dmdt = np.zeros_like(m)
    dFdt = np.zeros_like(F)
    #first tank
    #first_tank_volume = 2000 * 10**-3 # L; measured as the volume of the zone before the beads
 #   first_tank_surface = 0.005 #m^2
    dTdt[0] = (tank_surface*alpha*(T_outside - T[0]) + Boltzman_constant*epsilon*(T_outside**4 - T[0]**4) + alpha_conv*(T_in - T[0]))/(first_tank_volume*density_glass*heat_capacity_glass*Factor_packing/trunk)
    F[0] = volume_flow_CO2_sec
    # Trunk tanks
    for i in range(1, trunk):        
        # Calculate heat transfer
        heat_env = tank_surface*alpha*(T_outside - T[i]) + Boltzman_constant*epsilon*(T_outside**4 - T[i]**4)
        heat_conv = alpha_conv*(T[i-1] - T[i])*(heat_capacity_N2*density_N2*volume_flow_N2_sec + 
                                                 heat_capacity_CO2*density_CO2*F[i])
        heat_conv_no_CO2 = alpha_conv*(T[i-1] - T[i])*(heat_capacity_N2*density_N2*volume_flow_N2_sec)
        
        #desublimation
        if (T[i] < sublimation_point_CO2) and (F[i-1] > 0):
            dmdt[i] = (F[i-1] * density_CO2)
            F[i] = F[i-1] - dmdt[i] / density_CO2
            heat_desub = dmdt[i] * (enthalpy_of_sublimation_CO2 + (T[i] - sublimation_point_CO2)*heat_capacity_CO2)
            dTdt[i] = (heat_env + heat_conv_no_CO2 + heat_desub) / (first_tank_volume*density_glass*heat_capacity_glass*Factor_packing/trunk + m[i]*heat_capacity_CO2)
        #predesublimation
        elif T[i] < sublimation_point_CO2 and (F[i-2] == 0):
            dmdt[i] = 0
            dTdt[i] = 0
            F[i] = F[i-1] - dmdt[i] / density_CO2
        elif T[i] < sublimation_point_CO2 and F[i-1] == 0:
            dmdt[i] = 0
            F[i] = F[i-1] - dmdt[i] / density_CO2
            dTdt[i] = (heat_env + heat_conv_no_CO2 + heat_desub) / (first_tank_volume*density_glass*heat_capacity_glass*Factor_packing/trunk + m[i]*heat_capacity_CO2)
        #sublimation
        elif T[i] >= sublimation_point_CO2 and m[i] > 1e-4:
            dmdt[i] = -(heat_env + heat_conv) / enthalpy_of_sublimation_CO2 #kg/s
            dTdt[i] = 0 #K/s
            F[i] = F[i-1] - dmdt[i] / density_CO2
        #post-sublimation
        else:
            dmdt[i] = 0
            F[i] = F[i-1] - dmdt[i] / density_CO2
            dTdt[i] = (tank_surface*alpha*(T_outside - T[i]) + Boltzman_constant*epsilon*(T_outside**4 - T[i]**4) + alpha_conv*(T[i-1] - T[i])
                       *(heat_capacity_N2*density_N2*volume_flow_N2_sec + heat_capacity_CO2*density_CO2*F[i-1]))/(first_tank_volume*density_glass*heat_capacity_glass*Factor_packing/trunk + m[i]*heat_capacity_CO2)
    for i in range(trunk, T.size):
        # Calculate heat transfer
        heat_env = tank_surface*alpha*(T_outside - T[i]) + Boltzman_constant*epsilon*(T_outside**4 - T[i]**4)
        heat_conv = alpha_conv*(T[i-1] - T[i])*(heat_capacity_N2*density_N2*volume_flow_N2_sec + 
                                                 heat_capacity_CO2*density_CO2*F[i])
        heat_conv_no_CO2 = alpha_conv*(T[i-1] - T[i])*(heat_capacity_N2*density_N2*volume_flow_N2_sec)
        
        #desublimation
        if (T[i] <= sublimation_point_CO2) and (F[i-1] > 0):
            dmdt[i] = (F[i-1] * density_CO2)
            F[i] = F[i-1] - dmdt[i] / density_CO2
            heat_desub = dmdt[i] * (enthalpy_of_sublimation_CO2 + (T[i] - sublimation_point_CO2)*heat_capacity_CO2)
            dTdt[i] = (heat_env + heat_conv_no_CO2 + heat_desub) / (tank_volume*density_glass*heat_capacity_glass*Factor_packing + m[i]*heat_capacity_CO2)
        #predesublimation
        elif T[i] <= sublimation_point_CO2 and (F[i-3] <1e-3):
            dmdt[i] = 0
            dTdt[i] = 0
            F[i] = F[i-1] - dmdt[i] / density_CO2
        elif T[i] <= sublimation_point_CO2 and F[i-2] < 1e-3:
            dmdt[i] = 0
            F[i] = F[i-1] - dmdt[i] / density_CO2
            dTdt[i] = (heat_env + heat_conv) / (tank_volume*density_glass*heat_capacity_glass*Factor_packing + m[i]*heat_capacity_CO2)
        #sublimation
        # elif T[i] > sublimation_point_CO2 and m[i] > 1e-3:
        #     dmdt[i] = -(heat_env + heat_conv_no_CO2) / enthalpy_of_sublimation_CO2 #kg/s
        #     dTdt[i] = 0 #K/s
        #     CO2_flowrate[i] = CO2_flowrate[i-1] - (dmdt[i] / density_CO2) #Ln/s
        elif T[i] > sublimation_point_CO2 and m[i] > 0:
            dmdt[i] = -(heat_env + heat_conv) / enthalpy_of_sublimation_CO2#kg/s
            F[i] = F[i-1] - dmdt[i] / density_CO2
            heat_sub = dmdt[i] * (enthalpy_of_sublimation_CO2 + (T[i] - sublimation_point_CO2)*heat_capacity_CO2)
            dTdt[i] = (heat_env + heat_conv_no_CO2 + heat_sub) / (tank_volume*density_glass*heat_capacity_glass*Factor_packing + m[i]*heat_capacity_CO2)  #K/s
        #post-desublimation
        else:
            dmdt[i] = 0
            F[i] = F[i-1] - dmdt[i] / density_CO2
            dTdt[i] = (heat_env + heat_conv) / (tank_volume*density_glass*heat_capacity_glass*Factor_packing + m[i]*heat_capacity_CO2)
            
    return np.concatenate([dTdt, dmdt])

def reconstruct_flowrate(t, m_sim, rho_CO2):
    dm_dt = np.gradient(m_sim, t, axis=0)
    F = np.zeros_like(m_sim)
    F[:, 0] = volume_flow_CO2_sec
    for i in range(1, m_sim.shape[1]):
        F[:, i] = F[:, i-1] - dm_dt[:, i] / rho_CO2
    return F

def residuals_reg(params_array, t_eval, T0, T_measured):
    r = residuals(params_array, t_eval, T0, T_measured)
    reg = np.sqrt(lambda_reg) * (params_array - params_prior)
    return np.concatenate([r, reg])

# setting a time span
total_time = T_window["T210_PV"].size
time_increments = total_time
t_span = np.linspace(0, total_time, time_increments) #seconds
dt = total_time/(time_increments-1)

#T_steady_state = np.linspace(-178, -140, number_of_tanks)

#select which indices are actually measured against
measured_indices = np.array(np.concatenate(([0], np.arange(trunk, number_of_tanks))))


#T_steady_state_values = T_steady_state.values
#print(measured_indices.shape, T_steady_state.values.shape)
T_measured = T_window.loc[:, "T201_PV":"T225_PV"].values+273 #K

float_index = np.linspace(0, 14.9999, N_sim)
T_interpolated = []
T_interpolated.append(T_measured[:,1])
for i in range(1,N_sim):
    #interpolate N_meas
    T_interpolated.append(T_measured[:,int(float_index[i])] + (T_measured[:,int(float_index[i])+1]-T_measured[:,int(float_index[i])])*(float_index[i] - int(float_index[i])))

T_interpol = np.transpose(T_interpolated)
plt.plot(t_span, np.transpose(T_interpolated))
plt.xlabel("time [s]")
plt.ylabel("Temperature [K]")
plt.title("Interpolated tanks")

T_steady_state = np.concatenate([[T_measured[0,0]], T_interpol[0,:]])
# 1B: PCHIP (monotone cubic) — voorkomt overshoot en 'waviness'
pchip = PchipInterpolator(measured_indices, T_steady_state, extrapolate=False)
T0_pchip = pchip(np.arange(number_of_tanks))

mass_cap_init = 0.09

T0 = T0_pchip          # of je steady-state startwaarden
m0 = np.zeros(number_of_tanks)  # of een startmassa
#m0[0] = 0
y0 = np.concatenate([T0, m0])

alpha_dummy = 20 # W/m^2K

def model_T(params, t_eval, y0):
    """Return temperatures (len(t_eval) × n_states) for least_squares."""  
    sol = solve_ivp(
        lambda t, y: heat_and_massbalance(t, y, params),
        (t_eval[0], t_eval[-1]),
        y0,
        t_eval=t_eval,
        method='BDF',
        max_step=1.0,
        rtol=1e-6,
        atol=1e-8
    )

    if not sol.success:
        raise RuntimeError("ODE solver failed: " + str(sol.message))

    # sol.y has shape (n_states, n_times)
    return sol.y.T

def residuals(params_array, t_eval, y0, T_measured):
   # simulate
    sol = model_T(params_array, t_eval, y0)   # shape (nt, 2*N)
    T_sim = sol[:, :number_of_tanks]          # only temperatures
    # Ensure measured_indices are valid
    if np.any(np.array(measured_indices) < 0) or np.any(np.array(measured_indices) >= number_of_tanks):
        raise IndexError("measured_indices outside 0..number_of_tanks-1: " + str(measured_indices))
    sim_measured = T_sim[:, measured_indices[1:]]    # shape (nt, N_meas)
    # Compare to the same object you built for interpolation (T_interpol)
    if sim_measured.shape != T_interpol.shape:
        raise ValueError(f"Shape mismatch sim_measured {sim_measured.shape} vs T_interpol {T_interpol.shape}")
    res = sim_measured.flatten() - T_interpol.flatten()
    return res


#T_meas_flat = T_measured.flatten()

# initial guess and bounds (alpha > 0)
alpha0 = 0.01   # play with this
T_in0 = 283
first_tank_volume0 = 1e-6 #m^3
sublimation_speed_coefficient0 = 1e-2 #K^-1
mass_cap0 = mass_cap_init #kg
#enthalpy_of_sublimation_CO20 = 591 * 10**3 / 1#J/kg @ 180 K source: https://www.engineeringtoolbox.com/CO2-carbon-dioxide-properties-d_2017.html

x0 = np.array([alpha0, T_in0, first_tank_volume0])
lower = [0.0,   243,    0.1e-6]
upper = [2.0,   293,    100e-6]

params_prior = x0

delta = 1e-6

lambda_reg = 1e-6

fixed = x0.copy()
fixed_idx = [0,1,2]  # indexes to keep fixed
free_idx = [3]
def residuals_reduced(free_params, *args):
    p = fixed.copy()
    p[free_idx] = free_params[free_idx]
    return residuals(p, *args)

res = least_squares(
    residuals,
    x0=x0,
    bounds=(lower, upper),
    args=(t_span, y0, T_interpol),
    method='trf',   # trust-region reflective, werkt goed met bounds
    x_scale=[1,10,1],
    diff_step=[1e-5, 1e-3, 1e-4],
    xtol=1e-10,
    ftol=1e-10,
    gtol=1e-10,
    verbose=2
)

params_fit = res.x
print("Fitted parameters alpha_conv, T_in, first_tank_volume=", params_fit)

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
#T_sim = model_T(params_fit, t_span, y0)#.reshape(len(t_span), 2*number_of_tanks-1)[:,measured_indices]
sol_sim = model_T(params_fit, t_span, y0)   # shape (nt, 3*N)
n = number_of_tanks
T_sim = sol_sim[:, :n]
m_sim = sol_sim[:, n:]

T_sim_meas = T_sim[:, measured_indices]   # measured_indices must be in 0..number_of_tanks-1
m_sim_meas = m_sim[:, measured_indices]

# Plotting the CSTR-in-series: T (LHS axis) + m (RHS axis)
fig, axes = plt.subplots(4, 3, figsize=(10, 8))
axes = axes.flatten()  # easy index
for i in range(np.min([N_sim-1, 12])):  # max 12 subplots
    ax1 = axes[i]
    ax1.set_title(f"Tank {i+1}")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Temperature [K]", color='b')
    ax1.plot(t_span, T_sim_meas[:, i+1], label="T_model", color='b')
    ax1.plot(t_span, T_interpol[:, i+1], label="T_measured", color='c',)
    ax1.tick_params(axis='y', labelcolor='b')

    # Secondary y-axis for mass
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mass [kg]", color='r')
    ax2.plot(t_span, m_sim_meas[:, i+1], label="m_model", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

total_mass = np.zeros_like(m_sim_meas[:, 0])
for i in range(np.min([N_sim-1, 12])):
    total_mass = total_mass + m_sim_meas[:, i]
# Last subplot voor legend
ax11 = axes[11]
ax11.set_title("Total mass")
ax11.plot(t_span, total_mass, color='g', label='m_model')
handles = [
    plt.Line2D([0], [0], color='b', label='T_model'),
    plt.Line2D([0], [0], color='c', label='T_measured'),
    plt.Line2D(t_span, total_mass, color='r', label='m_model')
]
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.show()


for i in range(1,7,11):
    plt.plot(t_span, T_sim_meas[:, [i]], label="T_model", color='b', alpha=(0.5+i/22))
    plt.plot(t_span, T_interpol[:, [i]], label="T_measured", color='c')
plt.xlabel("Time [s]")
plt.ylabel("Temperature [K]")
plt.title("Sublimation model for 12 tanks")
plt.legend(loc='upper right', bbox_to_anchor = (1.15, 1))
plt.show()


#F_sim_meas = F_sim[:, measured_indices]


# p = res.x
# r0 = residuals(p, t_span, T0_pchip, T_measured)
# p2 = p.copy(); p2[0] += 1e-3
# r1 = residuals(p2, t_span, T0_pchip, T_measured)
# print("norm diff:", np.linalg.norm(r1-r0)) 


#plt.text(600, 600, f'alpha = {params_fit[0]:.2f}', fontsize=22)

# _______AAAA____       ____AAAA________
#        VVVV               VVVV        
#        (__)               (__)
#         \ \               / /
#          \ \   \\|||//   / /
#           > \   _   _   / <
#   hang     > \ / \ / \ / <
#   in        > \\_o_o_// <
#   there...   > ( (_) ) <
#               >|     |<
#              / |\___/| \
#              / (_____) \
#              /         \
#               /   o   \
#                ) ___ (   
#               / /   \ \  
#              ( /     \ )
#              ><       ><
#             ///\     /\\\
#             '''       '''
"""/ (tank_volume*density_glass*heat_capacity_glass + m[i]*heat_capacity_CO2) """ 
