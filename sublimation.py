import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import least_squares
from Heating_up import alpha_fit as alpha 
from Heating_up import T_outside_fit as T_outside
from Heating_up import epsilon_fit as epsilon
from scipy.interpolate import interp1d, PchipInterpolator
#from Model1 import temperature_steady_state
path = r"Data\Day 5 foil and 20 N2 0.06 CO2.csv"

# read correctly: EU numbers + parse first col as datetime
data = pd.read_csv(path, sep=',', decimal=',', parse_dates=[0])
data = data.rename(columns={data.columns[0]: 'DateTime'}).set_index('DateTime')

# ensure all remaining columns are numeric
data = data.apply(pd.to_numeric, errors='coerce')
window_size = 32
T_rolling = data.rolling(window=window_size).mean()
start = pd.Timestamp("2025-09-17 09:52:49.114000")
end = pd.Timestamp("2025-09-17 10:29:20.114000")
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

enthalpy_of_sublimation_CO2 = 591 * 10**3 *5 #J/kg (delete the 10)
sublimation_point_CO2 = -100 #C
heat_capacity_glass = 0.840 #J/gK
density_glass = 2.2 #g/mL

# getting the right thermal mass
heat_capacity_N2 = 1040 # J/kgK
density_N2 = 1.16 * 10**-3 #kg/Ln
volume_flow_N2_mins = T_window.values[100][0] #Ln/min
volume_flow_N2_sec = volume_flow_N2_mins / 60 #Ln/sec
thermal_mass_flow_N2 = heat_capacity_N2 * density_N2 * volume_flow_N2_sec #W/K

heat_capacity_CO2 = 735 # J/kgK @200K (near sublimation point)
density_CO2 = 1.815 * 10**-3 #kg/Ln
volume_flow_CO2_mins = T_window.values[50][2] #Ln/min
volume_flow_CO2_sec = volume_flow_CO2_mins / 60 #Ln/sec
thermal_mass_flow_CO2 = heat_capacity_CO2 * density_CO2 * volume_flow_CO2_sec #W/K

thermal_mass_flow_combined = thermal_mass_flow_CO2 + thermal_mass_flow_N2 #W/K

sublimation_heat_CO2 = enthalpy_of_sublimation_CO2 * volume_flow_CO2_sec * density_CO2 #W

# getting the right volume per tank
tank_surface = 0.0065 #m^2
trunk = 3
N_sim = 12
number_of_tanks = N_sim + trunk
tank_void = 160/N_sim #mL (void part)
void_fraction = 0.1 # guessed for now
Boltzman_constant = 5.67*10**-8 #W/m^2K^-4

#T_outside = 20 # °C
#T_in = 0 #°C



def heatbalance(T, m, t_span, params):
    alpha_conv, T_in, first_tank_volume, void_fraction, mass_cap = params
    dTdt = np.zeros_like(T)
    dmdt = np.zeros_like(m)
    tank_volume = tank_void/void_fraction #mL
    #first tank
    #first_tank_volume = 2000 * 10**-3 # L; measured as the volume of the zone before the beads
 #   first_tank_surface = 0.005 #m^2
    dTdt[0] = (tank_surface*alpha*(T_outside - T[0]) + Boltzman_constant*epsilon*(T_outside**4 - T[0]**4) + alpha_conv*(T_in - T[0]))/(first_tank_volume/trunk*density_glass*heat_capacity_glass)
    for i in range (1, trunk):
        """If the previous tank is already sublimated and the current tank is not,
        then add the heat of sublimation to the current tank.
        Always add heating by the environment (as experimentally determined at steady state).
        NB: this heat loss is now treated as a constant for each tank, while it should be a
        function of Delta_T."""
        if T[i] <= sublimation_point_CO2 and T[i-1] > sublimation_point_CO2 and m[i] < mass_cap:
            dTdt[i] = (tank_surface*alpha*(T_outside - T[i]) + Boltzman_constant*epsilon*(T_outside**4 - T[i]**4) + alpha_conv*(T[i-1] - T[i]) + sublimation_heat_CO2)/(first_tank_volume*tank_volume*density_glass*heat_capacity_glass/trunk)
            dmdt[i] = volume_flow_CO2_sec * density_CO2
        elif T[i] > sublimation_point_CO2 and m[i] > 0:
            dTdt[i] = (tank_surface*alpha*(T_outside - T[i]) + Boltzman_constant*epsilon*(T_outside**4 - T[i]**4) + alpha_conv*(T[i-1] - T[i]) + -sublimation_heat_CO2)/(first_tank_volume*tank_volume*density_glass*heat_capacity_glass/trunk)
            dmdt[i] = -volume_flow_CO2_sec * density_CO2
        else:
            dTdt[i] = (tank_surface*alpha*(T_outside - T[i]) + Boltzman_constant*epsilon*(T_outside**4 - T[i]**4) + alpha_conv*(T[i-1] - T[i]))/(first_tank_volume*tank_volume*density_glass*heat_capacity_glass/trunk)
            dmdt[i] = 0
    for i in range(trunk, T.size):
        if T[i] <= sublimation_point_CO2 and T[i-1] > sublimation_point_CO2 and m[i] < mass_cap:
            dTdt[i] = (tank_surface*alpha*(T_outside - T[i]) + Boltzman_constant*epsilon*(T_outside**4 - T[i]**4) + alpha_conv*(T[i-1] - T[i]) + sublimation_heat_CO2)/(tank_volume*density_glass*heat_capacity_glass)
            dmdt[i] = volume_flow_CO2_sec * density_CO2
        elif T[i] > sublimation_point_CO2 and m[i] > 0:
            dTdt[i] = (tank_surface*alpha*(T_outside - T[i]) + Boltzman_constant*epsilon*(T_outside**4 - T[i]**4) + alpha_conv*(T[i-1] - T[i]) - sublimation_heat_CO2)/(tank_volume*density_glass*heat_capacity_glass)
            dmdt[i] = -volume_flow_CO2_sec * density_CO2
        else:
            dTdt[i] = (tank_surface*alpha*(T_outside - T[i]) + Boltzman_constant*epsilon*(T_outside**4 - T[i]**4) + alpha_conv*(T[i-1] - T[i]))/(tank_volume*density_glass*heat_capacity_glass)
            dmdt[i] = 0
    return dTdt, dmdt

delta = 1e-6

lambda_reg = 1e-6
params_prior = np.array([2, 0, 0, 0])

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
measured_indices = np.array(np.concatenate(([0], np.arange(trunk, number_of_tanks-1))))


#T_steady_state_values = T_steady_state.values
#print(measured_indices.shape, T_steady_state.values.shape)
T_measured = T_window.loc[:, "T210_PV":"T225_PV"].values

float_index = np.linspace(0, 14.9999, N_sim)
T_interpolated = []
T_interpolated.append(T_measured[:,0])
for i in range(1,N_sim):
    #interpolate N_meas
    T_interpolated.append(T_measured[:,int(float_index[i])] + (T_measured[:,int(float_index[i])+1]-T_measured[:,int(float_index[i])])*(float_index[i] - int(float_index[i])))

T_interpol = np.transpose(T_interpolated)
plt.plot(t_span, np.transpose(T_interpolated))

T_steady_state = T_interpol[0,:]
# 1B: PCHIP (monotone cubic) — voorkomt overshoot en 'waviness'
pchip = PchipInterpolator(measured_indices, T_steady_state, extrapolate=False)
T0_pchip = pchip(np.arange(number_of_tanks-1))

alpha_dummy = 20 # W/m^2K

def model_T(params, t_eval, T0):
    """Return temperatures (len(t_eval)*n_tanks) for least_squares."""
    # odeint expects args as tuple; make sure alpha is a scalar
    sol = odeint(heatbalance, T0, t_eval, args=(params,))
    
    # sol shape = (len(t_eval), number_of_tanks)
    return sol

def residuals(params_array, t_eval, T0, T_measured):
    
   # if alpha < 0:
        # Give bigger residues when alpha is negative
   #     return 1e6 * np.ones_like(T_meas_flat)
    sim = model_T(params_array, t_eval, T0)
    sim_measured = sim[:, measured_indices[1:]]
   # print(sim_measured.shape)
   # sim_measured_flat = sim_measured.flatten()
    res = sim_measured.flatten() - T_interpol[:, 1:].flatten()
    # print("Residuals preview:", res[500:510])   # eerste 10 waardes
    # print("Shapes:", sim_measured.flatten().shape, T_measured.flatten().shape)
    # print("sim_measured:", sim_measured)
    return res


#T_meas_flat = T_measured.flatten()

# initial guess and bounds (alpha > 0)
alpha0 = 1   # play with this
T_in0 = 10
first_tank_volume0 = 1
void_fraction0 = 0.1
mass_cap0 = 0.005
x0 = np.array([alpha0, T_in0, first_tank_volume0, void_fraction0, mass_cap0])
lower = [0,     -10,    0,  0,  0]
upper = [2.0,    20,    10, 1,  1]

res = least_squares(
    residuals_reg,
    x0=x0,
    bounds=(lower, upper),
    args=(t_span, T0_pchip, T_measured),
    method='trf',   # trust-region reflective, werkt goed met bounds
    xtol=1e-10,
    ftol=1e-10,
    gtol=1e-10,
    verbose=2
)

params_fit = res.x
print("Fitted parameters alpha_conv, T_in, first_tank_volume, void_fraction=", params_fit)
params_fit = res.x

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
T_sim = model_T(params_fit, t_span, T0_pchip).reshape(len(t_span), number_of_tanks-1)[:,measured_indices]

#plotting the CSTR-in-series
plt.subplots(4,3,figsize=(10,8))
for i in range(np.min([N_sim-1, 16])):
    plt.subplot(4,3,i+1)
    plt.plot(t_span, T_sim[:,i+1], label="model", marker=',', color='b')
    plt.plot(t_span, T_interpol[:,i+1], label='measured', marker=',', color='r')
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [°C]")
    plt.title(f"Tank {i+1}")
plt.subplot(4,3,12)
plt.plot(0,0,label="model", marker=',', color='b')
plt.plot(0,0,label='measured', marker=',', color='r')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
#plt.text(600, 600, f'alpha = {params_fit[0]:.2f}', fontsize=22)
p = res.x
r0 = residuals(p, t_span, T0_pchip, T_measured)
p2 = p.copy(); p2[0] += 1e-3
r1 = residuals(p2, t_span, T0_pchip, T_measured)
print("norm diff:", np.linalg.norm(r1-r0))