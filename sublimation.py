# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 10:26:32 2025

@author: 20223544
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Model1 import heat_loss
import matplotlib.pyplot as plt
#from Model1 import temperature_steady_state
path = r"\\stfiler\exchange\TUE_Exchange\CO2 cryogenic\Sara, Stan, Tony\Day 2\Day 2 2.5CO2 17.5N2 attempt 2.csv"

# read correctly: EU numbers + parse first col as datetime
data = pd.read_csv(path, sep=',', decimal=',', parse_dates=[0])
data = data.rename(columns={data.columns[0]: 'DateTime'}).set_index('DateTime')

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

enthalpy_of_sublimation_CO2 = 591 * 10**3 #J/kg
sublimation_point_CO2 = -78 #K

# getting the right thermal mass
heat_capacity_N2 = 1040 # J/kgK
density_N2 = 1.16 * 10**-3 #kg/Ln
volume_flow_N2_mins = data.values[10][1] #Ln/min
volume_flow_N2_sec = volume_flow_N2_mins / 60 #Ln/sec
thermal_mass_flow_N2 = heat_capacity_N2 * density_N2 * volume_flow_N2_sec #W/K

heat_capacity_CO2 = 735 # J/kgK @200K (near sublimation point)
density_CO2 = 1.815 * 10**-3 #kg/Ln
volume_flow_CO2_mins = data.values[2000][3] #Ln/min
volume_flow_CO2_sec = volume_flow_CO2_mins / 60 #Ln/sec
thermal_mass_flow_CO2 = heat_capacity_CO2 * density_CO2 * volume_flow_CO2_sec #W/K

thermal_mass_flow_combined = thermal_mass_flow_CO2 + thermal_mass_flow_N2 #W/K

sublimation_heat_CO2 = enthalpy_of_sublimation_CO2 * volume_flow_CO2_sec * density_CO2 #W

# getting the right volume per tank
tank_volume = 10 * 10**-3 #L
number_of_tanks = 16

def heatbalance(T, heat_loss, T_in, tank_volume, thermal_mass, sublimation_heat):
    dqdt = np.zeros(number_of_tanks)
    #first tank
    first_tank_volume = 50 * 10**-3 # L; measured as the volume of the zone before the beads
    dqdt[0] = first_tank_volume*thermal_mass*(T[0] - T_in) - heat_loss[0] - sublimation_heat[0]
    
    #remaining tanks
    for i in range(1, number_of_tanks):
        dqdt[i] = tank_volume*thermal_mass*(T[i] - T[i-1]) - heat_loss[i] - sublimation_heat[i]
    return dqdt

T_in = -196

# setting a time span
total_time = 1500
time_increments = 1501
t_span = np.linspace(0, total_time, time_increments) #seconds
dt = total_time/(time_increments-1)

# modeling the heat of sublimation
def sublimation(CO2_mass_flow, thermal_mass):
    """Returns the temperature of each tank at each point in time. 
    Driving force: heat of sublimation and heating by environment."""
    # set a starting temperature (should be steady state temperature in the future)
    T = [[-170]*16 for i in range(t_span.size)]
    # set first tank to high starting temperature (because that implies that CO2 is already sublimated there)
    for i in range(t_span.size):
        T[i][0] = 0
    # loop over all timestamps over all tanks
    for j in range(t_span.size):
        for i in range(1, number_of_tanks):
            """If the previous tank is already sublimated and the current tank is not,
            then add the heat of sublimation to the current tank.
            Always add heating by the environment (as experimentally determined at steady state).
            NB: this heat loss is now treated as a constant for each tank, while it should be a
            function of Delta_T."""
            if T[j][i-1] >= sublimation_point_CO2 and T[j-1][i] < sublimation_point_CO2:
                T[j][i] = T[j-1][i] + (sublimation_heat_CO2/thermal_mass)*dt + heat_loss[i] # need to add accumulated CO2 as well
            else:
                T[j][i] = T[j-1][i] + heat_loss[i]
    return T
thermal_mass_system = 50
Temp_dist = sublimation(density_CO2 * volume_flow_CO2_sec, thermal_mass_flow_combined + thermal_mass_system)

#plotting the CSTR-in-series
plt.plot(t_span, Temp_dist)
plt.legend(["tank 1", "tank 2", "tank 3", "tank 4", "tank 5", "etc"], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()