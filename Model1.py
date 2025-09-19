import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

path = r"C:\Users\20223544\OneDrive - TU Eindhoven\Documents\saus\PPT\Cooling steady state 2.csv"

# read correctly: EU numbers + parse first col as datetime
data = pd.read_csv(path, sep=',', decimal=',', parse_dates=[0])
data = data.rename(columns={data.columns[0]: 'DateTime'}).set_index('DateTime')

# ensure all remaining columns are numeric
data = data.apply(pd.to_numeric, errors='coerce')
window_size = 32
T_rolling = data.rolling(window=window_size).mean()

# plot vs time index
for col in T_rolling.columns:
    plt.plot(T_rolling.index, T_rolling[col], label=col)

plt.xlabel("Time")
plt.ylabel("Rolling Mean")
plt.title("test 1")
plt.legend()
plt.show()


# modeling the heat of a tank
number_of_tanks = 17
T_outside = 0 #°C
def conduction(T, T_in, tank_volume, thermal_mass):
    adiabatic_loss_per_tank = np.zeros(number_of_tanks)
    heat_transfer_alpha = np.zeros(number_of_tanks)
    #first tank
    first_tank_volume = 80 * 10**-3 # L; measured as the volume of the zone before the beads
    first_tank_surface = 0.05 #m^2
    tank_surface = 0.03**2*math.pi
    adiabatic_loss_per_tank[0] = thermal_mass*(T[0] - T_in) #W
    heat_transfer_alpha[0] = adiabatic_loss_per_tank[0]/((T_outside - T[0])*first_tank_surface) #W/m^2K
    #remaining tanks
    for i in range(1, number_of_tanks):
        adiabatic_loss_per_tank[i] = thermal_mass*(T[i] - T[i-1]) #W
        heat_transfer_alpha[i] = adiabatic_loss_per_tank[i]/((T_outside-T[i])*tank_surface) #W/m^2K
    return adiabatic_loss_per_tank, heat_transfer_alpha

# getting the right thermal mass
heat_capacity_N2 = 1040 # J/kgK
density_N2 = 1.16 * 10**-3 #kg/Ln
volume_flow_N2_mins = data.values[10][0] #Ln/min
volume_flow_N2_sec = volume_flow_N2_mins / 60 #Ln/sec
thermal_mass_flow_N2 = heat_capacity_N2 * density_N2 * volume_flow_N2_sec #W/K

heat_capacity_CO2 = 735 # J/kgK @200K (near sublimation point)
density_CO2 = 1.815 * 10**-3 #kg/Ln
volume_flow_CO2_mins = data.values[10][3] #Ln/min
volume_flow_CO2_sec = volume_flow_CO2_mins / 60 #Ln/sec
thermal_mass_flow_CO2 = heat_capacity_CO2 * density_CO2 * volume_flow_CO2_sec #W/K

thermal_mass_flow_combined = thermal_mass_flow_CO2 + thermal_mass_flow_N2 #W/K

# getting the right volume per tank
tank_volume = 10 * 10**-6 #m^3

#selecting the right temperatures
target_time = pd.Timestamp("2025-09-15 14:48:25")
temperature_steady_state = T_rolling.loc[target_time]["T210_PV" : "T226_PV"].values
Tin = T_rolling.loc[target_time]["T201_PV"]
heat_loss, alpha = conduction(temperature_steady_state, Tin, tank_volume, thermal_mass_flow_combined)

path_conv = r"C:\Users\20223544\OneDrive - TU Eindhoven\Documents\GitHub\PPTCO2_awesome_team\Data\Day 5 foil and 20 N2 0.06 CO2.csv"

# read correctly: EU numbers + parse first col as datetime
data_conv = pd.read_csv(path_conv, sep=',', decimal=',', parse_dates=[0])
data_conv = data_conv.rename(columns={data_conv.columns[0]: 'DateTime'}).set_index('DateTime')

# ensure all remaining columns are numeric
data_conv = data_conv.apply(pd.to_numeric, errors='coerce')
T_rolling_conv = data_conv.rolling(window=window_size).mean()

# plot vs time index
for col in T_rolling.columns:
    plt.plot(T_rolling_conv.index, T_rolling_conv[col], label=col)

plt.xlabel("Time")
plt.ylabel("Rolling Mean")
plt.title("test 1")
plt.legend()
plt.show()

boltzman_constant = 5.67*10**-8 #W/m^2K^4

def radiation(T, T_in, tank_volume, thermal_mass):
    radiation_per_tank = np.zeros(number_of_tanks)
    heat_transfer_epsilon = np.zeros(number_of_tanks)
    #first tank
    T_outside_kelvin = T_outside + 273 #K
    T_kelvin = T + 273 #K
    T_in_kelvin = T_in + 273 #K
    first_tank_volume = 80 * 10**-6 # m^3; measured as the volume of the zone before the beads
    first_tank_surface = 0.05 #m^2
    tank_surface = 0.04*math.pi*0.5/16 #m^2
    radiation_per_tank[0] = thermal_mass*(T[0] - T_in) - alpha[0]*(T_outside - T[0])*first_tank_surface - heat_trans_lambda[0]*(T_outside - T[0])*first_tank_surface #W
    heat_transfer_epsilon[0] = radiation_per_tank[0]/((T_outside_kelvin**4 - T_kelvin[0]**4)*first_tank_surface*boltzman_constant) # dimensionless
    #remaining tanks
    for i in range(1, number_of_tanks):
        radiation_per_tank[i] = thermal_mass*(T[i] - T[i-1]) -alpha[i]*(T_outside - T[i])*tank_surface - heat_trans_lambda[i]*(T_outside - T[i])*tank_surface #W
        heat_transfer_epsilon[i] = radiation_per_tank[i]/((T_outside_kelvin**4-T_kelvin[i]**4)*tank_surface*boltzman_constant) # dimensionless
    return radiation_per_tank, heat_transfer_epsilon

def convection(T, T_in, tank_volume, thermal_mass):
    conduction_per_tank = np.zeros(number_of_tanks)
    heat_transfer_lambda = np.zeros(number_of_tanks)
    #first tank
    first_tank_volume = 80 * 10**-6 # m^3; measured as the volume of the zone before the beads
    first_tank_surface = 0.05 #m^2
    tank_surface = 0.04*math.pi*0.5/16 #m^2
    conduction_per_tank[0] = thermal_mass*(T[0] - T_in) - alpha[0]*(T_outside - T[0])*first_tank_surface #W
    heat_transfer_lambda[0] = conduction_per_tank[0]/((T_outside - T[0])*first_tank_surface) # W/m^2K
    #remaining tanks
    for i in range(1, number_of_tanks):
        conduction_per_tank[i] = thermal_mass*(T[i] - T[i-1]) -alpha[i]*(T_outside - T[i])*tank_surface #W
        heat_transfer_lambda[i] = conduction_per_tank[i]/((T_outside-T[i])*tank_surface) # W/m^2K
    return conduction_per_tank, heat_transfer_lambda

target_time_conv = pd.Timestamp("2025-09-17 09:51:51.114000")
temperature_steady_state_conv = T_rolling_conv.loc[target_time_conv]["T210_PV" : "T226_PV"].values
Tin_conv = T_rolling_conv.loc[target_time_conv]["T201_PV"]

convection_conv, heat_trans_lambda = convection(temperature_steady_state_conv, Tin_conv, tank_volume, thermal_mass_flow_combined)

path_rad_no_foil= r"C:\Users\20223544\TU Eindhoven\Vullings, Stan - PPT\Data\Day 3\N2 12.5 CO2 7.5 Experiment 2.csv"

# read correctly: EU numbers + parse first col as datetime
data_rad_no_foil = pd.read_csv(path_rad_no_foil, sep=',', decimal=',', parse_dates=[0])
data_rad_no_foil = data_rad_no_foil.rename(columns={data_rad_no_foil.columns[0]: 'DateTime'}).set_index('DateTime')

# ensure all remaining columns are numeric
data_rad_no_foil = data_rad_no_foil.apply(pd.to_numeric, errors='coerce')
T_rolling_rad_no_foil = data_rad_no_foil.rolling(window=window_size).mean()

# plot vs time index
for col in T_rolling.columns:
    plt.plot(T_rolling_rad_no_foil.index, T_rolling_rad_no_foil[col], label=col)

plt.xlabel("Time")
plt.ylabel("Rolling Mean")
plt.title("test 1")
plt.legend()
plt.show()

boltzman_constant = 5.67*10**-8 #W/m^2K^4


target_time_rad_no_foil = pd.Timestamp("2025-09-10 11:29:21.434000")
temperature_steady_state_rad_no_foil = T_rolling_rad_no_foil.loc[target_time_rad_no_foil]["T210_PV" : "T226_PV"].values
Tin_rad_no_foil = T_rolling_rad_no_foil.loc[target_time_rad_no_foil]["T201_PV"]

radiation_no_foil, epsilon_no_foil = radiation(temperature_steady_state_rad_no_foil, Tin_rad_no_foil, tank_volume, thermal_mass_flow_combined)

# plotting the heat losses
tanks = range(1, number_of_tanks)
#plt.plot(tanks, heat_loss[1:], label="heat loss")
#plt.plot(tanks, alpha[1:], label="alpha*A")
plt.plot(tanks, convection_conv[1:], label="convection")
plt.plot(tanks, heat_trans_lambda[1:], label="alpha")
plt.plot(tanks, radiation_no_foil[1:], label="radiation no foil")
plt.plot(tanks, epsilon_no_foil[1:], label="epsilon no foil")
plt.xlabel("tank number")
plt.ylabel("heat loss [W] / alpha*A [W/K]")
plt.legend()
plt.show()
