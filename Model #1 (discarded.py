import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = r"\\stfiler\exchange\TUE_Exchange\CO2 cryogenic\Sara, Stan, Tony\Day 2\trial day 2.csv"

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


# modeling the heat of a tank
number_of_tanks = 16

def heatbalance(T, T_in, tank_volume, thermal_mass):
    adiabatic_loss_per_tank = np.zeros(number_of_tanks)
    #first tank
    first_tank_volume = 50 * 10**-3 # L; measured as the volume of the zone before the beads
    adiabatic_loss_per_tank[0] = first_tank_volume*thermal_mass*(T[0] - T_in)
    
    #remaining tanks
    for i in range(1, number_of_tanks):
        adiabatic_loss_per_tank[i] = tank_volume*thermal_mass*(T[i] - T[i-1])
    return adiabatic_loss_per_tank

# getting the right thermal mass
heat_capacity_N2 = 1040 # J/kgK
density_N2 = 1.16 * 10**-3 #kg/Ln
volume_flow_N2_mins = data.values[10][1] #Ln/min
volume_flow_N2_sec = volume_flow_N2_mins / 60 #Ln/sec
thermal_mass_flow_N2 = heat_capacity_N2 * density_N2 * volume_flow_N2_sec #W/K

heat_capacity_CO2 = 735 # J/kgK @200K (near sublimation point)
density_CO2 = 1.815 * 10**-3 #kg/Ln
volume_flow_CO2_mins = data.values[10][3] #Ln/min
volume_flow_CO2_sec = volume_flow_CO2_mins / 60 #Ln/sec
thermal_mass_flow_CO2 = heat_capacity_CO2 * density_CO2 * volume_flow_CO2_sec #W/K

thermal_mass_flow_combined = thermal_mass_flow_CO2 + thermal_mass_flow_N2 #W/K

# getting the right volume per tank
tank_volume = 10 * 10**-3 #L

#selecting the right temperatures
target_time = pd.Timestamp("2025-09-08 14:23:10.765000")
temperature_steady_state = data.loc[target_time]["T210_PV" : "T226_PV"].values
Tin = data.loc[target_time]["T201_PV"]
heat_loss = heatbalance(temperature_steady_state, Tin, tank_volume, thermal_mass_flow_combined)

# plotting the heat losses
tanks = range(number_of_tanks)
plt.plot(tanks, heat_loss)
plt.xlabel("tank number")
plt.ylabel("heat loss [W]")


