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
Tin = -196

def heatbalance(T, T_in, tank_volume, heat_capacity):
    adiabatic_loss_per_tank = np.zeros(number_of_tanks)
    #first tank
    adiabatic_loss_per_tank[0] = tank_volume*heat_capacity*(T[0] - T_in)
    
    #remaining tanks
    for i in range(1, number_of_tanks):
        adiabatic_loss_per_tank[i] = tank_volume*heat_capacity*(T[i] - T[i-1])
    return adiabatic_loss_per_tank
#selecting the right temperatures
target_time = pd.Timestamp("2025-09-08 14:23:10.765000")
data_steady_state = data.loc[target_time]["T210_PV" : "T226_PV"].values
heat_loss = heatbalance(data_steady_state, Tin, 5, 5)
 
