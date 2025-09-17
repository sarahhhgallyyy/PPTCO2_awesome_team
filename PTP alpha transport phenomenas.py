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
    heat_transfer_alpha = np.zeros(number_of_tanks)
    #first tank
    T_outside = 20 #°C
    first_tank_volume = 80 * 10**-3 # L; measured as the volume of the zone before the beads
    first_tank_surface = 0.05 #m^2
    tank_surface = 0.03**2*math.pi
    adiabatic_loss_per_tank[0] = first_tank_volume*thermal_mass*(T[0] - T_in)
    heat_transfer_alpha[0] = adiabatic_loss_per_tank[0]/((T_outside - T[0])*first_tank_surface)
    #remaining tanks
    for i in range(1, number_of_tanks):
        adiabatic_loss_per_tank[i] = tank_volume*thermal_mass*(T[i] - T[i-1]) #W
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
tank_volume = 10 * 10**-3 #L

#selecting the right temperatures
target_time = pd.Timestamp("2025-09-15 14:48:25")
temperature_steady_state = data.loc[target_time]["T210_PV" : "T226_PV"].values
Tin = data.loc[target_time]["T201_PV"]
heat_loss, alpha = heatbalance(temperature_steady_state, Tin, tank_volume, thermal_mass_flow_combined)

# plotting the heat losses
tanks = range(1, number_of_tanks)
plt.plot(tanks, heat_loss[1:], label="heat loss")
plt.plot(tanks, alpha[1:], label="alpha*A")
plt.xlabel("tank number")
plt.ylabel("heat loss [W] / alpha*A [W/K]")
plt.legend()

#isolate heat transfer coefficient (alpha)
def conduction_from_qdiff(T_wall_C, T_inf_C, alpha_total, area_wall, q_isolated):
    T_wall_C = data.loc[target_time, "T210_PV":"T226_PV"].values.tolist() #list of 17 tank wall temperatures [°C]
    T_inf_C = 20 #[°C]
    alpha_total = alpha.tolist() #list of 17 total heat transfer coeffs (for q_tot) [W/m^2/K]
    area_wall = 0.05 #m^2
    q_isolated = heat_loss.tolist() #list of 17 measured isolated losses (rad+conv) [W]
    n = 17
    q_cond, U_cond = [0.0]*n, [0.0]*n
    T_inf_K = T_inf_C + 273.15
    for i in range(n):
        dT = (T_wall_C[i] + 273.15) - T_inf_K
        q_tot = alpha_total[i] * area_wall[i] * dT
        q_cond[i] = q_tot - q_isolated[i]
        U_cond[i] = q_cond[i]/(area_wall[i]*dT) if abs(area_wall[i]*dT) > 1e-12 else 0.0
    return q_cond, U_cond