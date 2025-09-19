# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 11:23:54 2025

@author: 20223544
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#from Model1 import temperature_steady_state
path = r"C:\Users\20223544\TU Eindhoven\Vullings, Stan - PPT\Data\Day 5\Day 5 foil 6.09 CO2 12.5 N2 finished, foil on.csv"

# read correctly: EU numbers + parse first col as datetime

enthalpy_of_sublimation_CO2 = 591 * 10**3 #J/kg
sublimation_point_CO2 = -78 #K

# getting the right thermal mass

heat_capacity_glass = 0.840 #J/gK
density_glass = 2.2 #g/mL

# getting the right volume per tank
tank_volume = 10 #mL
tank_surface = 0.005 #m^2
number_of_tanks = 16

T_outside = 20 # °C

def heatbalance(T, t_span, alpha):
    dTdt = np.zeros(number_of_tanks)
    #first tank
    first_tank_volume = 10 * 10**-3 # L; measured as the volume of the zone before the beads
    first_tank_surface = 0.005 #m^2
 #   dTdt[0] = first_tank_surface*alpha*(T_outside - T[0])/(first_tank_volume*density_glass*heat_capacity_glass)
    
    #remaining tanks
    for i in range(number_of_tanks):
        dTdt[i] = tank_surface*alpha*(T_outside - T[i])/(tank_volume*density_glass*heat_capacity_glass)
    return dTdt

T_in = -196

# setting a time span
total_time = 500
time_increments = 1501
t_span = np.linspace(0, total_time, time_increments) #seconds
dt = total_time/(time_increments-1)

T_steady_state = np.linspace(-178, -140, number_of_tanks)

alpha_dummy = 20 # W/m^2K

T = odeint(heatbalance, T_steady_state, t_span, args=(alpha_dummy,))



#plotting the CSTR-in-series
plt.plot(t_span, T)
plt.legend(["tank 1", "tank 2", "tank 3", "tank 4", "tank 5", "etc"], bbox_to_anchor=(1.05, 1), loc='upper left')
