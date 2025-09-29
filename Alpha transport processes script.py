#structure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def conduction_from_qdiff(T_wall_C, T_inf_C, alpha_total, area_wall, q_isolated):
    T_wall_C    : [] #list of 17 tank wall temperatures [°C]
    T_inf_C     : 20 #[°C]
    alpha_total : heat_transfer_alpha[i] #list of 17 total heat transfer coeffs (for q_tot) [W/m^2/K]
    area_wall   : 0.05 #m^2
    q_isolated  : [] #list of 17 measured isolated losses (rad+conv) [W]
    n = 17
    q_cond, U_cond = [0.0]*n, [0.0]*n
    T_inf_K = T_inf_C + 273.15
    for i in range(n):
        dT = (T_wall_C[i] + 273.15) - T_inf_K
        q_tot = alpha_total[i] * area_wall[i] * dT
        q_cond[i] = q_tot - q_isolated[i]
        U_cond[i] = q_cond[i]/(area_wall[i]*dT) if abs(area_wall[i]*dT) > 1e-12 else 0.0
    return q_cond, U_cond
