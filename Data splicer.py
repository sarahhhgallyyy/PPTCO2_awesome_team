#This is the code to snip the data in seperate files for easier access

#importing relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path  

#Define the path of the file to be split down here
path = r"Cooling steady state 2.csv"

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

#The data is cut up in this section
#Define the begin and end point with the use of the plot
begin_point = 0
end_point = 10000
data_cut = T_rolling.iloc[:, begin_point:end_point]

#Check if the data is cut correctly
for col in data_cut.columns:
    plt.plot(data_cut.index, data_cut[col], label=col)

plt.xlabel("Time")
plt.ylabel("Rolling Mean")
plt.title("Cut data")
plt.legend()
plt.show()

#If the data is cut correctly, save it in a new file
filepath = Path('C:\Users\20212148\Desktop\Cooling steady state 2 cut.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
data_cut.to_csv(filepath)  
