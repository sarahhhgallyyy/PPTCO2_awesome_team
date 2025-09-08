import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\20210996\OneDrive - TU Eindhoven\Desktop\TUe\Year 4 2025-26\PPT Lab\trial day 2.csv"

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
