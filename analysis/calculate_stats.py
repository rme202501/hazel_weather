import pandas as pd

data = pd.read_csv('nohistory_rolling_average.csv')

co2_mean = data['co2_rolling'].mean()
co2_median = data['co2_rolling'].median()
co2_std = data['co2_rolling'].std(ddof=0)
um_0_3_mean = data['0.3um_rolling'].mean()
um_0_3_median = data['0.3um_rolling'].median()
um_0_3_std = data['0.3um_rolling'].std(ddof=0)
print("CO2 Statistics:")
print(f"Mean: {co2_mean}")
print(f"Median: {co2_median}")
print(f"Standard Deviation: {co2_std}")
print("\n0.3um Particle Count Statistics:")
print(f"Mean: {um_0_3_mean}")
print(f"Median: {um_0_3_median}")
print(f"Standard Deviation: {um_0_3_std}")

# CO2 Statistics:
# Mean: 412.58824296779454
# Median: 409.0
# Standard Deviation: 28.33334441247773

# 0.3um Particle Count Statistics:
# Mean: 1105.3120260905014
# Median: 877.0
# Standard Deviation: 1265.42884076411