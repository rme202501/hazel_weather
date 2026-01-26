"""
Script to add a column to weatherid CSV files to allow training to use weather data from previous known weather conditions.
"""

import pandas as pd

# Load the data
import_path = 'rolling_average_new.csv'
export_path = 'history_aware_rolling_new.csv'

number_of_previous_conditions = 4  # Number of previous weather conditions to include

print(f"Loading data from {import_path}...")

df = pd.read_csv(import_path)

# Create new columns for previous weather conditions
for i in range(1, number_of_previous_conditions + 1):
    df[f'weather_idx_{i}'] = pd.Series(dtype='object')

for j in range(len(df)):
    if df['weather_idx'][j] < number_of_previous_conditions:
        # Not enough previous conditions, set to NaN
        for k in range(1, number_of_previous_conditions + 1):
            df.at[j, f'weather_idx_{k}'] = pd.NA
    else:
        current_weather_idx = df['weather_idx'][j]
        for k in range(1, number_of_previous_conditions + 1):
            previous_weather_idx = current_weather_idx - k
            df.at[j, f'weather_idx_{k}'] = previous_weather_idx

print("\nPrevious weather conditions added.")

# Save the updated DataFrame to a new CSV file
df.to_csv(export_path, index=False)
print(f"Data saved to {export_path}.")