"""
Data Preprocessing
- Split wind into prevailing direction & magnitude, gust direction & magnitude
- Split cloud cover tags into type and height (up to 4 cloud cover types)
- Strip % from RH
- Convert wind directions into numbers (map N, NE, E, SE, S, SW, W, NW to 0, 45, 90, 135, 180, 225, 270, 315)
- Convert cloud cover types into numbers (map CLR, FEW, SCT, BKN, OVC to 0, 1, 2, 3, 4)
"""

import pandas as pd
import numpy as np

# Load the data
import_path='data/cambridge/bosweather/bos_weather_utc.csv'
export_path='data/cambridge/bosweather/preprocessed_bos_weather_utc.csv'

print(f"Loading data from {import_path}...")
df = pd.read_csv(import_path)

# Inserting new columns for wind and cloud cover splits
print("\nInserting new columns for wind and cloud cover splits...")
id_wind = df.columns.get_loc('Wind (MPH)')
df.insert(loc=id_wind + 1, column='Prevailing Wind Direction', value=pd.Series(dtype='object'))
df.insert(loc=id_wind + 2, column='Prevailing Wind Magnitude (MPH)', value=pd.Series(dtype='float64'))
df.insert(loc=id_wind + 3, column='Gust Wind Direction', value=pd.Series(dtype='object'))
df.insert(loc=id_wind + 4, column='Gust Wind Magnitude (MPH)', value=pd.Series(dtype='float64'))

id_cloud = df.columns.get_loc('Sky Cond')
df.insert(loc=id_cloud + 1, column='Cloud Type 1', value=pd.Series(dtype='object'))
df.insert(loc=id_cloud + 2, column='Cloud Height 1 (100s of ft)', value=pd.Series(dtype='object'))
df.insert(loc=id_cloud + 3, column='Cloud Type 2', value=pd.Series(dtype='object'))
df.insert(loc=id_cloud + 4, column='Cloud Height 2 (100s of ft)', value=pd.Series(dtype='object'))
df.insert(loc=id_cloud + 5, column='Cloud Type 3', value=pd.Series(dtype='object'))
df.insert(loc=id_cloud + 6, column='Cloud Height 3 (100s of ft)', value=pd.Series(dtype='object'))
df.insert(loc=id_cloud + 7, column='Cloud Type 4', value=pd.Series(dtype='object'))
df.insert(loc=id_cloud + 8, column='Cloud Height 4 (100s of ft)', value=pd.Series(dtype='object'))

# Wind Splitting Into Array
print("\nSplitting wind into prevailing direction & magnitude, gust direction & magnitude...")
wind_column = df['Wind (MPH)']
for i in range(0, len(wind_column)):
    print(wind_column[i])
    split_wind_entry = wind_column[i].split()
    prev_dir_1 = np.nan
    prev_mag_1 = np.nan
    gust_dir_1 = np.nan
    gust_mag_1 = np.nan
    if len(split_wind_entry) == 1:
        prev_dir_1 = split_wind_entry[0]
    elif len(split_wind_entry) == 2:
        prev_dir_1 = split_wind_entry[0]
        prev_mag_1 = float(split_wind_entry[1]) if split_wind_entry[1].replace('.', '', 1).isdigit() else np.nan
    elif len(split_wind_entry) == 4:
        prev_dir_1 = split_wind_entry[0]
        prev_mag_1 = float(split_wind_entry[1]) if split_wind_entry[1].replace('.', '', 1).isdigit() else np.nan
        gust_dir_1 = split_wind_entry[2]
        gust_mag_1 = float(split_wind_entry[3]) if split_wind_entry[3].replace('.', '', 1).isdigit() else np.nan
    df.at[i, 'Prevailing Wind Direction'] = prev_dir_1
    df.at[i, 'Prevailing Wind Magnitude (MPH)'] = prev_mag_1
    df.at[i, 'Gust Wind Direction'] = gust_dir_1
    df.at[i, 'Gust Wind Magnitude (MPH)'] = gust_mag_1
    
print("\nWind splitting complete.")

# Cloud Cover Splitting Into Array
print("\nSplitting cloud cover tags into type and height...")
cloud_column = df['Sky Cond']
for i in range(0, len(cloud_column)):
    split_cloud_entry = cloud_column[i].split()
    cloud_types = []
    cloud_heights = []
    for j in range(len(split_cloud_entry)):
        cloud_types.append(split_cloud_entry[j][0:3])
        cloud_heights.append(split_cloud_entry[j][3:])
    
    df.at[i, 'Cloud Type 1'] = cloud_types[0] if len(cloud_types) > 0 else np.nan
    df.at[i, 'Cloud Height 1 (100s of ft)'] = cloud_heights[0] if len(cloud_heights) > 0 else np.nan
    df.at[i, 'Cloud Type 2'] = cloud_types[1] if len(cloud_types) > 1 else np.nan
    df.at[i, 'Cloud Height 2 (100s of ft)'] = cloud_heights[1] if len(cloud_heights) > 1 else np.nan
    df.at[i, 'Cloud Type 3'] = cloud_types[2] if len(cloud_types) > 2 else np.nan
    df.at[i, 'Cloud Height 3 (100s of ft)'] = cloud_heights[2] if len(cloud_heights) > 2 else np.nan
    df.at[i, 'Cloud Type 4'] = cloud_types[3] if len(cloud_types) > 3 else np.nan
    df.at[i, 'Cloud Height 4 (100s of ft)'] = cloud_heights[3] if len(cloud_heights) > 3 else np.nan

print("\nCloud cover splitting complete.")

# RH stripping
print("\nStripping % from RH values...")

# Convert 'Rel Hum' from percentage string to numeric (remove '%' sign)
if df['Rel Hum'].dtype == 'object':
    df['Rel Hum'] = df['Rel Hum'].str.replace('%', '').astype(float)

print("\nRH values successfully stripped.")

# Dropping original wind and cloud cover columns
df.drop(columns=['Wind (MPH)', 'Sky Cond'], inplace=True)
print("\nDropped original wind and cloud cover columns.")

df["cloud_code_1"] = df["Cloud Type 1"].astype("category").cat.codes
df["cloud_code_2"] = df["Cloud Type 2"].astype("category").cat.codes
df["cloud_code_3"] = df["Cloud Type 3"].astype("category").cat.codes
df["cloud_code_4"] = df["Cloud Type 4"].astype("category").cat.codes
print("\nConverted cloud cover types into numeric codes.")

df["prevailing_wind_dir_code"] = df["Prevailing Wind Direction"].astype("category").cat.codes
df["gust_wind_dir_code"] = df["Gust Wind Direction"].astype("category").cat.codes
print("\nConverted wind directions into numeric codes.")

df["weather_code"] = df["Weather"].astype("category").cat.codes
print("\nConverted weather descriptions into numeric codes.")

df = df.drop(columns=['Prevailing Wind Direction', 'Gust Wind Direction', 'Cloud Type 1', 'Cloud Type 2', 'Cloud Type 3', 'Cloud Type 4', 'Weather'])
print("\nDropped original categorical columns after encoding.")

# Save processed file
print(f"\nSaving preprocessed data to {export_path}...")
df.to_csv(export_path, index=False)