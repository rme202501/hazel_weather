"""
Script to merge air pollution data (every 10 seconds) with hourly weather data.
Matches each air pollution reading to the nearest weather timestamp (within 30 min window).
"""

import pandas as pd
from datetime import datetime, timedelta
import os
import glob

def parse_pollution_timestamp(ts):
    """Parse the air pollution timestamp format: YYYY-M-DTHH:MM:SS+00:00"""
    # Remove timezone info for easier parsing
    ts_clean = ts.replace('+00:00', '')
    return datetime.strptime(ts_clean, '%Y-%m-%dT%H:%M:%S')

def parse_weather_datetime(date_str, time_str):
    """Parse weather date and time into datetime object"""
    return datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M')

def find_nearest_weather(pollution_dt, weather_datetimes):
    """
    Find the nearest weather timestamp within 30 minutes of the pollution reading.
    Returns the index of the nearest weather reading, or None if none within 30 min.
    """
    min_diff = timedelta(minutes=30)
    nearest_idx = None
    
    for idx, weather_dt in enumerate(weather_datetimes):
        diff = abs(pollution_dt - weather_dt)
        if diff <= min_diff:
            min_diff = diff
            nearest_idx = idx
    
    return nearest_idx

def merge_pollution_with_weather(pollution_csv, weather_csv, output_csv):
    """
    Merge air pollution data with weather data based on timestamp matching.
    Matches pollution readings to weather data within +/- 30 minutes.
    
    Args:
        pollution_csv: Path to air pollution CSV file
        weather_csv: Path to weather CSV file  
        output_csv: Path for output merged CSV file
    """
    # Read the CSV files
    pollution_df = pd.read_csv(pollution_csv)
    weather_df = pd.read_csv(weather_csv)
    
    # Parse pollution timestamps
    pollution_df['datetime'] = pollution_df['UTC_timestamp'].apply(parse_pollution_timestamp)
    
    # Parse weather timestamps
    weather_df['datetime'] = weather_df.apply(
        lambda row: parse_weather_datetime(row['date'], row['time']), axis=1
    )
    
    # Extract only the columns we need from pollution data
    pollution_subset = pollution_df[['UTC_timestamp', 'datetime', 'co2', '0.3um']].copy()
    
    # Get list of weather datetimes for matching
    weather_datetimes = weather_df['datetime'].tolist()
    
    # Find nearest weather index for each pollution reading
    pollution_subset['weather_idx'] = pollution_subset['datetime'].apply(
        lambda dt: find_nearest_weather(dt, weather_datetimes)
    )
    
    # Prepare weather data for merge (add index column)
    weather_df['weather_idx'] = range(len(weather_df))
    weather_for_merge = weather_df.drop(columns=['datetime'])
    
    # Convert weather_idx to nullable Int64 to handle None values properly
    pollution_subset['weather_idx'] = pollution_subset['weather_idx'].astype('Int64')
    
    # Merge on weather_idx
    merged_df = pollution_subset.merge(
        weather_for_merge,
        on='weather_idx',
        how='left'
    )
    
    # Reorder columns: pollution data first, then weather data
    # Keep UTC_timestamp, co2, 0.3um, then all weather columns
    weather_columns = [col for col in weather_df.columns if col not in ['datetime', 'weather_idx']]
    output_columns = ['UTC_timestamp', 'co2', '0.3um'] + weather_columns
    
    # Filter to only columns that exist
    output_columns = [col for col in output_columns if col in merged_df.columns]
    merged_df = merged_df[output_columns]
    
    merged_df = merged_df.drop(columns=['date', 'time'])

    # Save to CSV
    merged_df.to_csv(output_csv, index=False)
    print(f"Merged data saved to: {output_csv}")
    print(f"Total rows: {len(merged_df)}")
    print(f"Rows with weather data: {merged_df['UTC_timestamp'].notna().sum()}")
    print(f"Rows without weather data: {merged_df['UTC_timestamp'].isna().sum()}")

    return merged_df

def process_folder(pollution_folder, weather_csv, output_folder=None):
    """
    Process all pollution data files in a folder and merge with weather data.
    
    Args:
        pollution_folder: Path to folder containing pollution CSV files
        weather_csv: Path to weather CSV file
        output_folder: Path to output folder (defaults to pollution_folder)
    """
    if output_folder is None:
        output_folder = pollution_folder
    
    # Find all data CSV files (not meta files)
    pollution_files = glob.glob(os.path.join(pollution_folder, '*_data.csv'))
    
    for pollution_file in pollution_files:
        basename = os.path.basename(pollution_file)
        output_name = basename.replace('_data.csv', '_merged.csv')
        output_path = os.path.join(output_folder, output_name)
        
        print(f"\nProcessing: {basename}")
        merge_pollution_with_weather(pollution_file, weather_csv, output_path)


# Example usage
if __name__ == "__main__":
    # Option 1: Process a single file
    # pollution_file = r"c:\Users\ryane\OneDrive\Desktop\HazeL\hazel_weather\data\1-08bos\260108_213900_data.csv"
    # weather_file = r"c:\Users\ryane\OneDrive\Desktop\HazeL\hazel_weather\data\bosweather\bos_weather_utc.csv"
    # output_file = r"c:\Users\ryane\OneDrive\Desktop\HazeL\hazel_weather\data\1-08bos\260108_213900_merged.csv"
    # merge_pollution_with_weather(pollution_file, weather_file, output_file)
    
    # Option 2: Process all files in a folder
    # Uncomment and modify paths as needed:
    
    # Example for Boston data folders
    bos_weather = r"c:\Users\ryane\OneDrive\Desktop\HazeL\hazel_weather\data\cambridge\bosweather\preprocessed_bos_weather_utc.csv"
    
    # Process each Boston data folder
    data_base = r"c:\Users\ryane\OneDrive\Desktop\HazeL\hazel_weather\data\cambridge"
    bos_folders = [
        "1-08bos", "1-09bos", "1-11bos", "1-12bos", 
        "1-13bos", "1-14bos", "1-15bos", "1-16bos"
    ]
    
    for folder in bos_folders:
        folder_path = os.path.join(data_base, folder)
        if os.path.exists(folder_path):
            process_folder(folder_path, bos_weather, r"c:\Users\ryane\OneDrive\Desktop\HazeL\hazel_weather\data\cambridge\merged_data")
