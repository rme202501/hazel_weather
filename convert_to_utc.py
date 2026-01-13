import pandas as pd
import os
from datetime import datetime
import pytz

def convert_csv_to_utc(csv_file_path, output_file_path=None):
    """
    Convert EST date/time to UTC timestamps in a CSV file.
    
    Args:
        csv_file_path: Path to the input CSV file
        output_file_path: Path to save the output CSV (if None, adds _utc to filename)
    """
    
    # Read the CSV
    df = pd.read_csv(csv_file_path)
    
    # Extract year and month from the file path (e.g., "1-08bos" -> January)
    filename = os.path.basename(os.path.dirname(csv_file_path))
    parts = filename.split('-')
    month = int(parts[0])
    day = int(parts[1].replace('bos', '').replace('hnl', ''))
    
    # Assume year is 2026 (adjust if needed)
    year = 2026
    
    # Define timezones
    est = pytz.timezone('US/Eastern')
    utc = pytz.UTC
    
    # Convert each row
    utc_timestamps = []
    
    for index, row in df.iterrows():
        try:
            date_str = str(int(row['Date']))
            time_str = str(row['Time (EST)'])
            
            # Create datetime string in EST
            datetime_str = f"{year}-{month:02d}-{date_str} {time_str}"
            
            # Parse as EST
            dt_est = est.localize(datetime.strptime(datetime_str, "%Y-%m-%d %H:%M"))
            
            # Convert to UTC
            dt_utc = dt_est.astimezone(utc)
            
            # Store as ISO format string
            utc_timestamps.append(dt_utc.isoformat())
            
        except Exception as e:
            print(f"Error converting row {index}: {e}")
            utc_timestamps.append(None)
    
    # Add the UTC column
    df['UTC_Timestamp'] = utc_timestamps
    
    # Optionally reorder columns to put UTC timestamp first
    cols = ['UTC_Timestamp'] + [col for col in df.columns if col != 'UTC_Timestamp']
    df = df[cols]
    
    # Save output
    if output_file_path is None:
        base, ext = os.path.splitext(csv_file_path)
        output_file_path = f"{base}_utc{ext}"
    
    df.to_csv(output_file_path, index=False)
    print(f"Saved converted file to: {output_file_path}")
    
    return df

def batch_convert_all_csvs(base_dir):
    """
    Convert all CSV files in the data directory to UTC.
    
    Args:
        base_dir: Base directory containing subdirectories with CSV files
    """
    
    data_dir = os.path.join(base_dir, 'data')
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    # Find all CSV files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv') and 'extracted' in file:
                csv_path = os.path.join(root, file)
                print(f"Converting: {csv_path}")
                try:
                    convert_csv_to_utc(csv_path)
                except Exception as e:
                    print(f"Error processing {csv_path}: {e}")


if __name__ == "__main__":
    # Convert single file
    csv_file = r"c:\Users\ryane\OneDrive\Desktop\HazeL\hazel_weather\data\1-08bos\1-08extractedWeather.csv"
    convert_csv_to_utc(csv_file)
    
    # Or convert all extracted weather CSVs
    # base_dir = r"c:\Users\ryane\OneDrive\Desktop\HazeL\hazel_weather"
    # batch_convert_all_csvs(base_dir)
