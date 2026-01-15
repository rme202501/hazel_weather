import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import os

def convert_boston_to_utc(input_csv, output_csv=None):
    """
    Convert date and time from Boston (Eastern Time) to UTC.
    
    Args:
        input_csv: Path to input CSV file with 'date' and 'time' columns
        output_csv: Path to output CSV file (optional, defaults to input_utc.csv)
    """
    # Set default output filename if not provided
    if output_csv is None:
        base_name = os.path.splitext(input_csv)[0]
        output_csv = f"{base_name}_utc.csv"
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Define timezones
    boston_tz = ZoneInfo("America/New_York")
    utc_tz = ZoneInfo("UTC")
    
    # Lists to store converted values
    utc_dates = []
    utc_times = []
    
    for idx, row in df.iterrows():
        # Get date and time from row (handle different column name cases)
        date_col = None
        time_col = None
        
        for col in df.columns:
            if col.lower() == 'date':
                date_col = col
            elif col.lower() == 'time':
                time_col = col
        
        if date_col is None or time_col is None:
            raise ValueError("CSV must have 'date' and 'time' columns")
        
        date_str = str(row[date_col])
        time_str = str(row[time_col])
        
        # Pad time with leading zeros if needed (e.g., 800 -> 0800)
        time_str = time_str.zfill(4)
        
        # Parse the date and time
        # Try different date formats
        date_formats = [
            "%Y-%m-%d",  # 2025-12-26
            "%m/%d/%Y",  # 12/26/2025
            "%m-%d-%Y",  # 12-26-2025
            "%d/%m/%Y",  # 26/12/2025
            "%Y/%m/%d",  # 2025/12/26
        ]
        
        parsed_date = None
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        
        if parsed_date is None:
            raise ValueError(f"Could not parse date: {date_str}")
        
        # Parse time (military format: HHMM or HH:MM)
        time_str_clean = time_str.replace(":", "")
        time_str_clean = time_str_clean.zfill(4)
        
        try:
            hours = int(time_str_clean[:2])
            minutes = int(time_str_clean[2:4])
        except ValueError:
            raise ValueError(f"Could not parse time: {time_str}")
        
        # Create Boston datetime
        boston_dt = datetime(
            parsed_date.year,
            parsed_date.month,
            parsed_date.day,
            hours,
            minutes,
            tzinfo=boston_tz
        )
        
        # Convert to UTC
        utc_dt = boston_dt.astimezone(utc_tz)
        
        # Format output
        utc_dates.append(utc_dt.strftime("%Y-%m-%d"))
        utc_times.append(utc_dt.strftime("%H%M"))  # Military time format
    
    # Create output dataframe
    # Keep all original columns and add UTC columns
    output_df = df.copy()
    output_df['date_utc'] = utc_dates
    output_df['time_utc'] = utc_times
    
    # Save to CSV
    output_df.to_csv(output_csv, index=False)
    print(f"Converted {len(df)} rows from Boston time to UTC")
    print(f"Output saved to: {output_csv}")
    
    return output_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_utc.py <input_csv> [output_csv]")
        print("Example: python convert_to_utc.py weather_data.csv weather_data_utc.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_boston_to_utc(input_file, output_file)
