import csv
from datetime import datetime, timedelta
import sys

def add_5_hours_to_csv(input_file, output_file):
    """
    Reads a CSV file and adds 5 hours to the time in the 2nd column.
    If the time exceeds 24 hours, adds a day to the date in the 1st column 
    and adjusts the time accordingly.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file
    """
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = list(reader)
    
    # Process the data (skip header if present)
    modified_rows = []
    header = rows[0]
    modified_rows.append(header)
    
    for row in rows[1:]:
        if len(row) < 2:
            modified_rows.append(row)
            continue
        
        try:
            date_str = row[0]  # First column: date (e.g., "2026-01-08")
            time_str = row[1]  # Second column: time (e.g., "12:54")
            
            # Combine date and time into a datetime object
            datetime_str = f"{date_str} {time_str}"
            dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            
            # Add 5 hours
            new_dt = dt + timedelta(hours=5)
            
            # Split back into date and time
            row[0] = new_dt.strftime("%Y-%m-%d")
            row[1] = new_dt.strftime("%H:%M")
            
            modified_rows.append(row)
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not process row: {row}. Error: {e}")
            modified_rows.append(row)
    
    # Write the modified data to output file
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(modified_rows)
    
    print(f"Successfully processed {len(modified_rows)-1} rows")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_utc.py <input_file> <output_file>")
        print("Example: python convert_to_utc.py input.csv output.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    add_5_hours_to_csv(input_file, output_file)
