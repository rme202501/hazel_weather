import csv
from datetime import datetime, timedelta
import sys

def add_year_and_month(day):
    """
    Given a day of the month, returns a full date string in the format "YYYY-MM-DD"
    by adding a fixed year and month (2026-01).
    
    Args:
        day: Day of the month as an integer or string.
    """
    year = 2026
    month = 1
    day_int = int(day)
    return f"{year:04d}-{month:02d}-{day_int:02d}"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python full_date.py <input_file> <output_file>")
        sys.exit(1)
    
    first_column = []

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header if present
        for row in reader:
            first_column.append(row[0])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        for i in range(len(first_column)):
            if "-" in first_column[i]:
                continue
            else:
                first_column[i] = add_year_and_month(first_column[i])
        for i in range(len(first_column)):
            writer.writerow([first_column[i]])

    print(f"Processed {len(first_column)} rows")
    print(f"Output written to: {output_file}")

    
    