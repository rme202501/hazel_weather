import csv
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

def process_file(input_file, output_file):
    # Read all rows (including header if present)
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    if not rows:
        # Empty input; create empty output
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            pass
        print("Processed 0 rows")
        print(f"Output written to: {output_file}")
        return

    # Heuristic: treat first row as header if it contains alphabetic characters
    first_row = rows[0]
    has_header = any(ch.isalpha() for ch in "".join(first_row))
    start_idx = 1 if has_header else 0

    processed_rows = rows[:start_idx]  # keep header unchanged if present

    for row in rows[start_idx:]:
        if not row:
            processed_rows.append(row)
            continue
        value = row[0]
        if "-" not in value:
            try:
                row[0] = add_year_and_month(value)
            except ValueError:
                print(f"Warning: Could not convert value '{value}' to full date.")
                pass
        processed_rows.append(row)

    # Write all columns for all rows
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(processed_rows)

    print(f"Processed {len(rows) - start_idx} data rows")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python full_date.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_file(input_file, output_file)