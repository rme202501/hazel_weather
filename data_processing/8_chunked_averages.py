"""Chunk and average readings per weather index.

Configure the variables below, then run the script directly. It groups rows by
`weather_idx`, assumes rows are already time-ordered, splits into N contiguous chunks
in that order, and averages CO2 and 0.3um within each chunk.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# === User configuration ===
INPUT_PATH = Path("7_history_aware_output.csv")
OUTPUT_PATH = Path("8_chunked_output.csv")
CHUNK_COUNT = 6

DATE_COL = "date"            # used when DATETIME_COL is None
TIME_COL = "time"            # used when DATETIME_COL is None
DATETIME_COL = None           # set to column name if a single datetime column exists
WEATHER_COL = "weather_idx"
CO2_COL = "co2"
UM03_COL = "0.3um"


def validate_columns(df: pd.DataFrame, columns: List[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def chunk_group(
    group: pd.DataFrame,
    chunk_count: int,
    weather_col: str,
    co2_col: str,
    um03_col: str,
    datetime_col: str | None,
    date_col: str,
    time_col: str,
) -> List[dict]:
    if chunk_count < 1:
        raise ValueError("CHUNK_COUNT must be >= 1")
    splits = np.array_split(group, chunk_count)
    rows: List[dict] = []
    for idx, chunk in enumerate(splits, start=1):
        if chunk.empty:
            continue
        start_row = chunk.iloc[0]
        end_row = chunk.iloc[-1]
        
        # Extract start date/time
        if datetime_col and datetime_col in chunk.columns:
            start_dt = pd.to_datetime(start_row[datetime_col])
            end_dt = pd.to_datetime(end_row[datetime_col])
            start_date = start_dt.strftime("%Y-%m-%d")
            start_time = _time_to_decimal_hours(start_dt.time())
            end_date = end_dt.strftime("%Y-%m-%d")
            end_time = _time_to_decimal_hours(end_dt.time())
        else:
            start_date = str(start_row[date_col])
            end_date = str(end_row[date_col])
            start_time = float(start_row[time_col])
            end_time = float(end_row[time_col])
        
        # Calculate average time
        avg_time = (start_time + end_time) / 2
        
        # Determine average date (if times cross midnight, handle accordingly)
        if start_date == end_date:
            avg_date = start_date
        else:
            # For multi-day chunks, use the date of the midpoint
            start_dt_full = pd.to_datetime(f"{start_date} {_decimal_hours_to_time(start_time)}")
            end_dt_full = pd.to_datetime(f"{end_date} {_decimal_hours_to_time(end_time)}")
            avg_dt_full = start_dt_full + (end_dt_full - start_dt_full) / 2
            avg_date = avg_dt_full.strftime("%Y-%m-%d")
        
        # Build the row dictionary
        row_dict = {
            "weather_idx": chunk[weather_col].iloc[0],
            "chunk_id": idx,
            "count": len(chunk),
            "start_date": start_date,
            "start_time": start_time,
            "end_date": end_date,
            "end_time": end_time,
            "avg_date": avg_date,
            "avg_time": avg_time,
            "co2_mean": chunk[co2_col].mean(),
            "um03_mean": chunk[um03_col].mean(),
        }
        
        # Add historical weather indices if they exist
        for col in chunk.columns:
            if col.startswith('weather_idx_'):
                row_dict[col] = chunk[col].iloc[0]
        
        rows.append(row_dict)
    return rows


def _decimal_hours_to_time(decimal_hours: float) -> str:
    """Convert decimal hours (e.g., 14.5) to HH:MM:SS format."""
    hours = int(decimal_hours)
    minutes_decimal = (decimal_hours - hours) * 60
    minutes = int(minutes_decimal)
    seconds = int((minutes_decimal - minutes) * 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _time_to_decimal_hours(time_obj) -> float:
    """Convert time object to decimal hours."""
    return time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600



def run() -> None:
    df = pd.read_csv(INPUT_PATH)

    required_cols = [WEATHER_COL, CO2_COL, UM03_COL]
    if DATETIME_COL is None:
        required_cols += [DATE_COL, TIME_COL]
    validate_columns(df, required_cols)

    df = df.copy()
    df[CO2_COL] = pd.to_numeric(df[CO2_COL], errors="coerce")
    df[UM03_COL] = pd.to_numeric(df[UM03_COL], errors="coerce")

    results: List[dict] = []
    for _, group in df.groupby(WEATHER_COL, sort=False):
        results.extend(
            chunk_group(
                group,
                CHUNK_COUNT,
                WEATHER_COL,
                CO2_COL,
                UM03_COL,
                DATETIME_COL,
                DATE_COL,
                TIME_COL,
            )
        )

    if not results:
        print("No output generated (check input data and chunk count)")
        return

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values(["weather_idx", "chunk_id"]).reset_index(drop=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out_df)} chunk rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
