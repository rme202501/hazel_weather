import pytesseract
import pandas as pd
import cv2
import re
from PIL import Image

# ---- CONFIG ----
IMAGE_PATH = "data/1-08/1-08weather.png"
OUTPUT_CSV = "weather_data.csv"

# If Tesseract is NOT in PATH (Windows example):
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---- LOAD & PREPROCESS IMAGE ----
img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Improve OCR accuracy
gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
gray = cv2.medianBlur(gray, 3)

# ---- OCR ----
text = pytesseract.image_to_string(gray, config="--psm 6")

# ---- PARSE ROWS ----
lines = [l.strip() for l in text.split("\n") if l.strip()]

rows = []
for line in lines:
    # Match rows starting with date + time (e.g., "09 11:54")
    if re.match(r"^\d{2}\s+\d{2}:\d{2}", line):
        parts = re.split(r"\s{2,}", line)
        rows.append(parts)

# ---- NORMALIZE COLUMNS ----
columns = [
    "Date", "Time", "Wind", "Visibility",
    "Weather", "Sky Conditions",
    "Temp_Air", "Temp_Dewpt",
    "Relative Humidity", "Wind_Chill",
    "Pressure_in", "Sea_Level_mb", "1hr_Precipitation", "3hr_Precipitation", "6hr_Precipitation"
]

cleaned_rows = []
for r in rows:
    if len(r) >= len(columns):
        cleaned_rows.append(r[:len(columns)])

df = pd.DataFrame(cleaned_rows, columns=columns)

# ---- SAVE ----
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved {len(df)} rows to {OUTPUT_CSV}")