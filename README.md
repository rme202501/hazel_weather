# HazeL Weather

Analyzing the relationship between weather conditions and air pollution (CO2 and 0.3µm particle counts) in the Boston/Cambridge area using machine learning and statistical methods.

## Overview

This project investigates how meteorological variables influence indoor/outdoor air quality metrics. It collects air pollution sensor data (CO2 concentration and 0.3µm particle counts), merges it with hourly weather observations from NOAA, and trains XGBoost regression models to predict pollution levels from weather features.

## Data Processing Pipeline

The data goes through an 8-step processing pipeline (documented in [data_processing/pipeline.txt](data_processing/pipeline.txt)):

1. **Weather Data Extraction** — Strip weather screenshots into CSV using Google Gemini 3 Pro
2. **Date Formatting** — Add year and month to NOAA day-only dates ([`data_processing/2_full_date.py`](data_processing/2_full_date.py))
3. **UTC Conversion** — Convert local weather timestamps to UTC by adding 5 hours ([`data_processing/3_convert_to_utc.py`](data_processing/3_convert_to_utc.py))
4. **Weather Preprocessing** — Split and encode wind direction/magnitude, cloud cover types/heights, and other categorical features ([`data_processing/4_weather_preprocessing.py`](data_processing/4_weather_preprocessing.py))
5. **Timestamp Matching** — Match each 10-second pollution reading to the nearest weather observation within ±30 minutes ([`data_processing/5_match_weather_id.py`](data_processing/5_match_weather_id.py))
6. **File Merging** — Combine all per-session merged files into one master CSV ([`data_processing/6_squish_merged.py`](data_processing/6_squish_merged.py))
7. **History-Aware Processing** — Add columns referencing previous weather conditions for temporal context ([`data_processing/7_history_aware_processing.py`](data_processing/7_history_aware_processing.py))
8. **Temporal Averaging** — Either rolling averages ([`data_processing/8_rolling_average.py`](data_processing/8_rolling_average.py)) or chunked averages ([`data_processing/8_chunked_averages.py`](data_processing/8_chunked_averages.py)) to reduce noise and data frequency

## Models

### XGBoost Regression

Two model variants are trained for each target variable (CO2 and 0.3µm particles):

- **No History Models** ([`models/no_history_xgb/`](models/no_history_xgb/)) — Use only current weather conditions
  - [`xgboost_co2.py`](models/no_history_xgb/xgboost_co2.py) — CO2 prediction
  - [`xgboost_0.3um.py`](models/no_history_xgb/xgboost_0.3um.py) — 0.3µm particle prediction

- **History-Aware Models** ([`models/history_xgb/`](models/history_xgb/)) — Include previous weather conditions as features
  - [`co2_history.py`](models/history_xgb/co2_history.py) — CO2 prediction with historical weather
  - [`03um_history.py`](models/history_xgb/03um_history.py) — 0.3µm particle prediction with historical weather

### Linear Regression Analysis

A comprehensive linear regression analysis ([`analysis/linear_regression_analysis.py`](analysis/linear_regression_analysis.py)) provides:
- Multivariate linear regression for both targets
- Univariate R² for each weather feature
- Bootstrap R² uncertainty estimation (1000 iterations, 95% CI)
- Correlation matrix heatmaps and scatter plots

## Weather Features

The models use a subset of the following weather features:

| Feature | Description |
|---------|-------------|
| Prevailing Wind Magnitude (MPH) | Prevailing wind speed |
| Gust Wind Magnitude (MPH) | Gust wind speed |
| Vis (MI) | Visibility in miles |
| Cloud Height 1–4 (100s of ft) | Cloud ceiling heights |
| Air Temp (F) | Air temperature |
| Dewpoint (F) | Dewpoint temperature |
| Rel Hum | Relative humidity |
| Wind Chill (F) | Wind chill factor |
| Heat Index (F) | Heat index |
| Sea Level Pressure (MB) | Sea level pressure |
| Precip 1hr / 3hr / 6hr | Precipitation accumulation |
| Categorical codes | Encoded cloud types, wind directions, weather conditions |

Feature selection experiments are documented in [1_chunk_experiment/features.txt](1_chunk_experiment/features.txt).

## Results

### Rolling Average Models (With History)

![Rolling CO2 Histogram](rolling_results/co2_roll_hist.png)
*CO2 prediction distribution — rolling average, w/ history*

![Rolling 0.3um Histogram](rolling_results/0.3_roll_hist.png)
*0.3µm particle prediction distribution — rolling average, w/ history*

![Rolling CO2 Linear Regression](rolling_results/co2_linear_reg_rolling.png)
*Linear regression analysis for CO2 — rolling average model*

![Rolling 0.3um Linear Regression](rolling_results/0.3um_linear_reg_rolling.png)
*Linear regression analysis for 0.3µm particles — rolling average model*

![Rolling Correlation Heatmap](rolling_results/linear_regression_results_rolling.png)
*Correlation matrix heatmap — rolling average model*

### 1 Chunk per Hour Experiment

![1 Chunk CO2 Training](1_chunk_experiment/all_training_process_co2.png)
*CO2 prediction training progress — 1 chunk per hour*

![1 Chunk 0.3um Training](1_chunk_experiment/all_training_process_0.3um.png)
*0.3µm particle prediction training progress — 1 chunk per hour*

![1 Chunk CO2 Feature Importance](1_chunk_experiment/all_feature_importance_co2.png)
*Feature importance for CO2 — 1 chunk per hour*

![1 Chunk 0.3um Feature Importance](1_chunk_experiment/all_feature_importance_0.3um.png)
*Feature importance for 0.3µm particles — 1 chunk per hour*

![1 Chunk Correlation Heatmap](1_chunk_experiment/linear_regression_results.png)
*Correlation matrix heatmap — 1 chunk per hour*

![0.3um vs Features](1_chunk_experiment/particle_vs_features_linear_fit.png)
*0.3µm particle count vs individual weather features with linear fits*

![CO2 vs Features](1_chunk_experiment/co2_vs_features_linear_fit.png)
*CO2 vs individual weather features with linear fits*

### 6 Chunks per Hour Experiment

![6 Chunks CO2 Training](6_chunks/all_training_process_co2.png)
*CO2 prediction training progress — 6 chunks per hour*

![6 Chunks 0.3um Training](6_chunks/all_training_process_0.3um.png)
*0.3µm particle prediction training progress — 6 chunks per hour*

![6 Chunks CO2 Feature Importance](6_chunks/all_feature_importance_co2.png)
*Feature importance for CO2 — 6 chunks per hour*

![6 Chunks 0.3um Feature Importance](6_chunks/all_feature_importance_0.3um.png)
*Feature importance for 0.3µm particles — 6 chunks per hour*

![6 Chunks Correlation Heatmap](6_chunks/linear_regression_results.png)
*Correlation matrix heatmap — 6 chunks per hour*

![0.3um vs Features](6_chunks/particle_vs_features_linear_fit.png)
*0.3µm particle count vs individual weather features with linear fits*

![CO2 vs Features](6_chunks/co2_vs_features_linear_fit.png)
*CO2 vs individual weather features with linear fits*

## Project Structure

```
hazel_weather/
├── data_processing/          # Pipeline scripts (steps 2–8)
│   ├── 2_full_date.py
│   ├── 3_convert_to_utc.py
│   ├── 4_weather_preprocessing.py
│   ├── 5_match_weather_id.py
│   ├── 6_squish_merged.py
│   ├── 7_history_aware_processing.py
│   ├── 8_chunked_averages.py
│   ├── 8_rolling_average.py
│   └── pipeline.txt
├── models/
│   ├── no_history_xgb/       # XGBoost without historical weather
│   └── history_xgb/          # XGBoost with historical weather
├── analysis/
│   ├── linear_regression_analysis.py
│   └── calculate_stats.py
├── data/                     # Raw sensor data (per session)
├── 5_merged_data/            # Merged pollution + weather index files
├── 6_chunks/                 # Chunked output data
├── 1_chunk_experiment/       # Results for 1 chunk/hour experiment
├── rolling_results/          # Results for rolling average models
├── no_rolling_results/       # Results without rolling averages
└── to_do.txt                 # Planned improvements
```

## Requirements

- Python 3.10+
- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib
- seaborn
- scipy

## Usage

### Run the data processing pipeline:

```bash
# Step 2: Add full dates
python data_processing/2_full_date.py input.csv output.csv

# Step 3: Convert to UTC
python data_processing/3_convert_to_utc.py input.csv output.csv

# Steps 4–8: Edit import/export paths in each script and run directly
python data_processing/4_weather_preprocessing.py
python data_processing/5_match_weather_id.py
python data_processing/6_squish_merged.py
python data_processing/7_history_aware_processing.py
python data_processing/8_chunked_averages.py
```

### Train models:

```bash
# XGBoost with history
python models/history_xgb/co2_history.py
python models/history_xgb/03um_history.py

# XGBoost without history
python models/no_history_xgb/xgboost_co2.py
python models/no_history_xgb/xgboost_0.3um.py
```

### Run analysis:

```bash
python analysis/linear_regression_analysis.py
```