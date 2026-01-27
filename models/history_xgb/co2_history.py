import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import json
from datetime import datetime
import random

# 1. Load Data
pollution_data = pd.read_csv('6_chunks/8_chunked_output.csv')
weather_data = pd.read_csv('4_preprocessed_bos_weather_utc.csv')
num_history_steps = 0

# Set weather_data index for efficient lookups
weather_data_indexed = weather_data.set_index(weather_data.index)  # Use positional index

# Define weather feature columns
weather_feature_cols = [
                        'Prevailing Wind Magnitude (MPH)', 
                        'Gust Wind Magnitude (MPH)', 
                        # 'Vis (MI)', 
                        # 'Cloud Height 1 (100s of ft)', 
                        # 'Cloud Height 2 (100s of ft)', 
                        # 'Cloud Height 3 (100s of ft)', 
                        # 'Cloud Height 4 (100s of ft)', 
                        # 'Air Temp (F)', 
                        # 'Dewpoint (F)', 
                        # '6hr Max (F)', 
                        # '6hr Min (F)', 
                        # 'Rel Hum', 
                        'Wind Chill (F)', 
                        # 'Heat Index (F)', 
                        'Sea Level Pressure (MB)', 
                        # 'Precip 1hr', 
                        # 'Precip 3hr', 
                        # 'Precip 6hr',
                        # 'cloud_code_1', 
                        # 'cloud_code_2', 
                        # 'cloud_code_3', 
                        # 'cloud_code_4', 
                        'prevailing_wind_dir_code', 
                        'gust_wind_dir_code', 
                        # 'weather_code'
                        ]
# Randomly ignore some weather features during training
# ignore_probability = 0.3  # 30% chance to ignore each feature
# ignored_cols = [col for col in weather_feature_cols if random.random() < ignore_probability]
# ignored_cols = ['cloud_code_1', 'cloud_code_2', 'cloud_code_3', 'cloud_code_4', 'prevailing_wind_dir_code', 'gust_wind_dir_code', 'weather_code']
ignored_cols = []
weather_feature_cols = [col for col in weather_feature_cols if col not in ignored_cols]

print("\n" + "="*60)
print("FEATURES CONFIGURATION")
print("="*60)
if ignored_cols:
    print(f"Ignored {len(ignored_cols)} features:")
    for col in ignored_cols:
        print(f"  - {col}")
else:
    print("No features were ignored")
print(f"\nUsing {len(weather_feature_cols)} features for training")
print("="*60 + "\n")


def get_historical_weather_features(pollution_df, weather_df, history_steps=num_history_steps):
    """
    Join pollution data with current and historical weather data.
    
    pollution_df: DataFrame with weather_idx, weather_idx_1, weather_idx_2, etc.
    weather_df: DataFrame with weather features (indexed by row number)
    history_steps: Number of historical steps to include (weather_idx_1, weather_idx_2, etc.)
    """
    X = pd.DataFrame(index=pollution_df.index)
    
    # Current weather (weather_idx) - filter out NaN values first
    valid_current_idx = pollution_df['weather_idx'].notna()
    if valid_current_idx.any():
        current_weather = weather_df.loc[pollution_df.loc[valid_current_idx, 'weather_idx'].astype(int).values][weather_feature_cols].reset_index(drop=True)
        current_weather.index = pollution_df.loc[valid_current_idx].index
        current_weather.columns = [f'{col}_current' for col in weather_feature_cols]
        X = X.join(current_weather, how='left')
    
    # Historical weather
    for step in range(1, history_steps):
        idx_col = f'weather_idx_{step}'
        if idx_col in pollution_df.columns:
            # Filter out rows with NaN indices
            valid_idx = pollution_df[idx_col].notna()
            if valid_idx.any():
                historical_weather = weather_df.loc[pollution_df.loc[valid_idx, idx_col].astype(int).values][weather_feature_cols].reset_index(drop=True)
                # Align indices back to original pollution_df
                temp_idx = pollution_df.loc[valid_idx].index
                historical_weather.index = temp_idx
                historical_weather.columns = [f'{col}_h{step}' for col in weather_feature_cols]
                # Join, filling NaN for rows that didn't have valid historical indices
                X = X.join(historical_weather, how='left')

    return X

# Build feature matrix with historical weather data
X = get_historical_weather_features(pollution_data, weather_data, history_steps=num_history_steps)

# Extract target from pollution data
y = pollution_data['co2_mean']

# Drop rows where target or key indices are NaN
valid_rows = (~y.isna()) & (~pollution_data['weather_idx'].isna())
X = X[valid_rows]
X.to_csv('model_inputs.csv')
y = y[valid_rows]

categorical_cols = ['cloud_code_1', 'cloud_code_2', 'cloud_code_3', 'cloud_code_4', 'prevailing_wind_dir_code', 'gust_wind_dir_code', 'weather_code']

# Create categorical column names with all suffixes
categorical_feature_names = []
categorical_feature_names.extend([f'{col}_current' for col in categorical_cols])
for step in range(1, num_history_steps + 1):
    categorical_feature_names.extend([f'{col}_h{step}' for col in categorical_cols])

# Convert categorical columns to category type (only for columns that exist)
for col in categorical_feature_names:
    if col in X.columns:
        X[col] = X[col].astype('category')

train_split_ratio = 0.8

# Split by weather_idx to keep all rows with same weather_idx together
unique_weather_indices = pollution_data.loc[X.index, 'weather_idx'].unique()
indices_list = list(unique_weather_indices)
random.seed(42)
random.shuffle(indices_list)

# Split the unique indices
train_split_point = int(len(indices_list) * train_split_ratio)
train_weather_indices = set(indices_list[:train_split_point])

# Create mask based on weather_idx for all rows in X/y
train_mask = pollution_data.loc[X.index, 'weather_idx'].isin(train_weather_indices).values

X_train = X[train_mask].copy()
y_train = y[train_mask].copy()
X_test = X[~train_mask].copy()
y_test = y[~train_mask].copy()
print(len(X_train))
print(len(X_test))

# Ensure categorical columns remain categorical after split
for col in categorical_feature_names:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

# 3. Initialize the Model
# use XGBRegressor for predicting continuous numbers (prices, temp, etc.)
model = xgb.XGBRegressor(
    n_estimators=75,     # Number of trees
    learning_rate=0.01,     # How much each tree contributes (step size)
    max_depth=10,           # Depth of each tree (complexity)
    objective='reg:squarederror', # Specify the learning task
    eval_metric='rmse',    # Metric to evaluate during training
    random_state=42,
    enable_categorical=True,  # Enable categorical feature support
)

# 4. Train the Model
print("="*60)
print("Starting XGBoost Model Training...")
print("="*60)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}")
print(f"Trees to build: {model.n_estimators}")
print("="*60)

start_time = time.time()

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=10  # Print progress every 10 trees
)

training_time = time.time() - start_time
print("="*60)
print(f"Training completed in {training_time:.2f} seconds")
print(f"Average time per tree: {training_time/model.n_estimators:.3f} seconds")
print("="*60)

# Baseline: always predict the mean of training set
mean_train = y_train.mean()
baseline_preds = [mean_train] * len(y_test)
baseline_mse = mean_squared_error(y_test, baseline_preds)
baseline_rmse = baseline_mse ** 0.5

# 6. Make Predictions
print("\nMaking predictions on test set...")
preds = model.predict(X_test)

# 7. Evaluate
print("\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
rmse = mse ** 0.5
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print("="*60)

print("\n" + "-"*60)
print("BASELINE COMPARISON (predicting mean of y_train)")
print("-"*60)
print(f"Baseline RMSE: {baseline_rmse:.2f}")
print(f"Model RMSE: {rmse:.2f}")
print(f"Improvement: {baseline_rmse - rmse:.2f} ({((baseline_rmse - rmse) / baseline_rmse * 100):.1f}%)")
print("="*60)

# 5. Plot Training Progress
print("\nGenerating training progress visualization...")
results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_axis, results['validation_0']['rmse'], label='Train', linewidth=2)
ax.plot(x_axis, results['validation_1']['rmse'], label='Validation', linewidth=2)
ax.legend(fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_xlabel('Tree', fontsize=12)
ax.set_title('6 Chunks per Hour CO2 Prediction w/ No Attention to History, Wind Based Features', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Annotate final values
final_train_rmse = results['validation_0']['rmse'][-1]
final_test_rmse = results['validation_1']['rmse'][-1]
ax.text(0.02, 0.98, f'Final Train RMSE: {final_train_rmse:.2f}\nFinal Validation RMSE: {final_test_rmse:.2f}\nFinal Validation R²: {r2:.4f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('training_process_co2.png', dpi=300, bbox_inches='tight')
print("Graph displayed successfully!")

plt.figure(figsize=(10, 8))
xgb.plot_importance(model, importance_type='gain', max_num_features=20) # 'gain' is often preferred over default 'weight'
plt.title('Feature Importance (Gain)')
plt.tight_layout()
plt.savefig('feature_importance_co2.png', dpi=300, bbox_inches='tight')

# 8. Save the Model
print("\nSaving trained model...")
model_dir = 'models/history_xgb/saved_models'
os.makedirs(model_dir, exist_ok=True)

# Generate filename with timestamp and metrics
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'xgboost_0.3um_{timestamp}_rmse{rmse:.2f}_r2{r2:.4f}.json'
model_path = os.path.join(model_dir, model_filename)

# Save model
model.save_model(model_path)
print(f"Model saved to: {model_path}")

# Save metrics
metrics_filename = f'xgboost_0.3um_{timestamp}_metrics.json'
metrics_path = os.path.join(model_dir, metrics_filename)

metrics_data = {
    'timestamp': timestamp,
    'model_file': model_filename,
    'metrics': {
        'RMSE': float(rmse),
        'R2': float(r2),
        'MSE': float(mse)
    },
    'training_info': {
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X_train.shape[1],
        'n_estimators': model.n_estimators,
        'training_time_seconds': training_time
    }
}

with open(metrics_path, 'w') as f:
    json.dump(metrics_data, f, indent=4)

print(f"Metrics saved to: {metrics_path}")
print("="*60)
print("\nTraining complete!")