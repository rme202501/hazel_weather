import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# 1. Load Data
data = pd.read_csv('nohistory_rolling_average.csv')

X = data[['Prevailing Wind Magnitude (MPH)', 'Gust Wind Magnitude (MPH)', 'Vis (MI)', 
          'Cloud Height 1 (100s of ft)', 'Cloud Height 2 (100s of ft)', 'Cloud Height 3 (100s of ft)', 'Cloud Height 4 (100s of ft)', 
          'Air Temp (F)', 'Dewpoint (F)', '6hr Max (F)', '6hr Min (F)', 'Rel Hum', 'Wind Chill (F)', 
          'Heat Index (F)', 'Sea Level Pressure (MB)', 'Precip 1hr', 'Precip 3hr', 'Precip 6hr',
          'cloud_code_1', 'cloud_code_2', 'cloud_code_3', 'cloud_code_4', 'prevailing_wind_dir_code', 'gust_wind_dir_code', 'weather_code']]
y = data['co2_rolling']

categorical_cols = ['cloud_code_1', 'cloud_code_2', 'cloud_code_3', 'cloud_code_4', 'prevailing_wind_dir_code', 'gust_wind_dir_code', 'weather_code']

train_split_ratio = 0.8
train_split_point = int(len(y) * train_split_ratio)

X_train = X.iloc[:train_split_point]
y_train = y.iloc[:train_split_point]
X_test = X.iloc[train_split_point:]
y_test = y.iloc[train_split_point:]

# 3. Initialize the Model
# use XGBRegressor for predicting continuous numbers (prices, temp, etc.)
model = xgb.XGBRegressor(
    n_estimators=500,      # Number of trees
    learning_rate=0.01,     # How much each tree contributes (step size)
    max_depth=25,           # Depth of each tree (complexity)
    objective='reg:squarederror', # Specify the learning task
    eval_metric='rmse',    # Metric to evaluate during training
    random_state=42
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

# 5. Plot Training Progress
print("\nGenerating training progress visualization...")
results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_axis, results['validation_0']['rmse'], label='Train', linewidth=2)
ax.plot(x_axis, results['validation_1']['rmse'], label='Test', linewidth=2)
ax.legend(fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_title('CO2 Prediction w/o Rolling Averages & No Attention to History', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Annotate final values
final_train_rmse = results['validation_0']['rmse'][-1]
final_test_rmse = results['validation_1']['rmse'][-1]
ax.text(0.02, 0.98, f'Final Train RMSE: {final_train_rmse:.2f}\nFinal Validation RMSE: {final_test_rmse:.2f}\nFinal Validation R²: {r2:.4f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()
print("Graph displayed successfully!")

# 8. Save the Model
print("\nSaving trained model...")
model_dir = 'models/xgboost/saved_models'
os.makedirs(model_dir, exist_ok=True)

# Generate filename with timestamp and metrics
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'xgboost_co2_{timestamp}_rmse{rmse:.2f}_r2{r2:.4f}.json'
model_path = os.path.join(model_dir, model_filename)

# Save model
model.save_model(model_path)
print(f"Model saved to: {model_path}")

# Save metrics
metrics_filename = f'xgboost_co2_{timestamp}_metrics.json'
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