"""
Linear Regression Analysis Script
Compares particle count (0.3um) and CO2 levels with weather variables.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
print("Loading data from merged_output.csv...")
df = pd.read_csv('merged_output.csv')

# Define the columns we're interested in
target_vars = ['0.3um', 'co2']
feature_vars = ['Vis (MI)', 'Air Temp (F)', 'Dewpoint (F)', 'Rel Hum', 'Sea Level Pressure (MB)']

# Clean the data
print("\nCleaning data...")

# Convert 'Rel Hum' from percentage string to numeric (remove '%' sign)
if df['Rel Hum'].dtype == 'object':
    df['Rel Hum'] = df['Rel Hum'].str.replace('%', '').astype(float)

# Select only the columns we need
columns_needed = target_vars + feature_vars
df_analysis = df[columns_needed].copy()

# Drop rows with missing values
df_analysis = df_analysis.dropna()
print(f"Data shape after removing missing values: {df_analysis.shape}")

# Display basic statistics
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)
print(df_analysis.describe())

# Prepare features (X) and targets (y)
X = df_analysis[feature_vars]
y_particle = df_analysis['0.3um']
y_co2 = df_analysis['co2']

# Split data into training and testing sets
X_train, X_test, y_particle_train, y_particle_test = train_test_split(
    X, y_particle, test_size=0.2, random_state=42
)
_, _, y_co2_train, y_co2_test = train_test_split(
    X, y_co2, test_size=0.2, random_state=42
)

# ============================================================
# LINEAR REGRESSION FOR 0.3um PARTICLE COUNT
# ============================================================
print("\n" + "="*60)
print("LINEAR REGRESSION RESULTS: 0.3um Particle Count")
print("="*60)

model_particle = LinearRegression()
model_particle.fit(X_train, y_particle_train)

# Predictions
y_particle_pred = model_particle.predict(X_test)

# Metrics
r2_particle = r2_score(y_particle_test, y_particle_pred)
rmse_particle = np.sqrt(mean_squared_error(y_particle_test, y_particle_pred))
mae_particle = mean_absolute_error(y_particle_test, y_particle_pred)

print(f"\nModel Performance:")
print(f"  R² Score:              {r2_particle:.4f}")
print(f"  RMSE:                  {rmse_particle:.4f}")
print(f"  MAE:                   {mae_particle:.4f}")

print(f"\nIntercept: {model_particle.intercept_:.4f}")
print("\nCoefficients:")
for feature, coef in zip(feature_vars, model_particle.coef_):
    print(f"  {feature:30s}: {coef:.6f}")

# ============================================================
# LINEAR REGRESSION FOR CO2
# ============================================================
print("\n" + "="*60)
print("LINEAR REGRESSION RESULTS: CO2")
print("="*60)

model_co2 = LinearRegression()
model_co2.fit(X_train, y_co2_train)

# Predictions
y_co2_pred = model_co2.predict(X_test)

# Metrics
r2_co2 = r2_score(y_co2_test, y_co2_pred)
rmse_co2 = np.sqrt(mean_squared_error(y_co2_test, y_co2_pred))
mae_co2 = mean_absolute_error(y_co2_test, y_co2_pred)

print(f"\nModel Performance:")
print(f"  R² Score:              {r2_co2:.4f}")
print(f"  RMSE:                  {rmse_co2:.4f}")
print(f"  MAE:                   {mae_co2:.4f}")

print(f"\nIntercept: {model_co2.intercept_:.4f}")
print("\nCoefficients:")
for feature, coef in zip(feature_vars, model_co2.coef_):
    print(f"  {feature:30s}: {coef:.6f}")

# ============================================================
# CORRELATION ANALYSIS
# ============================================================
print("\n" + "="*60)
print("CORRELATION MATRIX")
print("="*60)
correlation_matrix = df_analysis.corr()
print("\nCorrelation with 0.3um:")
for feature in feature_vars:
    print(f"  {feature:30s}: {correlation_matrix.loc['0.3um', feature]:.4f}")

print("\nCorrelation with CO2:")
for feature in feature_vars:
    print(f"  {feature:30s}: {correlation_matrix.loc['co2', feature]:.4f}")

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n" + "="*60)
print("Generating visualizations...")
print("="*60)

# Create figure with subplots for summary
fig, axes = plt.subplots(3, 2, figsize=(14, 15))

# 1. Correlation Heatmap
plt.subplot(3, 2, 1)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Correlation Matrix Heatmap', fontsize=12, fontweight='bold')

# 2. Actual vs Predicted for 0.3um
plt.subplot(3, 2, 2)
plt.scatter(y_particle_test, y_particle_pred, alpha=0.5, edgecolors='none')
plt.plot([y_particle_test.min(), y_particle_test.max()], 
         [y_particle_test.min(), y_particle_test.max()], 'r--', lw=2)
plt.xlabel('Actual 0.3um')
plt.ylabel('Predicted 0.3um')
plt.title(f'0.3um: Actual vs Predicted (R² = {r2_particle:.4f})', fontsize=12, fontweight='bold')

# 3. Actual vs Predicted for CO2
plt.subplot(3, 2, 3)
plt.scatter(y_co2_test, y_co2_pred, alpha=0.5, edgecolors='none', color='green')
plt.plot([y_co2_test.min(), y_co2_test.max()], 
         [y_co2_test.min(), y_co2_test.max()], 'r--', lw=2)
plt.xlabel('Actual CO2')
plt.ylabel('Predicted CO2')
plt.title(f'CO2: Actual vs Predicted (R² = {r2_co2:.4f})', fontsize=12, fontweight='bold')

# 4. Residuals for 0.3um
plt.subplot(3, 2, 4)
residuals_particle = y_particle_test - y_particle_pred
plt.scatter(y_particle_pred, residuals_particle, alpha=0.5, edgecolors='none')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted 0.3um')
plt.ylabel('Residuals')
plt.title('0.3um: Residual Plot', fontsize=12, fontweight='bold')

# 5. Residuals for CO2
plt.subplot(3, 2, 5)
residuals_co2 = y_co2_test - y_co2_pred
plt.scatter(y_co2_pred, residuals_co2, alpha=0.5, edgecolors='none', color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted CO2')
plt.ylabel('Residuals')
plt.title('CO2: Residual Plot', fontsize=12, fontweight='bold')

# 6. Feature Importance Comparison
plt.subplot(3, 2, 6)
x_pos = np.arange(len(feature_vars))
width = 0.35
bars1 = plt.bar(x_pos - width/2, np.abs(model_particle.coef_), width, label='0.3um', alpha=0.8)
bars2 = plt.bar(x_pos + width/2, np.abs(model_co2.coef_), width, label='CO2', alpha=0.8)
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.title('Feature Importance (Absolute Coefficients)', fontsize=12, fontweight='bold')
plt.xticks(x_pos, [f.replace(' ', '\n') for f in feature_vars], fontsize=8)
plt.legend()

plt.tight_layout()
plt.savefig('linear_regression_results.png', dpi=150, bbox_inches='tight')
print("Saved visualization to 'linear_regression_results.png'")

# ============================================================
# INDIVIDUAL LINEAR FIT PLOTS: 0.3um vs Each Feature
# ============================================================
print("Generating individual linear fit plots for 0.3um...")

fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
axes2 = axes2.flatten()

for i, feature in enumerate(feature_vars):
    ax = axes2[i]
    
    # Get data
    x_data = df_analysis[feature].values
    y_data = df_analysis['0.3um'].values
    
    # Fit simple linear regression for this feature
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    r_squared = r_value ** 2
    
    # Create fit line
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = slope * x_fit + intercept
    
    # Plot scatter and fit line
    ax.scatter(x_data, y_data, alpha=0.3, s=5, edgecolors='none', label='Data')
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Linear Fit')
    
    # Add equation and R² to plot
    equation = f'y = {slope:.4f}x + {intercept:.2f}'
    ax.text(0.05, 0.95, f'{equation}\nR² = {r_squared:.4f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('0.3um Particle Count', fontsize=11)
    ax.set_title(f'0.3um vs {feature}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

# Hide the 6th subplot (we only have 5 features)
axes2[5].axis('off')

plt.suptitle('0.3um Particle Count vs Weather Variables (Simple Linear Regression)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('particle_vs_features_linear_fit.png', dpi=150, bbox_inches='tight')
print("Saved visualization to 'particle_vs_features_linear_fit.png'")

# ============================================================
# INDIVIDUAL LINEAR FIT PLOTS: CO2 vs Each Feature
# ============================================================
print("Generating individual linear fit plots for CO2...")

fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
axes3 = axes3.flatten()

for i, feature in enumerate(feature_vars):
    ax = axes3[i]
    
    # Get data
    x_data = df_analysis[feature].values
    y_data = df_analysis['co2'].values
    
    # Fit simple linear regression for this feature
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    r_squared = r_value ** 2
    
    # Create fit line
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = slope * x_fit + intercept
    
    # Plot scatter and fit line
    ax.scatter(x_data, y_data, alpha=0.3, s=5, edgecolors='none', color='green', label='Data')
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Linear Fit')
    
    # Add equation and R² to plot
    equation = f'y = {slope:.4f}x + {intercept:.2f}'
    ax.text(0.05, 0.95, f'{equation}\nR² = {r_squared:.4f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('CO2 (ppm)', fontsize=11)
    ax.set_title(f'CO2 vs {feature}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

# Hide the 6th subplot (we only have 5 features)
axes3[5].axis('off')

plt.suptitle('CO2 vs Weather Variables (Simple Linear Regression)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('co2_vs_features_linear_fit.png', dpi=150, bbox_inches='tight')
print("Saved visualization to 'co2_vs_features_linear_fit.png'")

# ============================================================
# PRINT INDIVIDUAL R² VALUES
# ============================================================
print("\n" + "="*60)
print("SIMPLE LINEAR REGRESSION R² VALUES (Individual Features)")
print("="*60)
print(f"\n{'Feature':<30} {'0.3um R²':<15} {'CO2 R²':<15}")
print("-" * 60)

for feature in feature_vars:
    x_data = df_analysis[feature].values
    
    # 0.3um
    y_particle_data = df_analysis['0.3um'].values
    _, _, r_particle, _, _ = stats.linregress(x_data, y_particle_data)
    r2_particle_single = r_particle ** 2
    
    # CO2
    y_co2_data = df_analysis['co2'].values
    _, _, r_co2, _, _ = stats.linregress(x_data, y_co2_data)
    r2_co2_single = r_co2 ** 2
    
    print(f"{feature:<30} {r2_particle_single:<15.4f} {r2_co2_single:<15.4f}")

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)

summary_data = {
    'Metric': ['R² Score', 'RMSE', 'MAE', 'Intercept'] + feature_vars,
    '0.3um Model': [f"{r2_particle:.4f}", f"{rmse_particle:.4f}", f"{mae_particle:.4f}", 
                   f"{model_particle.intercept_:.4f}"] + [f"{c:.6f}" for c in model_particle.coef_],
    'CO2 Model': [f"{r2_co2:.4f}", f"{rmse_co2:.4f}", f"{mae_co2:.4f}",
                 f"{model_co2.intercept_:.4f}"] + [f"{c:.6f}" for c in model_co2.coef_]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save summary to CSV
summary_df.to_csv('regression_summary.csv', index=False)
print("\nSaved summary to 'regression_summary.csv'")

plt.show()

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
