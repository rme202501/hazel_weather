"""
Linear Regression Analysis Script
Compares particle count (0.3um) and CO2 levels with weather variables.
Merges data CSV (with weather index) with weather CSV before analysis.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats

# ============================================================
# BOOTSTRAP CONFIGURATION
# ============================================================
N_BOOTSTRAP = 1000  # Number of bootstrap iterations
CONFIDENCE_LEVEL = 0.95  # Confidence level for intervals
RANDOM_STATE = 42  # For reproducibility

def bootstrap_r2(X, y, n_iterations=N_BOOTSTRAP, confidence=CONFIDENCE_LEVEL, random_state=RANDOM_STATE):
    """
    Perform bootstrapping to estimate R² uncertainty.
    Returns: mean R², std, and confidence interval bounds.
    """
    np.random.seed(random_state)
    n_samples = len(X)
    r2_scores = []
    
    for _ in range(n_iterations):
        # Bootstrap sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        y_boot = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
        
        # Fit model and calculate R²
        model = LinearRegression()
        model.fit(X_boot, y_boot)
        y_pred = model.predict(X_boot)
        r2 = r2_score(y_boot, y_pred)
        r2_scores.append(r2)
    
    r2_scores = np.array(r2_scores)
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    
    # Confidence interval using percentile method
    alpha = 1 - confidence
    ci_lower = np.percentile(r2_scores, 100 * alpha / 2)
    ci_upper = np.percentile(r2_scores, 100 * (1 - alpha / 2))
    
    return mean_r2, std_r2, ci_lower, ci_upper, r2_scores

def bootstrap_simple_r2(x, y, n_iterations=N_BOOTSTRAP, confidence=CONFIDENCE_LEVEL, random_state=RANDOM_STATE):
    """
    Perform bootstrapping for simple linear regression (single feature).
    Returns: mean R², std, and confidence interval bounds.
    """
    np.random.seed(random_state)
    n_samples = len(x)
    r2_scores = []
    
    for _ in range(n_iterations):
        # Bootstrap sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]
        
        # Calculate R² using scipy linregress
        _, _, r_value, _, _ = stats.linregress(x_boot, y_boot)
        r2 = r_value ** 2
        r2_scores.append(r2)
    
    r2_scores = np.array(r2_scores)
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    
    # Confidence interval using percentile method
    alpha = 1 - confidence
    ci_lower = np.percentile(r2_scores, 100 * alpha / 2)
    ci_upper = np.percentile(r2_scores, 100 * (1 - alpha / 2))
    
    return mean_r2, std_r2, ci_lower, ci_upper, r2_scores

# ============================================================
# CONFIGURATION - Set these paths and column names
# ============================================================
data_file = '1_chunk_experiment/8_chunked_output.csv'  # CSV with weather index
weather_file = '4_preprocessed_bos_weather_utc.csv'  # Weather data CSV
weather_index_column = 'weather_idx'  # Column name in data_file that references weather_file
weather_id_column = 'id'  # Column name in weather_file that matches the weather index
merge_on_index = True  # If True, merge on index position instead of columns

# Load the data
print(f"Loading data from {data_file}...")
df = pd.read_csv(data_file)

# Load weather data
print(f"Loading weather data from {weather_file}...")
weather_df = pd.read_csv(weather_file)

# Merge data with weather data
print(f"\nMerging data with weather data...")
if merge_on_index:
    # Merge weather_index_column onto weather_df row index
    if weather_index_column in df.columns:
        print(f"Merging '{weather_index_column}' (from data) onto weather row index...")
        df = df.merge(weather_df, left_on=weather_index_column, right_index=True, how='left')
        print(f"Data shape after merge: {df.shape}")
    else:
        print(f"Warning: '{weather_index_column}' not found in data file. Using data as-is.")
else:
    # Alternative: If weather data should be matched by index position
    print("Using index-based merge...")
    df = df.reset_index(drop=True)
    weather_df = weather_df.reset_index(drop=True)
    df = pd.concat([df, weather_df], axis=1)

# Define the columns we're interested in
target_vars = ['um03_mean', 'co2_mean']
um_col = 'um03_mean'
co2_col = 'co2_mean'
feature_vars = ['Vis (MI)', 'Air Temp (F)', 'Dewpoint (F)', 'Rel Hum', 'Sea Level Pressure (MB)', 'Precip 1hr', 'Prevailing Wind Magnitude (MPH)', 'time', 'Cloud Height 1 (100s of ft)']

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
y_particle = df_analysis[um_col]
y_co2 = df_analysis[co2_col]

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

# Bootstrap R² uncertainty for 0.3um
print(f"\nBootstrapping R² ({N_BOOTSTRAP} iterations)...")
mean_r2_particle, std_r2_particle, ci_lower_particle, ci_upper_particle, r2_dist_particle = bootstrap_r2(X, y_particle)
print(f"\nBootstrap R² Results:")
print(f"  Mean R²:               {mean_r2_particle:.4f}")
print(f"  Std Dev:               {std_r2_particle:.4f}")
print(f"  95% CI:                [{ci_lower_particle:.4f}, {ci_upper_particle:.4f}]")

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

# Bootstrap R² uncertainty for CO2
print(f"\nBootstrapping R² ({N_BOOTSTRAP} iterations)...")
mean_r2_co2, std_r2_co2, ci_lower_co2, ci_upper_co2, r2_dist_co2 = bootstrap_r2(X, y_co2)
print(f"\nBootstrap R² Results:")
print(f"  Mean R²:               {mean_r2_co2:.4f}")
print(f"  Std Dev:               {std_r2_co2:.4f}")
print(f"  95% CI:                [{ci_lower_co2:.4f}, {ci_upper_co2:.4f}]")

# ============================================================
# CORRELATION ANALYSIS
# ============================================================
print("\n" + "="*60)
print("CORRELATION MATRIX")
print("="*60)
correlation_matrix = df_analysis.corr()
print("\nCorrelation with 0.3um:")
for feature in feature_vars:
    print(f"  {feature:30s}: {correlation_matrix.loc[um_col, feature]:.4f}")

print("\nCorrelation with CO2:")
for feature in feature_vars:
    print(f"  {feature:30s}: {correlation_matrix.loc[co2_col, feature]:.4f}")
# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n" + "="*60)
print("Generating visualizations...")
print("="*60)

graphs_per_row = 3

# Create figure for correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Correlation Matrix Heatmap (6 chunks per hour)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('linear_regression_results.png', dpi=150, bbox_inches='tight')
print("Saved visualization to 'linear_regression_results.png'")

# ============================================================
# INDIVIDUAL LINEAR FIT PLOTS: 0.3um vs Each Feature
# ============================================================
print("Generating individual linear fit plots for 0.3um...")

# Calculate grid dimensions based on number of features
num_features = len(feature_vars)
num_rows = math.ceil(num_features / graphs_per_row)
figsize_height = 5 * num_rows
figsize_width = 5 * graphs_per_row

fig2, axes2 = plt.subplots(num_rows, graphs_per_row, figsize=(figsize_width, figsize_height))
axes2 = axes2.flatten()

for i, feature in enumerate(feature_vars):
    ax = axes2[i]
    
    # Get data
    x_data = df_analysis[feature].values
    y_data = df_analysis[um_col].values
    
    # Fit simple linear regression for this feature
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

plt.suptitle('0.3um Particle Count vs Weather Variables (6 chunks per hour)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('particle_vs_features_linear_fit.png', dpi=150, bbox_inches='tight')
print("Saved visualization to 'particle_vs_features_linear_fit.png'")

# ============================================================
# INDIVIDUAL LINEAR FIT PLOTS: CO2 vs Each Feature
# ============================================================
print("Generating individual linear fit plots for CO2...")

# Calculate grid dimensions based on number of features
num_features = len(feature_vars)
num_rows = math.ceil(num_features / graphs_per_row)
figsize_height = 5 * num_rows
figsize_width = 5 * graphs_per_row

fig3, axes3 = plt.subplots(num_rows, graphs_per_row, figsize=(figsize_width, figsize_height))
axes3 = axes3.flatten()

for i, feature in enumerate(feature_vars):
    ax = axes3[i]
    
    # Get data
    x_data = df_analysis[feature].values
    y_data = df_analysis[co2_col].values
    
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

plt.suptitle('CO2 vs Weather Variables (6 chunks per hour)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('co2_vs_features_linear_fit.png', dpi=150, bbox_inches='tight')
print("Saved visualization to 'co2_vs_features_linear_fit.png'")
print("\n" + "="*60)
print("SIMPLE LINEAR REGRESSION R² VALUES (Individual Features)")
print("="*60)
print(f"\n{'Feature':<30} {'0.3um R²':<15} {'CO2 R²':<15}")
print("-" * 60)

for feature in feature_vars:
    x_data = df_analysis[feature].values
    
    # 0.3um
    y_particle_data = df_analysis[um_col].values
    _, _, r_particle, _, _ = stats.linregress(x_data, y_particle_data)
    r2_particle_single = r_particle ** 2
    
    # CO2
    y_co2_data = df_analysis[co2_col].values
    _, _, r_co2, _, _ = stats.linregress(x_data, y_co2_data)
    r2_co2_single = r_co2 ** 2
    
    print(f"{feature:<30} {r2_particle_single:<15.4f} {r2_co2_single:<15.4f}")

# ============================================================
# BOOTSTRAP R² UNCERTAINTY FOR INDIVIDUAL FEATURES
# ============================================================
print("\n" + "="*80)
print("BOOTSTRAP R² UNCERTAINTY FOR INDIVIDUAL FEATURES")
print("="*80)
print(f"\n{'Feature':<25} {'0.3um R² (±std)':<20} {'0.3um 95% CI':<25} {'CO2 R² (±std)':<20} {'CO2 95% CI':<25}")
print("-" * 115)

bootstrap_results = []
for feature in feature_vars:
    x_data = df_analysis[feature].values
    y_particle_data = df_analysis[um_col].values
    y_co2_data = df_analysis[co2_col].values
    
    # Bootstrap for 0.3um
    mean_r2_p, std_r2_p, ci_lo_p, ci_hi_p, _ = bootstrap_simple_r2(x_data, y_particle_data)
    
    # Bootstrap for CO2
    mean_r2_c, std_r2_c, ci_lo_c, ci_hi_c, _ = bootstrap_simple_r2(x_data, y_co2_data)
    
    print(f"{feature:<25} {mean_r2_p:.4f} (±{std_r2_p:.4f})     [{ci_lo_p:.4f}, {ci_hi_p:.4f}]          {mean_r2_c:.4f} (±{std_r2_c:.4f})     [{ci_lo_c:.4f}, {ci_hi_c:.4f}]")
    
    bootstrap_results.append({
        'Feature': feature,
        '0.3um_R2_mean': mean_r2_p,
        '0.3um_R2_std': std_r2_p,
        '0.3um_R2_CI_lower': ci_lo_p,
        '0.3um_R2_CI_upper': ci_hi_p,
        'CO2_R2_mean': mean_r2_c,
        'CO2_R2_std': std_r2_c,
        'CO2_R2_CI_lower': ci_lo_c,
        'CO2_R2_CI_upper': ci_hi_c
    })

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)

# Create comprehensive summary with all R² values and uncertainties
summary_rows = []

# Section 1: Multivariate Model Results
summary_rows.append({
    'Model/Feature': 'MULTIVARIATE MODEL',
    'Metric': 'Test R²',
    '0.3um Value': f"{r2_particle:.4f}",
    'CO2 Value': f"{r2_co2:.4f}"
})
summary_rows.append({
    'Model/Feature': 'MULTIVARIATE MODEL',
    'Metric': 'Bootstrap R² Mean',
    '0.3um Value': f"{mean_r2_particle:.4f}",
    'CO2 Value': f"{mean_r2_co2:.4f}"
})
summary_rows.append({
    'Model/Feature': 'MULTIVARIATE MODEL',
    'Metric': 'Bootstrap R² Std Dev',
    '0.3um Value': f"{std_r2_particle:.4f}",
    'CO2 Value': f"{std_r2_co2:.4f}"
})
summary_rows.append({
    'Model/Feature': 'MULTIVARIATE MODEL',
    'Metric': 'Bootstrap 95% CI Lower',
    '0.3um Value': f"{ci_lower_particle:.4f}",
    'CO2 Value': f"{ci_lower_co2:.4f}"
})
summary_rows.append({
    'Model/Feature': 'MULTIVARIATE MODEL',
    'Metric': 'Bootstrap 95% CI Upper',
    '0.3um Value': f"{ci_upper_particle:.4f}",
    'CO2 Value': f"{ci_upper_co2:.4f}"
})
summary_rows.append({
    'Model/Feature': 'MULTIVARIATE MODEL',
    'Metric': 'RMSE',
    '0.3um Value': f"{rmse_particle:.4f}",
    'CO2 Value': f"{rmse_co2:.4f}"
})
summary_rows.append({
    'Model/Feature': 'MULTIVARIATE MODEL',
    'Metric': 'MAE',
    '0.3um Value': f"{mae_particle:.4f}",
    'CO2 Value': f"{mae_co2:.4f}"
})
summary_rows.append({
    'Model/Feature': 'MULTIVARIATE MODEL',
    'Metric': 'Intercept',
    '0.3um Value': f"{model_particle.intercept_:.4f}",
    'CO2 Value': f"{model_co2.intercept_:.4f}"
})

# Add coefficients for multivariate model
for feature, coef_p, coef_c in zip(feature_vars, model_particle.coef_, model_co2.coef_):
    summary_rows.append({
        'Model/Feature': 'MULTIVARIATE MODEL',
        'Metric': f'Coefficient: {feature}',
        '0.3um Value': f"{coef_p:.6f}",
        'CO2 Value': f"{coef_c:.6f}"
    })

# Section 2: Individual Feature R² Results with Uncertainties
for result in bootstrap_results:
    feature = result['Feature']
    
    summary_rows.append({
        'Model/Feature': feature,
        'Metric': 'Univariate R²',
        '0.3um Value': f"{result['0.3um_R2_mean']:.4f}",
        'CO2 Value': f"{result['CO2_R2_mean']:.4f}"
    })
    summary_rows.append({
        'Model/Feature': feature,
        'Metric': 'Bootstrap R² Std Dev',
        '0.3um Value': f"{result['0.3um_R2_std']:.4f}",
        'CO2 Value': f"{result['CO2_R2_std']:.4f}"
    })
    summary_rows.append({
        'Model/Feature': feature,
        'Metric': 'Bootstrap 95% CI Lower',
        '0.3um Value': f"{result['0.3um_R2_CI_lower']:.4f}",
        'CO2 Value': f"{result['CO2_R2_CI_lower']:.4f}"
    })
    summary_rows.append({
        'Model/Feature': feature,
        'Metric': 'Bootstrap 95% CI Upper',
        '0.3um Value': f"{result['0.3um_R2_CI_upper']:.4f}",
        'CO2 Value': f"{result['CO2_R2_CI_upper']:.4f}"
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# Save comprehensive summary to CSV
summary_df.to_csv('regression_summary.csv', index=False)
print("\nSaved comprehensive summary with all R² values and uncertainties to 'regression_summary.csv'")

# Save bootstrap results for individual features to CSV
bootstrap_df = pd.DataFrame(bootstrap_results)
bootstrap_df.to_csv('bootstrap_r2_summary.csv', index=False)
print("Saved bootstrap results to 'bootstrap_r2_summary.csv'")

plt.show()

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
