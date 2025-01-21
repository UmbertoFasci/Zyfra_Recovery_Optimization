import mlflow
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import Data
train_df = pd.read_csv('../datasets/gold_recovery_train.csv')
test_df = pd.read_csv('../datasets/gold_recovery_test.csv')
full_df = pd.read_csv('../datasets/gold_recovery_full.csv')

# Recovery Calculcation
def calculate_recovery(row, concentration_col, feed_col, tails_col):
    C = row[concentration_col]
    F = row[feed_col]
    T = row[tails_col]
    
    # Avoid division by zero
    if F == 0 or (C - T) == 0:
        return 0
    
    recovery = C * (F - T) / (F * (C - T)) * 100
    
    # Handle edge cases
    if np.isnan(recovery) or np.isinf(recovery):
        return 0
    
    return recovery

# Calculate recovery for rougher output
train_df['calculated_recovery'] = train_df.apply(
    lambda row: calculate_recovery(
        row,
        'rougher.output.concentrate_au',  # C
        'rougher.input.feed_au',          # F
        'rougher.output.tail_au'          # T
    ),
    axis=1
)

# Bi-Directional Rolling Average Interpolation
def rolling_average_interpolate(series, window=100):
    # Create forward and backward rolling means
    forward_roll = series.rolling(window=window, min_periods=1).mean()
    backward_roll = series[::-1].rolling(window=window, min_periods=1).mean()[::-1]
    
    # Combine forward and backward rolls
    combined_roll = (forward_roll + backward_roll) / 2
    
    # Only fill the NaN values in the original series
    result = series.copy()
    result[series.isna()] = combined_roll[series.isna()]
    return result

filled_train_df = train_df.copy()
filled_test_df = test_df.copy()

# Separate numeric and non-numeric columns
numeric_columns = train_df.select_dtypes(include=[np.number]).columns
non_numeric_columns = train_df.select_dtypes(exclude=[np.number]).columns

test_numeric_columns = test_df.select_dtypes(include=[np.number]).columns
test_non_numeric_columns = test_df.select_dtypes(exclude=[np.number]).columns

# Function to apply multiple interpolation and choose best one.
def advanced_fill(series):
    # Try different window sizes for rolling average
    windows = [50, 100, 200]
    best_filled = None
    least_missing = float('inf')
    
    for window in windows:
        filled = rolling_average_interpolate(series, window=window)
        missing_count = filled.isnull().sum()
        if missing_count < least_missing:
            least_missing = missing_count
            best_filled = filled
    
    # Final fallback: forward fill and backward fill
    if best_filled.isnull().sum() > 0:
        best_filled = best_filled.ffill().bfill()
        
    return best_filled

print(f"Beginning missing value imputation procedure on training data...")

# Fill non-numeric columns
print("\nFilling non-numeric columns...")
for column in non_numeric_columns:
    filled_train_df[column] = filled_train_df[column].ffill().bfill()

# Fill numeric columns with advanced method
print("\nFilling numeric columns...")
for column in numeric_columns:
    original_missing = filled_train_df[column].isnull().sum()
    if original_missing > 0:
        filled_train_df[column] = advanced_fill(filled_train_df[column])
        remaining_missing = filled_train_df[column].isnull().sum()

# Final verification
final_missing = filled_train_df.isnull().sum().sum()
print("\nFinal verification:")
print(f"Total missing values before: {train_df.isnull().sum().sum()}")
print(f"Total missing values after: {final_missing}")

print(f"Beginning missing value imputation procedure on testing data...")

# Fill non-numeric columns
print("\nFilling non-numeric columns...")
for column in test_non_numeric_columns:
    filled_test_df[column] = filled_test_df[column].ffill().bfill()

# Fill numeric columns with advanced method
print("\nFilling numeric columns...")
for column in test_numeric_columns:
    original_missing = filled_test_df[column].isnull().sum()
    if original_missing > 0:
        filled_test_df[column] = advanced_fill(filled_test_df[column])
        remaining_missing = filled_test_df[column].isnull().sum()

# Final verification
final_missing = filled_test_df.isnull().sum().sum()
print("\nFinal verification:")
print(f"Total missing values before: {test_df.isnull().sum().sum()}")
print(f"Total missing values after: {final_missing}")

# The filled dataframe is now stored in 'filled_test_df'
print("\nFilled dataframe is stored in 'filled_test_df' variable")