import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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

# Feature Preparation
def prepare_features(train_df, test_df=None):
    # Select relevant numerical features from training data
    feature_columns = [col for col in train_df.columns if any(x in col for x in [ 
        'feed', 'particle_size', 'concentration', 'state', 'floatbank'])]
    
    # Remove target columns
    target_columns = ['rougher.output.recovery', 'final.output.recovery']
    feature_columns = [col for col in feature_columns if col not in target_columns]
    
    # Only keep features present in both datasets
    if test_df is not None:
        feature_columns = [col for col in feature_columns if col in test_df.columns]
        print(f"Number of aligned features: {len(feature_columns)}")
    
    X_train = train_df[feature_columns]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if test_df is not None:
        X_test = test_df[feature_columns]
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, feature_columns, scaler
    
    return X_train_scaled, feature_columns, scaler

# sMAPE Calculator
def calculate_smape(y_true, y_pred):
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle cases where both true and predicted values are 0
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Create a mask for valid entries (non-zero denominator)
    valid_mask = denominator != 0
    
    if not np.any(valid_mask):
        return 0.0  # Return 0 if all denominators are 0
    
    # Calculate sMAPE only for valid entries
    numerator = np.abs(y_true - y_pred)
    smape = np.mean(np.divide(numerator[valid_mask], denominator[valid_mask])) * 100
    
    return smape

# Final sMAPE
def calculate_final_smape(y_true_rougher, y_pred_rougher, y_true_final, y_pred_final):
    rougher_smape = calculate_smape(y_true_rougher, y_pred_rougher)
    final_smape = calculate_smape(y_true_final, y_pred_final)
    
    # Print sNAPE information
    print(f"Rougher sMAPE components:")
    print(f"  Range of true values: [{np.min(y_true_rougher):.2f}, {np.max(y_true_rougher):.2f}]")
    print(f"  Range of predicted values: [{np.min(y_pred_rougher):.2f}, {np.max(y_pred_rougher):.2f}]")
    print(f"  Calculated rougher sMAPE: {rougher_smape:.2f}")
    
    print(f"\nFinal sMAPE components:")
    print(f"  Range of true values: [{np.min(y_true_final):.2f}, {np.max(y_true_final):.2f}]")
    print(f"  Range of predicted values: [{np.min(y_pred_final):.2f}, {np.max(y_pred_final):.2f}]")
    print(f"  Calculated final sMAPE: {final_smape:.2f}")
    
    return 0.25 * rougher_smape + 0.75 * final_smape

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="", verbose=True):
    # Original metrics calculation
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, test_pred),
        'smape': calculate_smape(y_test, test_pred)
    }
    
    train_metrics = {
        'mae': mean_absolute_error(y_train, train_pred),
        'smape': calculate_smape(y_train, train_pred)
    }
    
    # Add MLflow logging
    if mlflow.active_run():
        log_model_metrics(metrics, "test")
        log_model_metrics(train_metrics, "train")
    
    if verbose:
        print(f"\n{model_name} Evaluation Results:")
        print("-" * 40)
        print(f"Training MAE: {train_metrics['mae']:.4f}")
        print(f"Test MAE: {metrics['mae']:.4f}")
        print(f"Training sMAPE: {train_metrics['smape']:.4f}")
        print(f"Test sMAPE: {metrics['smape']:.4f}")
    
    return metrics, train_metrics

def tune_random_forest_with_mlflow(X_train, X_test, y_train, y_test, target_type):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        base_rf,
        param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    with mlflow.start_run(run_name=f"rf_optimization_{target_type}"):
        # Log the search space
        mlflow.log_params({"search_space": str(param_grid)})
        
        # Perform grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model and evaluate
        best_rf = grid_search.best_estimator_
        metrics, train_metrics = evaluate_model(
            best_rf, X_train, X_test, y_train, y_test, 
            f"Tuned Random Forest - {target_type}"
        )
        
        # Log parameters
        mlflow.log_params(grid_search.best_params_)
        
        # Log metrics
        log_model_metrics(metrics, "test")
        log_model_metrics(train_metrics, "train")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': best_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_dict(
            feature_importance.to_dict(orient='records'), 
            f"feature_importance_{target_type}.json"
        )
        
        # Log the model
        mlflow.sklearn.log_model(
            best_rf, 
            f"model_{target_type}",
            registered_model_name=f"gold_recovery_{target_type}"
        )
        
        return best_rf, grid_search.best_params_
    
def train_models_with_tracking(X_train, X_test, y_train_rougher, y_test_rougher, 
                             y_train_final, y_test_final):
    """Main training function with MLflow experiment tracking"""
    
    # Train rougher recovery model
    print("\nTraining Rougher Recovery Model...")
    rougher_model, rougher_params = tune_random_forest_with_mlflow(
        X_train, X_test, y_train_rougher, y_test_rougher, "rougher"
    )
    
    # Train final recovery model
    print("\nTraining Final Recovery Model...")
    final_model, final_params = tune_random_forest_with_mlflow(
        X_train, X_test, y_train_final, y_test_final, "final"
    )
    
    # Calculate and log final combined sMAPE
    with mlflow.start_run(run_name="final_evaluation"):
        rougher_pred = rougher_model.predict(X_test)
        final_pred = final_model.predict(X_test)
        
        final_smape = calculate_final_smape(
            y_test_rougher, rougher_pred,
            y_test_final, final_pred
        )
        
        mlflow.log_metric("final_combined_smape", final_smape)
        
        print(f"\nFinal Combined sMAPE: {final_smape:.4f}")
    
    return rougher_model, final_model

def log_model_metrics(metrics, stage="training"):
    """MLflow logging"""
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"{stage}_{metric_name}", value)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("Zyfra RF Model Optimization")
    
    filled_train_df = train_df.copy()
    filled_test_df = test_df.copy()
    
    X_train_scaled, X_test_scaled, feature_columns, scaler = prepare_features(
        filled_train_df, filled_test_df)
    
    y_train_rougher = filled_train_df['rougher.output.recovery']
    y_train_final = filled_train_df['final.output.recovery']
    
    y_train_rougher = advanced_fill(y_train_rougher)
    y_train_final = advanced_fill(y_train_final)

    # Sample data before split
    sample_size = min(3000, len(X_train_scaled))
    indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
    
    X_train_scaled = X_train_scaled[indices]
    y_train_rougher = y_train_rougher.iloc[indices]
    y_train_final = y_train_final.iloc[indices]
    
    X_train, X_test, y_train_rougher, y_test_rougher = train_test_split(
        X_train_scaled, y_train_rougher, test_size=0.2, random_state=12345)
    _, _, y_train_final, y_test_final = train_test_split(
        X_train_scaled, y_train_final, test_size=0.2, random_state=12345)
    
    rougher_model, final_model = train_models_with_tracking(
        X_train, X_test,
        y_train_rougher, y_test_rougher,
        y_train_final, y_test_final)