"""
Machine Learning dataset builder for trading predictions.

This module provides functions to create ML-ready datasets from processed features,
with proper time series handling to avoid look-ahead bias.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def build_ml_dataset(
    features_df: pd.DataFrame,
    horizon_hours: int = 4,
    threshold: float = 0.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a binary classification dataset for trading predictions.
    
    Args:
        features_df: DataFrame with features and price data (timestamp index)
        horizon_hours: Number of hours ahead to predict (default: 4)
        threshold: Minimum log return threshold for positive label (default: 0.0)
    
    Returns:
        Tuple of (X: features DataFrame, y: binary labels Series)
        
    Process:
        - Compute future_log_ret = log(close_t+h / close_t) using horizon_hours
        - Label y = 1 if future_log_ret > threshold else 0
        - Shift labels so features at time t correspond to future return t→t+h
        - Drop the last horizon_hours rows that have no future data
        - Drop any remaining NaN rows
        - X contains only feature columns (excludes raw price columns)
        
    Note:
        Index is kept aligned for backtesting timestamp matching.
    """
    # Work with a copy to avoid modifying original data
    df = features_df.copy()
    
    # Validate required columns
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column for label creation")
    
    print(f"🎯 Building ML dataset with {horizon_hours}h horizon, threshold={threshold}")
    print(f"   Input shape: {df.shape}")
    
    # Calculate future log returns
    future_close = df['close'].shift(-horizon_hours)
    current_close = df['close']
    future_log_ret = np.log(future_close / current_close)
    
    # Create binary labels
    labels = (future_log_ret > threshold).astype(int)
    
    # Remove the last horizon_hours rows (no future data available)
    df = df.iloc[:-horizon_hours].copy()
    labels = labels.iloc[:-horizon_hours].copy()
    
    # Define feature columns (exclude raw price/volume data that could cause leakage)
    exclude_patterns = [
        'open', 'high', 'low', 'close', 'volume',  # Raw OHLCV data
        'future_', 'target_', 'label_'  # Any obvious target columns
    ]
    
    # Get feature columns
    feature_columns = []
    for col in df.columns:
        is_feature = True
        for pattern in exclude_patterns:
            if pattern in col.lower():
                is_feature = False
                break
        
        if is_feature and df[col].dtype in ['int64', 'float64']:
            feature_columns.append(col)
    
    # Extract features
    X = df[feature_columns].copy()
    y = labels.copy()
    
    print(f"   Selected {len(feature_columns)} feature columns")
    print(f"   Excluded columns: {[col for col in df.columns if col not in feature_columns]}")
    
    # Ensure indices are aligned
    assert len(X) == len(y), f"Feature and label lengths don't match: {len(X)} vs {len(y)}"
    assert X.index.equals(y.index), "Feature and label indices don't match"
    
    # Drop any remaining NaN rows
    initial_rows = len(X)
    
    # Create mask for rows without NaN values in features or labels
    feature_mask = ~X.isnull().any(axis=1)
    label_mask = ~y.isnull()
    valid_mask = feature_mask & label_mask
    
    X = X[valid_mask]
    y = y[valid_mask]
    
    final_rows = len(X)
    dropped_rows = initial_rows - final_rows
    
    if dropped_rows > 0:
        print(f"   Dropped {dropped_rows:,} rows with NaN values")
    
    # Label distribution analysis
    label_counts = y.value_counts().sort_index()
    pos_rate = y.mean()
    
    print(f"   Final shape: X={X.shape}, y={y.shape}")
    print(f"   Label distribution: {label_counts.to_dict()}")
    print(f"   Positive rate: {pos_rate:.1%}")
    print(f"   Feature columns: {feature_columns[:10]}{'...' if len(feature_columns) > 10 else ''}")
    
    # Validate final dataset
    if len(X) == 0:
        raise ValueError("No valid samples remaining after preprocessing")
    
    if y.nunique() < 2:
        print(f"⚠️  Warning: Labels have only {y.nunique()} unique value(s)")
    
    # Final index alignment check
    assert X.index.equals(y.index), "Final feature and label indices don't match"
    
    return X, y


def create_time_based_splits(X: pd.DataFrame, y: pd.Series,
                           train_pct: float = 0.7,
                           valid_pct: float = 0.15,
                           test_pct: float = 0.15) -> dict:
    """
    Create time-based train/validation/test splits.
    
    Args:
        X: Features DataFrame with timestamp index
        y: Labels Series with timestamp index
        train_pct: Training set percentage (default: 0.7)
        valid_pct: Validation set percentage (default: 0.15)
        test_pct: Test set percentage (default: 0.15)
    
    Returns:
        Dictionary with split data and metadata
    """
    assert abs(train_pct + valid_pct + test_pct - 1.0) < 0.001, "Percentages must sum to 1.0"
    
    n_samples = len(X)
    train_end = int(n_samples * train_pct)
    valid_end = int(n_samples * (train_pct + valid_pct))
    
    # Create splits maintaining temporal order
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_valid = X.iloc[train_end:valid_end]
    y_valid = y.iloc[train_end:valid_end]
    
    X_test = X.iloc[valid_end:]
    y_test = y.iloc[valid_end:]
    
    # Create metadata
    splits_info = {
        'n_total': n_samples,
        'train_size': len(X_train),
        'valid_size': len(X_valid),
        'test_size': len(X_test),
        'train_period': (X_train.index.min(), X_train.index.max()),
        'valid_period': (X_valid.index.min(), X_valid.index.max()),
        'test_period': (X_test.index.min(), X_test.index.max()),
    }
    
    print(f"📊 Created time-based splits:")
    print(f"   Train: {len(X_train):,} samples ({len(X_train)/n_samples:.1%}) - {splits_info['train_period'][0]} to {splits_info['train_period'][1]}")
    print(f"   Valid: {len(X_valid):,} samples ({len(X_valid)/n_samples:.1%}) - {splits_info['valid_period'][0]} to {splits_info['valid_period'][1]}")
    print(f"   Test:  {len(X_test):,} samples ({len(X_test)/n_samples:.1%}) - {splits_info['test_period'][0]} to {splits_info['test_period'][1]}")
    
    # Check label distribution in each split
    for split_name, y_split in [("Train", y_train), ("Valid", y_valid), ("Test", y_test)]:
        pos_rate = y_split.mean()
        print(f"   {split_name} positive rate: {pos_rate:.1%}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_valid': X_valid, 'y_valid': y_valid,
        'X_test': X_test, 'y_test': y_test,
        'splits_info': splits_info
    }


def get_feature_importance_by_correlation(X: pd.DataFrame, y: pd.Series,
                                        top_n: int = 20) -> pd.DataFrame:
    """
    Calculate feature importance using correlation with target.
    
    Args:
        X: Features DataFrame
        y: Target Series
        top_n: Number of top features to return
    
    Returns:
        DataFrame with feature importance metrics
    """
    importance_data = []
    
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            # Calculate correlation with target
            corr = X[col].corr(y)
            abs_corr = abs(corr)
            
            # Basic statistics
            mean_val = X[col].mean()
            std_val = X[col].std()
            null_pct = X[col].isnull().mean()
            
            importance_data.append({
                'feature': col,
                'correlation': corr,
                'abs_correlation': abs_corr,
                'mean': mean_val,
                'std': std_val,
                'null_pct': null_pct
            })
    
    # Create DataFrame and sort by absolute correlation
    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values('abs_correlation', ascending=False)
    
    print(f"📈 Top {min(top_n, len(importance_df))} features by correlation:")
    for i, (_, row) in enumerate(importance_df.head(top_n).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<25} {row['correlation']:>8.4f}")
    
    return importance_df.head(top_n)


def validate_dataset_for_training(X: pd.DataFrame, y: pd.Series) -> bool:
    """
    Validate dataset for ML training.
    
    Args:
        X: Features DataFrame
        y: Labels Series
    
    Returns:
        True if dataset is valid for training
    """
    print("🔍 Validating dataset for ML training...")
    
    issues = []
    
    # Check basic requirements
    if len(X) == 0:
        issues.append("Empty dataset")
    
    if len(X) != len(y):
        issues.append(f"Feature/label length mismatch: {len(X)} vs {len(y)}")
    
    if not X.index.equals(y.index):
        issues.append("Feature/label index mismatch")
    
    # Check for NaN values
    if X.isnull().any().any():
        nan_features = X.columns[X.isnull().any()].tolist()
        issues.append(f"NaN values in features: {nan_features}")
    
    if y.isnull().any():
        issues.append(f"NaN values in labels: {y.isnull().sum()} out of {len(y)}")
    
    # Check label distribution
    if y.nunique() < 2:
        issues.append(f"Insufficient label diversity: {y.nunique()} unique values")
    
    # Check for constant features
    constant_features = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        issues.append(f"Constant features detected: {constant_features}")
    
    # Check minimum sample size
    min_samples = 1000
    if len(X) < min_samples:
        issues.append(f"Dataset too small: {len(X)} < {min_samples} samples")
    
    # Report results
    if issues:
        print("❌ Dataset validation failed:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Dataset validation passed")
        print(f"   Samples: {len(X):,}")
        print(f"   Features: {len(X.columns)}")
        print(f"   Label balance: {y.value_counts().to_dict()}")
        return True


if __name__ == "__main__":
    # Example usage and testing
    print("Testing ML dataset builder...")
    
    # Create sample features DataFrame
    np.random.seed(42)
    n_samples = 2000
    
    # Generate timestamp index
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='H', tz='UTC')
    
    # Generate realistic price data
    price = 50000
    prices = []
    for i in range(n_samples):
        change = np.random.normal(0, 0.02)
        price *= (1 + change)
        prices.append(price)
    
    # Create sample features DataFrame
    sample_df = pd.DataFrame({
        'close': prices,
        'open': [p * np.random.uniform(0.999, 1.001) for p in prices],
        'high': [p * np.random.uniform(1.001, 1.02) for p in prices],
        'low': [p * np.random.uniform(0.98, 0.999) for p in prices],
        'volume': np.random.uniform(100, 1000, n_samples),
        'log_ret_1h': np.random.normal(0, 0.02, n_samples),
        'ma_20': np.random.uniform(45000, 55000, n_samples),
        'ma_50': np.random.uniform(45000, 55000, n_samples),
        'rsi_14': np.random.uniform(20, 80, n_samples),
        'vol_24h': np.random.uniform(0.1, 0.5, n_samples),
        'funding_rate_zscore_30d': np.random.normal(0, 1, n_samples),
    }, index=dates)
    
    try:
        # Test ML dataset building
        print(f"\n📊 Testing build_ml_dataset...")
        X, y = build_ml_dataset(
            sample_df,
            horizon_hours=4,
            threshold=0.01
        )
        
        # Test validation
        is_valid = validate_dataset_for_training(X, y)
        
        # Test splits
        splits = create_time_based_splits(X, y)
        
        # Test feature importance
        importance = get_feature_importance_by_correlation(X, y, top_n=5)
        
        print(f"\n✅ ML dataset builder test successful!")
        print(f"   Final dataset: {X.shape}")
        print(f"   Validation passed: {is_valid}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()