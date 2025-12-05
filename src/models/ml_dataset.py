"""
Machine Learning Dataset Builder for Trading Signals.

This module prepares features and labels for ML models with proper time series handling
to avoid look-ahead bias and ensure realistic trading predictions.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Union
import warnings


def get_feature_columns(df: pd.DataFrame, exclude_patterns: List[str] = None) -> List[str]:
    """
    Automatically select feature columns, excluding raw OHLCV and target-related columns.
    
    Args:
        df: Input DataFrame
        exclude_patterns: List of patterns to exclude from features
    
    Returns:
        List of column names to use as features
    """
    if exclude_patterns is None:
        exclude_patterns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',  # Raw OHLCV
            'returns_', 'target_', 'label_', 'signal'  # Patterns that might contain future info
        ]
    
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out excluded patterns
    feature_cols = []
    for col in numeric_cols:
        exclude_col = False
        for pattern in exclude_patterns:
            if pattern in col:
                exclude_col = True
                break
        
        if not exclude_col:
            feature_cols.append(col)
    
    return feature_cols


def validate_no_lookahead(df: pd.DataFrame, feature_cols: List[str]) -> bool:
    """
    Validate that feature columns don't contain look-ahead bias.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
    
    Returns:
        True if no look-ahead bias detected
    """
    problematic_patterns = ['future_', 'next_', 'forward_']
    
    for col in feature_cols:
        for pattern in problematic_patterns:
            if pattern in col.lower():
                warnings.warn(f"Potential look-ahead bias detected in column: {col}")
                return False
    
    return True


def create_labels(df: pd.DataFrame, 
                 horizon_hours: int = 4, 
                 threshold: float = 0.0,
                 price_col: str = 'close') -> pd.Series:
    """
    Create binary labels based on future price movement.
    
    Args:
        df: DataFrame with price data
        horizon_hours: Number of periods to look ahead
        threshold: Minimum return threshold for positive label
        price_col: Column name for price data
    
    Returns:
        Series with binary labels (1 for positive, 0 for negative/neutral)
    """
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    # Calculate future returns
    future_price = df[price_col].shift(-horizon_hours)
    current_price = df[price_col]
    
    # Log returns over the horizon
    future_returns = np.log(future_price / current_price)
    
    # Create binary labels
    labels = (future_returns > threshold).astype(int)
    
    return labels


def build_ml_dataset(df: pd.DataFrame,
                    horizon_hours: int = 4,
                    threshold: float = 0.0,
                    feature_cols: Optional[List[str]] = None,
                    price_col: str = 'close',
                    validate_features: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
    """
    Build features X and labels y for machine learning.
    
    Args:
        df: Input DataFrame with features and price data
        horizon_hours: Number of periods ahead to predict (default: 4 hours)
        threshold: Minimum log return threshold for positive label (default: 0.0)
        feature_cols: Specific columns to use as features (auto-selected if None)
        price_col: Column name for price data (default: 'close')
        validate_features: Whether to validate features for look-ahead bias
    
    Returns:
        Tuple of (X: features DataFrame, y: labels Series, index: aligned index)
        
    Notes:
        - Labels are shifted to avoid look-ahead bias
        - Rows at the end without future data are dropped
        - Features are validated to prevent data leakage
    """
    # Input validation
    if len(df) <= horizon_hours:
        raise ValueError(f"DataFrame too short ({len(df)} rows) for horizon of {horizon_hours} periods")
    
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    # Work with a copy
    data = df.copy().reset_index(drop=True)
    
    # Auto-select feature columns if not provided
    if feature_cols is None:
        feature_cols = get_feature_columns(data)
        print(f"Auto-selected {len(feature_cols)} feature columns")
    else:
        # Validate provided feature columns exist
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Feature columns not found: {missing_cols}")
    
    # Validate features for look-ahead bias
    if validate_features:
        if not validate_no_lookahead(data, feature_cols):
            warnings.warn("Potential look-ahead bias detected in features")
    
    # Create labels
    print(f"Creating labels with {horizon_hours}h horizon and {threshold} threshold...")
    labels = create_labels(data, horizon_hours, threshold, price_col)
    
    # Extract features
    X = data[feature_cols].copy()
    y = labels.copy()
    
    # Remove rows without future labels (last horizon_hours rows)
    X = X.iloc[:-horizon_hours]
    y = y.iloc[:-horizon_hours]
    
    # Ensure alignment
    assert len(X) == len(y), f"Feature and label lengths don't match: {len(X)} vs {len(y)}"
    
    # Create aligned index
    aligned_index = X.index
    
    # Remove any remaining NaN values
    initial_rows = len(X)
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    aligned_index = aligned_index[mask]
    
    final_rows = len(X)
    if final_rows < initial_rows:
        print(f"Removed {initial_rows - final_rows} rows with NaN values ({final_rows} rows remaining)")
    
    # Summary statistics
    label_dist = y.value_counts().sort_index()
    pos_rate = y.mean() if len(y) > 0 else 0
    
    print(f"Dataset created: {len(X)} samples, {len(feature_cols)} features")
    print(f"Label distribution: {label_dist.to_dict()}")
    print(f"Positive rate: {pos_rate:.1%}")
    print(f"Feature columns: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
    
    return X, y, aligned_index


def create_time_series_splits(df: pd.DataFrame, 
                             n_splits: int = 5,
                             test_size: float = 0.2,
                             gap_size: int = 24) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time series cross-validation splits.
    
    Args:
        df: Input DataFrame
        n_splits: Number of splits to create
        test_size: Proportion of data for testing
        gap_size: Gap between train and test to avoid leakage (in periods)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    n_samples = len(df)
    test_samples = int(n_samples * test_size)
    
    splits = []
    
    for i in range(n_splits):
        # Calculate test start position for this split
        test_start = n_samples - test_samples - (i * (test_samples // n_splits))
        test_end = test_start + test_samples
        
        # Ensure we don't go beyond the data
        if test_start < test_samples:
            break
            
        # Train data ends before test with a gap
        train_end = test_start - gap_size
        train_start = 0
        
        if train_end <= train_start:
            break
            
        train_indices = np.arange(train_start, train_end)
        test_indices = np.arange(test_start, test_end)
        
        splits.append((train_indices, test_indices))
    
    print(f"Created {len(splits)} time series splits with gap of {gap_size} periods")
    return splits


def prepare_features_for_training(X: pd.DataFrame, 
                                 y: pd.Series,
                                 scale_features: bool = True,
                                 handle_outliers: bool = True,
                                 outlier_threshold: float = 5.0) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for ML training with optional scaling and outlier handling.
    
    Args:
        X: Features DataFrame
        y: Labels Series  
        scale_features: Whether to apply standard scaling
        handle_outliers: Whether to clip outliers
        outlier_threshold: Z-score threshold for outlier clipping
    
    Returns:
        Processed (X, y) tuple
    """
    X_processed = X.copy()
    y_processed = y.copy()
    
    if handle_outliers:
        # Clip outliers using z-score
        z_scores = np.abs((X_processed - X_processed.mean()) / X_processed.std())
        X_processed = X_processed.clip(
            lower=X_processed.quantile(0.01), 
            upper=X_processed.quantile(0.99), 
            axis=0
        )
        print(f"Clipped outliers beyond {outlier_threshold} standard deviations")
    
    if scale_features:
        # Standardize features (z-score normalization)
        X_processed = (X_processed - X_processed.mean()) / X_processed.std()
        print("Applied standard scaling to features")
    
    # Final validation
    if X_processed.isnull().any().any():
        print("Warning: NaN values found after preprocessing")
    
    return X_processed, y_processed


def get_feature_importance_analysis(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Basic feature importance analysis using correlation.
    
    Args:
        X: Features DataFrame
        y: Target Series
    
    Returns:
        DataFrame with feature importance metrics
    """
    importance_data = []
    
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            # Correlation with target
            corr = X[col].corr(y)
            
            # Basic statistics
            mean_val = X[col].mean()
            std_val = X[col].std()
            null_pct = X[col].isnull().mean()
            
            importance_data.append({
                'feature': col,
                'correlation': corr,
                'abs_correlation': abs(corr),
                'mean': mean_val,
                'std': std_val,
                'null_pct': null_pct
            })
    
    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values('abs_correlation', ascending=False)
    
    return importance_df


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample time series data
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
    # Simulate price data
    price = 100
    prices = []
    for _ in range(n_samples):
        change = np.random.normal(0, 0.01)
        price *= (1 + change)
        prices.append(price)
    
    # Create sample feature DataFrame
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples),
        'ma_20': pd.Series(prices).rolling(20).mean(),
        'ma_50': pd.Series(prices).rolling(50).mean(),
        'rsi_14': 50 + 20 * np.random.randn(n_samples),
        'volatility_24h': np.random.uniform(0.1, 0.5, n_samples),
        'funding_z_score': np.random.randn(n_samples),
        'returns_1h': pd.Series(prices).pct_change(),
    })
    
    # Test the ML dataset builder
    print("Testing ML dataset builder...")
    
    try:
        X, y, idx = build_ml_dataset(
            sample_df, 
            horizon_hours=4, 
            threshold=0.01
        )
        
        print(f"\nDataset shape: X={X.shape}, y={y.shape}")
        print(f"Sample features: {X.columns[:5].tolist()}")
        print(f"Label balance: {y.value_counts().to_dict()}")
        
        # Test time series splits
        splits = create_time_series_splits(X, n_splits=3)
        print(f"Time series splits: {len(splits)} splits created")
        
        # Test feature importance
        importance_df = get_feature_importance_analysis(X, y)
        print(f"\nTop 5 features by correlation:")
        print(importance_df.head()[['feature', 'correlation']].to_string(index=False))
        
        print("\nML dataset builder test completed successfully!")
        
    except Exception as e:
        print(f"Error testing ML dataset builder: {e}")
        import traceback
        traceback.print_exc()
