"""
Feature engineering module for trading data.

This module provides functions to build technical indicators and features
from raw OHLCV data for machine learning and backtesting.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build comprehensive features from raw OHLCV data.
    
    Args:
        raw_df: DataFrame with timestamp index and columns: open, high, low, close, volume
                Optionally includes funding_rate column
    
    Returns:
        DataFrame with original data plus engineered features
    """
    # Work with a copy
    df = raw_df.copy()
    
    # Ensure timestamp is the index
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    
    print("🔧 Building features...")
    print(f"   Input shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # === Log Returns ===
    print("   Adding log returns...")
    df['log_ret_1h'] = np.log(df['close'] / df['close'].shift(1))
    df['log_ret_4h'] = np.log(df['close'] / df['close'].shift(4))
    df['log_ret_24h'] = np.log(df['close'] / df['close'].shift(24))
    
    # === Moving Averages ===
    print("   Adding moving averages...")
    for window in [10, 20, 50, 100]:
        df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
    
    # === Technical Indicators ===
    print("   Adding technical indicators...")
    
    # RSI(14)
    df['rsi_14'] = _calculate_rsi(df['close'], window=14)
    
    # ATR(14)
    df['atr_14'] = _calculate_atr(df['high'], df['low'], df['close'], window=14)
    
    # 24h rolling volatility
    df['vol_24h'] = df['log_ret_1h'].rolling(window=24).std()
    
    # === Time-based Features ===
    print("   Adding time-based features...")
    
    # Extract time components
    hour = df.index.hour
    dow = df.index.dayofweek  # Monday=0, Sunday=6
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    
    # === Funding Rate Features (if available) ===
    if 'funding_rate' in df.columns:
        print("   Adding funding rate features...")
        
        # 24h change in funding rate
        df['funding_rate_24h_change'] = df['funding_rate'].diff(24)
        
        # 30-day z-score of funding rate
        df['funding_rate_zscore_30d'] = _calculate_zscore(df['funding_rate'], window=30*24)  # 30 days * 24 hours
    
    # === Additional Features ===
    print("   Adding additional features...")
    
    # Price relative to moving averages
    for window in [20, 50]:
        df[f'price_vs_ma_{window}'] = df['close'] / df[f'ma_{window}'] - 1
    
    # High-Low spread
    df['hl_spread_pct'] = (df['high'] - df['low']) / df['close'] * 100
    
    # Close position within the bar
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['close_position'] = df['close_position'].fillna(0.5)  # Fill when high == low
    
    # Volume features
    df['volume_ma_24h'] = df['volume'].rolling(window=24).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_24h']
    
    # Momentum features
    df['momentum_3h'] = (df['close'] / df['close'].shift(3)) - 1
    df['momentum_12h'] = (df['close'] / df['close'].shift(12)) - 1
    
    # === Clean up ===
    print("   Cleaning up NaN values...")
    
    initial_rows = len(df)
    
    # Drop rows with NaN values (primarily from moving averages and indicators)
    df = df.dropna()
    
    final_rows = len(df)
    dropped_rows = initial_rows - final_rows
    
    print(f"   Dropped {dropped_rows:,} rows with NaN values")
    print(f"   Final shape: {df.shape}")
    
    # Feature summary
    feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'funding_rate']]
    print(f"   Added {len(feature_cols)} features: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
    
    return df


def build_and_save_features(raw_path: Path, out_path: Optional[Path] = None) -> Path:
    """
    Load raw data, build features, and save to processed directory.
    
    Args:
        raw_path: Path to raw parquet file
        out_path: Output path (defaults to data/processed/btcusdt_1h_features.parquet)
    
    Returns:
        Path to saved features file
    """
    raw_path = Path(raw_path)
    
    if out_path is None:
        # Default to processed directory
        processed_dir = raw_path.parent.parent / "processed"
        processed_dir.mkdir(exist_ok=True)
        out_path = processed_dir / "btcusdt_1h_features.parquet"
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Loading raw data from: {raw_path}")
    
    # Load raw data
    raw_df = pd.read_parquet(raw_path)
    print(f"   Loaded {len(raw_df):,} rows")
    
    if hasattr(raw_df.index, 'tz') and raw_df.index.tz is None:
        # Ensure timezone awareness if needed
        raw_df.index = raw_df.index.tz_localize('UTC')
    
    # Build features
    features_df = build_features(raw_df)
    
    # Save features
    print(f"💾 Saving features to: {out_path}")
    features_df.to_parquet(out_path, engine="pyarrow")
    
    print(f"✅ Feature engineering completed!")
    print(f"   Input: {len(raw_df):,} rows")
    print(f"   Output: {len(features_df):,} rows")
    print(f"   Features: {len(features_df.columns)} columns")
    
    return out_path


def _calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    # True Range components
    hl = high - low
    hc = np.abs(high - close.shift(1))
    lc = np.abs(low - close.shift(1))
    
    # True Range is the maximum of the three
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    
    # ATR is the rolling mean of True Range
    atr = tr.rolling(window=window).mean()
    
    return atr


def _calculate_zscore(series: pd.Series, window: int = 720) -> pd.Series:
    """Calculate rolling z-score."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    zscore = (series - rolling_mean) / rolling_std
    
    return zscore


def get_feature_groups() -> dict:
    """
    Return a dictionary of feature groups for easy analysis.
    
    Returns:
        Dictionary mapping feature group names to column patterns
    """
    return {
        'returns': ['log_ret_1h', 'log_ret_4h', 'log_ret_24h'],
        'moving_averages': ['ma_10', 'ma_20', 'ma_50', 'ma_100'],
        'price_ratios': ['price_vs_ma_20', 'price_vs_ma_50'],
        'technical': ['rsi_14', 'atr_14', 'vol_24h'],
        'price_action': ['hl_spread_pct', 'close_position'],
        'volume': ['volume_ma_24h', 'volume_ratio'],
        'momentum': ['momentum_3h', 'momentum_12h'],
        'time': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'],
        'funding': ['funding_rate_24h_change', 'funding_rate_zscore_30d']
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Testing feature engineering...")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=2000, freq='H', tz='UTC')
    np.random.seed(42)
    
    # Generate realistic price data
    price = 50000  # Starting BTC price
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # Random walk with some trend
        change = np.random.normal(0, 0.02)  # 2% hourly volatility
        price *= (1 + change)
        prices.append(price)
        
        # Random volume
        volume = np.random.uniform(10, 1000)
        volumes.append(volume)
    
    # Create OHLCV data
    sample_df = pd.DataFrame({
        'open': [p * np.random.uniform(0.999, 1.001) for p in prices],
        'high': [p * np.random.uniform(1.001, 1.02) for p in prices],
        'low': [p * np.random.uniform(0.98, 0.999) for p in prices],
        'close': prices,
        'volume': volumes,
        'funding_rate': np.random.normal(0.0001, 0.0005, len(dates))  # Mock funding rate
    }, index=dates)
    
    try:
        # Test feature building
        features_df = build_features(sample_df)
        
        print(f"\n✅ Feature engineering test successful!")
        print(f"   Input shape: {sample_df.shape}")
        print(f"   Output shape: {features_df.shape}")
        
        # Show sample features
        feature_groups = get_feature_groups()
        print(f"\n📊 Feature groups:")
        for group, features in feature_groups.items():
            available = [f for f in features if f in features_df.columns]
            print(f"   {group}: {len(available)} features")
        
        print(f"\n📈 Sample feature values:")
        sample_features = ['log_ret_1h', 'ma_20', 'rsi_14', 'vol_24h']
        for feature in sample_features:
            if feature in features_df.columns:
                value = features_df[feature].iloc[-1]
                print(f"   {feature}: {value:.6f}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()