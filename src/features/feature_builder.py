"""
Feature engineering module for Bybit BTCUSDT perpetual futures data.

Transforms raw OHLCV + sentiment data into ML-ready features.
"""
import pandas as pd
import numpy as np
from typing import Optional


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    # True Range calculation
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr


def calculate_z_score(series: pd.Series, window: int = 30) -> pd.Series:
    """Calculate rolling z-score."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    z_score = (series - rolling_mean) / rolling_std
    return z_score


def add_cyclical_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """Add cyclical time features (hour and day of week as sin/cos)."""
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Extract hour and day of week
    hour = df[timestamp_col].dt.hour
    day_of_week = df[timestamp_col].dt.dayofweek
    
    # Convert to cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: raw Bybit BTCUSDT 1h dataframe with OHLCV, funding_rate, open_interest etc.
    Output: dataframe with additional feature columns and with rows with NaNs dropped.
    """
    # Create a copy to avoid modifying the original
    features_df = df.copy()
    
    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print("Building features...")
    
    # === Returns Features ===
    print("Adding return features...")
    features_df['returns_1h'] = np.log(features_df['close'] / features_df['close'].shift(1))
    features_df['returns_4h'] = np.log(features_df['close'] / features_df['close'].shift(4))
    features_df['returns_24h'] = np.log(features_df['close'] / features_df['close'].shift(24))
    
    # === Moving Averages ===
    print("Adding moving averages...")
    for window in [10, 20, 50, 100]:
        features_df[f'ma_{window}'] = features_df['close'].rolling(window=window).mean()
        # Add relative position to moving average
        features_df[f'price_vs_ma_{window}'] = features_df['close'] / features_df[f'ma_{window}'] - 1
    
    # === Technical Indicators ===
    print("Adding technical indicators...")
    features_df['rsi_14'] = calculate_rsi(features_df['close'], window=14)
    features_df['atr_14'] = calculate_atr(features_df['high'], features_df['low'], features_df['close'], window=14)
    
    # ATR as percentage of close price
    features_df['atr_pct'] = features_df['atr_14'] / features_df['close'] * 100
    
    # === Volatility Features ===
    print("Adding volatility features...")
    features_df['volatility_24h'] = features_df['returns_1h'].rolling(window=24).std()
    
    # Volume features
    features_df['volume_ma_24h'] = features_df['volume'].rolling(window=24).mean()
    features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma_24h']
    
    # === Sentiment Features ===
    print("Adding sentiment features...")
    
    # Funding rate features (if available)
    if 'funding_rate' in features_df.columns:
        features_df['funding_z_score'] = calculate_z_score(features_df['funding_rate'], window=30)
        
        # Additional funding features
        features_df['funding_ma_7d'] = features_df['funding_rate'].rolling(window=24*7).mean()
        features_df['funding_vs_ma'] = features_df['funding_rate'] - features_df['funding_ma_7d']
    else:
        print("Warning: funding_rate column not found, skipping funding features")
        features_df['funding_z_score'] = np.nan
        features_df['funding_ma_7d'] = np.nan
        features_df['funding_vs_ma'] = np.nan
    
    # Open interest features (if available)
    if 'open_interest' in features_df.columns:
        features_df['oi_z_score'] = calculate_z_score(features_df['open_interest'], window=30)
        
        # OI rate of change
        features_df['oi_change_1h'] = features_df['open_interest'].pct_change()
        features_df['oi_change_24h_pct'] = features_df['open_interest'].pct_change(24)
    else:
        print("Warning: open_interest column not found, skipping OI features")
        features_df['oi_z_score'] = np.nan
        features_df['oi_change_1h'] = np.nan
        features_df['oi_change_24h_pct'] = np.nan
    
    # === Price Action Features ===
    print("Adding price action features...")
    
    # High-Low spread
    features_df['hl_spread'] = (features_df['high'] - features_df['low']) / features_df['close']
    
    # Close position within the bar
    features_df['close_position'] = (features_df['close'] - features_df['low']) / (features_df['high'] - features_df['low'])
    
    # Price momentum
    features_df['price_momentum_3h'] = (features_df['close'] / features_df['close'].shift(3)) - 1
    features_df['price_momentum_12h'] = (features_df['close'] / features_df['close'].shift(12)) - 1
    
    # === Trend Features ===
    print("Adding trend features...")
    
    # Simple trend: are we above/below MA?
    features_df['above_ma_20'] = (features_df['close'] > features_df['ma_20']).astype(int)
    features_df['above_ma_50'] = (features_df['close'] > features_df['ma_50']).astype(int)
    
    # MA crossovers
    features_df['ma_20_vs_50'] = features_df['ma_20'] / features_df['ma_50'] - 1
    
    # === Time-based Features ===
    print("Adding time features...")
    features_df = add_cyclical_features(features_df)
    
    # === Clean up and finalize ===
    print("Cleaning up NaN values...")
    
    # Count initial NaN rows
    initial_rows = len(features_df)
    
    # Drop rows where key indicators are NaN
    # Use the 100-period MA as the cutoff since it requires the most historical data
    features_df = features_df.dropna(subset=['ma_100', 'atr_14', 'rsi_14'])
    
    final_rows = len(features_df)
    dropped_rows = initial_rows - final_rows
    
    print(f"Dropped {dropped_rows} rows with NaN values ({initial_rows} -> {final_rows})")
    
    # Reset index
    features_df = features_df.reset_index(drop=True)
    
    print(f"Feature engineering complete. Final dataset shape: {features_df.shape}")
    print(f"Feature columns: {[col for col in features_df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]}")
    
    return features_df


def get_feature_groups() -> dict:
    """Return a dictionary of feature groups for easy selection."""
    return {
        'returns': ['returns_1h', 'returns_4h', 'returns_24h'],
        'moving_averages': ['ma_10', 'ma_20', 'ma_50', 'ma_100'],
        'price_vs_ma': ['price_vs_ma_10', 'price_vs_ma_20', 'price_vs_ma_50', 'price_vs_ma_100'],
        'technical': ['rsi_14', 'atr_14', 'atr_pct'],
        'volatility': ['volatility_24h'],
        'volume': ['volume_ma_24h', 'volume_ratio'],
        'sentiment': ['funding_z_score', 'funding_ma_7d', 'funding_vs_ma', 'oi_z_score', 'oi_change_1h', 'oi_change_24h_pct'],
        'price_action': ['hl_spread', 'close_position', 'price_momentum_3h', 'price_momentum_12h'],
        'trend': ['above_ma_20', 'above_ma_50', 'ma_20_vs_50'],
        'time': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    }


if __name__ == "__main__":
    # Example usage
    print("Feature builder module loaded successfully!")
    print("Feature groups available:", list(get_feature_groups().keys()))
