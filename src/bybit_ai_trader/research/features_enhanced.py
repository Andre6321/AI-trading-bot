"""
Enhanced feature engineering with multi-timeframe and advanced indicators.
Adds 20+ new features for improved model performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import talib


def add_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features from multiple timeframes (1H, 4H, 1D aggregations).
    
    Args:
        df: DataFrame with OHLCV data at 1H frequency
        
    Returns:
        DataFrame with multi-timeframe features
    """
    df = df.copy()
    
    print("   Adding multi-timeframe features...")
    
    # 4H aggregations (for 4H trend)
    df['close_4h'] = df['close'].rolling(4).mean()
    df['high_4h'] = df['high'].rolling(4).max()
    df['low_4h'] = df['low'].rolling(4).min()
    
    # Daily aggregations (for daily trend)
    df['close_1d'] = df['close'].rolling(24).mean()
    df['high_1d'] = df['high'].rolling(24).max()
    df['low_1d'] = df['low'].rolling(24).min()
    df['volume_1d'] = df['volume'].rolling(24).sum()
    
    # Multi-timeframe RSI (using TA-Lib on full series, then resample)
    df['rsi_1h'] = talib.RSI(df['close'], timeperiod=14)
    
    # For 4H and 1D RSI, use resampled close prices
    # Simple approximation: calculate RSI on smoothed prices
    df['rsi_4h'] = talib.RSI(df['close'].rolling(4).mean(), timeperiod=14)
    df['rsi_1d'] = talib.RSI(df['close'].rolling(24).mean(), timeperiod=14)
    
    # Trend alignment (all timeframes pointing same direction)
    df['ma_20_1h'] = df['close'].rolling(20).mean()
    df['ma_20_4h'] = df['close'].rolling(80).mean()  # 20 * 4H
    df['ma_20_1d'] = df['close'].rolling(480).mean()  # 20 * 1D
    
    df['trend_alignment'] = (
        ((df['close'] > df['ma_20_1h']).astype(int) +
         (df['close'] > df['ma_20_4h']).astype(int) +
         (df['close'] > df['ma_20_1d']).astype(int)) / 3
    )
    
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility and Bollinger Band features.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with volatility features
    """
    df = df.copy()
    
    print("   Adding volatility features...")
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    # Bollinger Band Width (volatility indicator)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Price position in BB (0 = at lower band, 1 = at upper band)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # ATR (Average True Range)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # ATR as percentage of price (normalized volatility)
    df['atr_pct'] = df['atr'] / df['close']
    
    # ATR percentile (is current volatility high or low relative to recent history?)
    df['atr_percentile'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 1 else np.nan
    )
    
    # Historical volatility (standard deviation of returns)
    df['hist_volatility'] = df['close'].pct_change().rolling(24).std() * np.sqrt(24)
    
    # Volatility regime (high/low)
    df['volatility_regime'] = (df['atr_percentile'] > 0.7).astype(int)
    
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume analysis features.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with volume features
    """
    df = df.copy()
    
    print("   Adding volume features...")
    
    # Volume moving averages
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ma_50'] = df['volume'].rolling(50).mean()
    
    # Volume ratio (current vs average)
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # Volume spike detection
    df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)
    
    # VWAP (Volume Weighted Average Price)
    df['vwap'] = (df['close'] * df['volume']).rolling(24).sum() / df['volume'].rolling(24).sum()
    df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    
    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma'] = df['obv'].rolling(20).mean()
    df['obv_trend'] = (df['obv'] > df['obv_ma']).astype(int)
    
    # Volume-Price Trend
    df['vpt'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
    
    return df


def add_trend_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend strength and momentum features.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with trend strength features
    """
    df = df.copy()
    
    print("   Adding trend strength features...")
    
    # ADX (Average Directional Index) - trend strength
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Strong trend: ADX > 25
    df['strong_trend'] = (df['adx'] > 25).astype(int)
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
        df['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df['macd_cross'] = ((df['macd'] > df['macd_signal']).astype(int).diff() != 0).astype(int)
    
    # Parabolic SAR
    df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
    df['sar_trend'] = (df['close'] > df['sar']).astype(int)
    
    # CCI (Commodity Channel Index)
    df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
    
    # Williams %R
    df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Momentum oscillators
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
    
    # Rate of Change
    df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
    df['roc_20'] = talib.ROC(df['close'], timeperiod=20)
    
    return df


def add_market_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market structure features (highs/lows, support/resistance).
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with market structure features
    """
    df = df.copy()
    
    print("   Adding market structure features...")
    
    # Higher highs / Lower lows detection
    df['swing_high'] = df['high'].rolling(5, center=True).max() == df['high']
    df['swing_low'] = df['low'].rolling(5, center=True).min() == df['low']
    
    # Distance from recent high/low
    df['distance_from_high'] = (df['close'] - df['high'].rolling(50).max()) / df['high'].rolling(50).max()
    df['distance_from_low'] = (df['close'] - df['low'].rolling(50).min()) / df['low'].rolling(50).min()
    
    # Candle patterns
    df['body_size'] = abs(df['close'] - df['open']) / df['open']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
    
    # Doji detection (small body, long shadows)
    df['is_doji'] = (df['body_size'] < 0.001).astype(int)
    
    # Support/Resistance zones (price clustering)
    df['price_level'] = (df['close'] // 1000) * 1000  # Round to nearest 1000
    df['level_touches'] = df.groupby('price_level')['close'].transform('count')
    
    return df


def add_funding_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add funding rate features for sentiment analysis.
    
    Args:
        df: DataFrame with funding_rate column (if available)
        
    Returns:
        DataFrame with funding rate features
    """
    if 'funding_rate' not in df.columns:
        print("   Skipping funding rate features (not available)")
        return df
    
    print("   Adding funding rate features...")
    
    # Moving averages of funding rate
    df['funding_rate_ma_8h'] = df['funding_rate'].rolling(8).mean()
    df['funding_rate_ma_24h'] = df['funding_rate'].rolling(24).mean()
    df['funding_rate_ma_168h'] = df['funding_rate'].rolling(168).mean()  # Weekly
    
    # Funding rate volatility
    df['funding_rate_std'] = df['funding_rate'].rolling(24).std()
    
    # Extreme funding rates (potential reversal signal)
    df['funding_rate_extreme'] = (
        df['funding_rate'].abs() > df['funding_rate'].rolling(168).quantile(0.95)
    ).astype(int)
    
    # Funding rate momentum
    df['funding_rate_change'] = df['funding_rate'].diff()
    df['funding_rate_trend'] = np.where(
        df['funding_rate_ma_8h'] > df['funding_rate_ma_24h'], 1, -1
    )
    
    # Positive vs negative funding (market bias)
    df['funding_positive_ratio'] = (
        df['funding_rate'].rolling(24).apply(lambda x: (x > 0).sum() / len(x))
    )
    
    return df


def build_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build comprehensive feature set with all enhancements.
    
    Args:
        df: Raw OHLCV DataFrame
        
    Returns:
        DataFrame with all features
    """
    print("🔧 Building ENHANCED features...")
    print(f"   Input shape: {df.shape}")
    
    # Start with basic features from original build_features
    from bybit_ai_trader.research.features import build_features
    df = build_features(df)
    
    # Add new advanced features
    df = add_multi_timeframe_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_trend_strength_features(df)
    df = add_market_structure_features(df)
    df = add_funding_rate_features(df)  # NEW
    
    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan)
    
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    
    print(f"   Dropped {dropped_rows} rows with NaN values")
    print(f"   Final shape: {df.shape}")
    print(f"   Total features: {len(df.columns)}")
    
    return df


def get_feature_importance_groups() -> Dict[str, List[str]]:
    """
    Group features by category for analysis.
    
    Returns:
        Dictionary of feature groups
    """
    return {
        "price_action": ["log_ret_1h", "log_ret_4h", "log_ret_24h", "momentum_3h", "momentum_12h"],
        "trend": ["ma_10", "ma_20", "ma_50", "ma_100", "adx", "strong_trend", "sar_trend"],
        "momentum": ["rsi_14", "rsi_1h", "rsi_4h", "rsi_1d", "macd", "macd_hist", "cci", "williams_r"],
        "volatility": ["atr_14", "atr_pct", "atr_percentile", "bb_width", "bb_position", "hist_volatility"],
        "volume": ["vol_24h", "volume_ratio", "volume_spike", "obv_trend", "vpt"],
        "multi_timeframe": ["trend_alignment", "close_4h", "close_1d"],
        "market_structure": ["distance_from_high", "distance_from_low", "body_size"]
    }
