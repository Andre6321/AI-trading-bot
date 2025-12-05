"""
Trading strategies for backtesting.

This module contains various trading strategy implementations that generate
signals based on technical indicators and market conditions.
"""
import pandas as pd
import numpy as np
from typing import Union


def ma_trend_strategy(df: pd.DataFrame) -> pd.Series:
    """
    Moving Average Trend Strategy.
    
    Uses MA20 and MA50:
    - signal = 1 when close > MA50 and MA20 > MA50
    - signal = 0 otherwise (flat)
    
    Args:
        df: DataFrame with 'close', 'ma_20', 'ma_50' columns
    
    Returns:
        pd.Series of signals aligned with df index
    """
    # Validate required columns
    required_cols = ['close', 'ma_20', 'ma_50']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for MA trend strategy: {missing_cols}")
    
    # Generate signals
    condition1 = df['close'] > df['ma_50']  # Price above long MA
    condition2 = df['ma_20'] > df['ma_50']  # Short MA above long MA
    
    signals = pd.Series(0, index=df.index)  # Default to flat
    signals[condition1 & condition2] = 1   # Long when both conditions met
    
    return signals


def momentum_strategy(df: pd.DataFrame, 
                     rsi_oversold: float = 30, 
                     rsi_overbought: float = 70) -> pd.Series:
    """
    RSI Momentum Strategy with trend filter.
    
    Args:
        df: DataFrame with 'rsi_14', 'close', 'ma_50' columns
        rsi_oversold: RSI level for oversold (buy signal)
        rsi_overbought: RSI level for overbought (sell signal)
    
    Returns:
        pd.Series of signals (-1, 0, 1)
    """
    required_cols = ['rsi_14', 'close', 'ma_50']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for momentum strategy: {missing_cols}")
    
    signals = pd.Series(0, index=df.index)
    
    # Long signals: RSI oversold AND price above MA50 (trend filter)
    long_condition = (df['rsi_14'] < rsi_oversold) & (df['close'] > df['ma_50'])
    signals[long_condition] = 1
    
    # Short signals: RSI overbought AND price below MA50
    short_condition = (df['rsi_14'] > rsi_overbought) & (df['close'] < df['ma_50'])
    signals[short_condition] = -1
    
    return signals


def mean_reversion_strategy(df: pd.DataFrame, 
                           z_threshold: float = 2.0) -> pd.Series:
    """
    Mean Reversion Strategy using price deviation from MA.
    
    Args:
        df: DataFrame with 'close', 'ma_20' columns
        z_threshold: Z-score threshold for mean reversion signals
    
    Returns:
        pd.Series of signals (-1, 0, 1)
    """
    required_cols = ['close', 'ma_20']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for mean reversion strategy: {missing_cols}")
    
    # Calculate rolling z-score of price vs MA20
    price_deviation = df['close'] - df['ma_20']
    rolling_std = price_deviation.rolling(window=20).std()
    z_score = price_deviation / rolling_std
    
    signals = pd.Series(0, index=df.index)
    
    # Mean reversion logic: buy when oversold, sell when overbought
    signals[z_score < -z_threshold] = 1   # Buy when price below MA by z_threshold std devs
    signals[z_score > z_threshold] = -1   # Sell when price above MA by z_threshold std devs
    
    return signals


def volatility_breakout_strategy(df: pd.DataFrame, 
                                atr_multiplier: float = 2.0) -> pd.Series:
    """
    Volatility Breakout Strategy using ATR.
    
    Args:
        df: DataFrame with 'close', 'ma_20', 'atr_14' columns
        atr_multiplier: Multiplier for ATR to define breakout threshold
    
    Returns:
        pd.Series of signals (-1, 0, 1)
    """
    required_cols = ['close', 'ma_20', 'atr_14']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for volatility breakout strategy: {missing_cols}")
    
    # Calculate breakout levels
    upper_breakout = df['ma_20'] + (df['atr_14'] * atr_multiplier)
    lower_breakout = df['ma_20'] - (df['atr_14'] * atr_multiplier)
    
    signals = pd.Series(0, index=df.index)
    
    # Breakout signals
    signals[df['close'] > upper_breakout] = 1   # Long breakout
    signals[df['close'] < lower_breakout] = -1  # Short breakout
    
    return signals


def funding_sentiment_strategy(df: pd.DataFrame, 
                              funding_threshold: float = 1.5) -> pd.Series:
    """
    Sentiment Strategy using funding rate z-score.
    
    Contrarian strategy: when funding is extremely positive (longs pay shorts),
    expect reversal to downside and vice versa.
    
    Args:
        df: DataFrame with 'funding_z_score' column
        funding_threshold: Z-score threshold for extreme funding
    
    Returns:
        pd.Series of signals (-1, 0, 1)
    """
    required_cols = ['funding_z_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for funding sentiment strategy: {missing_cols}")
    
    signals = pd.Series(0, index=df.index)
    
    # Contrarian signals based on extreme funding
    signals[df['funding_z_score'] > funding_threshold] = -1   # Short when funding very positive
    signals[df['funding_z_score'] < -funding_threshold] = 1   # Long when funding very negative
    
    return signals


def combined_strategy(df: pd.DataFrame) -> pd.Series:
    """
    Combined Strategy using multiple signals.
    
    Combines trend, momentum, and sentiment for robust signals.
    
    Args:
        df: DataFrame with all required columns for sub-strategies
    
    Returns:
        pd.Series of signals (-1, 0, 1)
    """
    signals = pd.Series(0, index=df.index)
    
    try:
        # Get individual strategy signals
        ma_signals = ma_trend_strategy(df)
        momentum_signals = momentum_strategy(df)
        
        # Optional sentiment signal if funding data available
        if 'funding_z_score' in df.columns and not df['funding_z_score'].isna().all():
            sentiment_signals = funding_sentiment_strategy(df)
        else:
            sentiment_signals = pd.Series(0, index=df.index)
        
        # Combine signals with weighted approach
        # MA trend gets 40% weight, momentum 40%, sentiment 20%
        combined_score = (0.4 * ma_signals + 
                         0.4 * momentum_signals + 
                         0.2 * sentiment_signals)
        
        # Convert to discrete signals
        signals[combined_score > 0.5] = 1
        signals[combined_score < -0.5] = -1
        
    except ValueError as e:
        print(f"Warning: Could not generate combined strategy - {e}")
        # Fallback to simple MA strategy
        if all(col in df.columns for col in ['close', 'ma_20', 'ma_50']):
            signals = ma_trend_strategy(df)
    
    return signals


def get_strategy_list() -> dict:
    """
    Return dictionary of available strategies.
    
    Returns:
        Dict mapping strategy names to functions
    """
    return {
        'ma_trend': ma_trend_strategy,
        'momentum': momentum_strategy,
        'mean_reversion': mean_reversion_strategy,
        'volatility_breakout': volatility_breakout_strategy,
        'funding_sentiment': funding_sentiment_strategy,
        'combined': combined_strategy
    }


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing strategies
    np.random.seed(42)
    
    # Generate sample price data
    dates = pd.date_range('2023-01-01', periods=500, freq='H')
    price = 100
    prices = []
    for _ in range(500):
        change = np.random.normal(0, 0.01)
        price *= (1 + change)
        prices.append(price)
    
    # Create sample DataFrame with indicators
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'ma_20': pd.Series(prices).rolling(20).mean(),
        'ma_50': pd.Series(prices).rolling(50).mean(),
        'rsi_14': 50 + 20 * np.random.randn(500),  # Mock RSI
        'atr_14': np.random.uniform(0.5, 2.0, 500),  # Mock ATR
        'funding_z_score': np.random.randn(500)  # Mock funding z-score
    })
    
    # Test strategies
    strategies = get_strategy_list()
    
    print("Testing trading strategies on sample data:")
    for name, strategy_func in strategies.items():
        try:
            signals = strategy_func(sample_df)
            signal_counts = signals.value_counts().sort_index()
            print(f"\n{name}:")
            print(f"  Signal distribution: {signal_counts.to_dict()}")
            print(f"  Non-zero signals: {(signals != 0).sum()}/{len(signals)} ({(signals != 0).mean()*100:.1f}%)")
        except Exception as e:
            print(f"\n{name}: Error - {e}")
    
    print("\nStrategy testing complete!")
