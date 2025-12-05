"""
Vectorized backtesting module for trading signal evaluation.

This module provides efficient backtesting capabilities without loops,
supporting position tracking, fee calculation, and comprehensive performance metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings


def calculate_performance_stats(equity_curve: pd.Series, 
                               returns: pd.Series,
                               trades_df: pd.DataFrame,
                               freq: str = '1H') -> Dict:
    """
    Calculate comprehensive performance statistics from equity curve and trades.
    
    Args:
        equity_curve: Series with portfolio equity over time
        returns: Series with period returns
        trades_df: DataFrame with individual trade information
        freq: Frequency of data ('1H', '1D', etc.) for annualization
    
    Returns:
        Dictionary with performance metrics
    """
    # Annualization factor based on frequency
    freq_map = {
        '1H': 365.25 * 24,      # Hourly
        '4H': 365.25 * 6,       # 4-hourly  
        '1D': 365.25,           # Daily
        '1W': 52.25,            # Weekly
    }
    periods_per_year = freq_map.get(freq, 365.25 * 24)  # Default to hourly
    
    # Basic metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    total_periods = len(equity_curve) - 1
    
    # CAGR
    years = total_periods / periods_per_year
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    
    # Volatility and Sharpe ratio
    returns_clean = returns.dropna()
    if len(returns_clean) > 1:
        volatility = returns_clean.std() * np.sqrt(periods_per_year)
        sharpe = (returns_clean.mean() * periods_per_year) / volatility if volatility > 0 else 0
    else:
        volatility = 0
        sharpe = 0
    
    # Drawdown analysis
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Find max drawdown duration
    dd_periods = (drawdown < -0.01)  # Consider 1%+ as significant drawdown
    if dd_periods.any():
        # Find consecutive periods of drawdown
        dd_groups = (dd_periods != dd_periods.shift()).cumsum()
        dd_durations = dd_periods.groupby(dd_groups).sum()
        max_dd_duration = dd_durations.max() if len(dd_durations) > 0 else 0
    else:
        max_dd_duration = 0
    
    # Trade analysis
    if len(trades_df) > 0:
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df)
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if losing_trades['pnl'].sum() != 0 else np.inf
        
        # Trade duration analysis
        avg_trade_duration = trades_df['duration'].mean()
        max_trade_duration = trades_df['duration'].max()
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        avg_trade_duration = 0
        max_trade_duration = 0
    
    # Calmar ratio (CAGR / |Max Drawdown|)
    calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_dd_duration,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_trades': len(trades_df),
        'avg_trade_duration': avg_trade_duration,
        'max_trade_duration': max_trade_duration,
        'periods_per_year': periods_per_year,
        'total_years': years
    }


def backtest_signals(df: pd.DataFrame,
                    signal_col: str,
                    fee_rate: float = 0.0004,
                    initial_capital: float = 10000.0,
                    execution_price: str = 'open',  # 'open' or 'close'
                    freq: str = '1H') -> Dict:
    """
    Vectorized backtesting of trading signals.
    
    Args:
        df: DataFrame with OHLCV data and signal column
        signal_col: Name of column containing signals (-1, 0, 1)
        fee_rate: Trading fee rate per side (default 0.04%)
        initial_capital: Starting portfolio value
        execution_price: Price to use for trades ('open' or 'close')
        freq: Data frequency for performance calculation
    
    Returns:
        Dictionary containing:
        - 'equity_curve': Portfolio value over time
        - 'stats': Performance statistics
        - 'trades': Individual trade details
        - 'positions': Position information over time
    """
    # Input validation
    required_cols = ['open', 'high', 'low', 'close', signal_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if execution_price not in ['open', 'close']:
        raise ValueError("execution_price must be 'open' or 'close'")
    
    # Work with a copy
    data = df.copy().reset_index(drop=True)
    
    # Ensure signals are valid
    valid_signals = [-1, 0, 1]
    invalid_signals = data[signal_col].dropna().unique()
    invalid_signals = [s for s in invalid_signals if s not in valid_signals]
    if invalid_signals:
        warnings.warn(f"Invalid signal values found: {invalid_signals}. Should be -1, 0, or 1.")
    
    # Fill NaN signals with 0 (no position)
    data[signal_col] = data[signal_col].fillna(0)
    
    # Shift signals to next period (trade on signal, execute next period)
    data['position'] = data[signal_col].shift(1).fillna(0)
    
    # Calculate position changes (when we enter/exit)
    data['position_change'] = data['position'].diff().fillna(data['position'])
    
    # Determine execution price
    data['exec_price'] = data[execution_price]
    
    # Calculate returns for each position
    data['price_change'] = data['exec_price'].pct_change()
    data['strategy_returns'] = data['position'].shift(1) * data['price_change']
    
    # Calculate trading costs (fees)
    # Fee is charged when position changes (entering or exiting)
    data['trade_occurred'] = (data['position_change'] != 0).astype(float)
    data['fees'] = data['trade_occurred'] * fee_rate
    
    # For position changes, we pay fees on the traded amount
    # Simplification: assume fees are proportional to portfolio value
    data['net_returns'] = data['strategy_returns'] - data['fees']
    
    # Calculate equity curve
    data['equity_multiplier'] = (1 + data['net_returns']).fillna(1)
    data['equity_curve'] = initial_capital * data['equity_multiplier'].cumprod()
    
    # Create trades DataFrame
    trades_list = []
    
    # Find all position changes
    position_changes = data[data['trade_occurred'] == 1].copy()
    
    if len(position_changes) > 0:
        trade_id = 0
        current_position = 0
        entry_price = None
        entry_idx = None
        
        for idx, row in position_changes.iterrows():
            new_position = row['position']
            
            # If we're entering a new position from flat
            if current_position == 0 and new_position != 0:
                entry_price = row['exec_price']
                entry_idx = idx
                current_position = new_position
                
            # If we're changing position (exit and potentially re-enter)
            elif current_position != 0 and new_position != current_position:
                # Close current trade
                if entry_price is not None and entry_idx is not None:
                    exit_price = row['exec_price']
                    
                    # Calculate PnL
                    if current_position == 1:  # Long position
                        pnl_pct = (exit_price / entry_price - 1) - (2 * fee_rate)  # Entry + exit fees
                    else:  # Short position (-1)
                        pnl_pct = (entry_price / exit_price - 1) - (2 * fee_rate)  # Entry + exit fees
                    
                    pnl_dollar = initial_capital * pnl_pct  # Simplified PnL calculation
                    
                    trades_list.append({
                        'trade_id': trade_id,
                        'entry_time': data.iloc[entry_idx].get('timestamp', entry_idx),
                        'exit_time': row.get('timestamp', idx),
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': current_position,
                        'duration': idx - entry_idx,
                        'pnl_pct': pnl_pct,
                        'pnl': pnl_dollar
                    })
                    trade_id += 1
                
                # Start new position if not flat
                if new_position != 0:
                    entry_price = row['exec_price']
                    entry_idx = idx
                    current_position = new_position
                else:
                    current_position = 0
                    entry_price = None
                    entry_idx = None
    
    trades_df = pd.DataFrame(trades_list)
    
    # Calculate performance statistics
    returns_series = data['net_returns'].fillna(0)
    equity_series = data['equity_curve']
    
    stats = calculate_performance_stats(equity_series, returns_series, trades_df, freq)
    
    # Prepare position information
    positions_df = data[['position', 'position_change', 'exec_price', 'equity_curve', 'strategy_returns', 'net_returns']].copy()
    if 'timestamp' in data.columns:
        positions_df['timestamp'] = data['timestamp']
    
    return {
        'equity_curve': equity_series,
        'stats': stats,
        'trades': trades_df,
        'positions': positions_df,
        'summary': {
            'initial_capital': initial_capital,
            'final_capital': equity_series.iloc[-1],
            'total_return_pct': stats['total_return'] * 100,
            'cagr_pct': stats['cagr'] * 100,
            'max_drawdown_pct': stats['max_drawdown'] * 100,
            'sharpe_ratio': stats['sharpe_ratio'],
            'total_trades': stats['total_trades'],
            'win_rate_pct': stats['win_rate'] * 100
        }
    }


def compare_strategies(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple strategy backtest results.
    
    Args:
        results_dict: Dictionary with strategy names as keys and backtest results as values
    
    Returns:
        DataFrame comparing key metrics across strategies
    """
    comparison_data = []
    
    for strategy_name, results in results_dict.items():
        stats = results['stats']
        summary = results['summary']
        
        comparison_data.append({
            'Strategy': strategy_name,
            'Total Return (%)': summary['total_return_pct'],
            'CAGR (%)': summary['cagr_pct'], 
            'Sharpe Ratio': stats['sharpe_ratio'],
            'Calmar Ratio': stats['calmar_ratio'],
            'Max Drawdown (%)': summary['max_drawdown_pct'],
            'Win Rate (%)': summary['win_rate_pct'],
            'Profit Factor': stats['profit_factor'],
            'Total Trades': stats['total_trades'],
            'Avg Trade Duration': stats['avg_trade_duration']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df.round(2)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    
    # Generate sample OHLCV data
    price = 100
    prices = []
    for _ in range(1000):
        change = np.random.normal(0, 0.01)
        price *= (1 + change)
        prices.append(price)
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    
    # Generate sample signals (simple momentum strategy)
    sample_data['returns'] = sample_data['close'].pct_change()
    sample_data['signal'] = np.where(sample_data['returns'] > 0.01, 1, 
                                   np.where(sample_data['returns'] < -0.01, -1, 0))
    
    # Run backtest
    results = backtest_signals(sample_data, 'signal', fee_rate=0.001)
    
    print("Sample Backtest Results:")
    print(f"Total Return: {results['summary']['total_return_pct']:.2f}%")
    print(f"CAGR: {results['summary']['cagr_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['summary']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['summary']['max_drawdown_pct']:.2f}%")
    print(f"Total Trades: {results['summary']['total_trades']}")
    print(f"Win Rate: {results['summary']['win_rate_pct']:.2f}%")
