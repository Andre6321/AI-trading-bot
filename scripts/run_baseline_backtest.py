"""
Run baseline MA trend strategy backtest.

This script:
1. Loads processed features from data/processed/btcusdt_1h_features.parquet
2. Generates MA trend strategy signals 
3. Runs backtest using backtest_signals
4. Prints performance statistics
5. Saves equity curve plot to outputs/baseline_ma_equity.png

Usage: python scripts/run_baseline_backtest.py
"""
import os
import sys
from pathlib import Path
import warnings

# Add src to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Import our modules
from backtest.backtester import backtest_signals
from backtest.strategies import ma_trend_strategy


def setup_matplotlib():
    """Configure matplotlib for better plots."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 10
    plt.rcParams['lines.linewidth'] = 1.5


def create_equity_plot(results: dict, strategy_name: str, output_path: Path):
    """
    Create and save equity curve plot.
    
    Args:
        results: Backtest results dictionary
        strategy_name: Name of the strategy for title
        output_path: Path to save the plot
    """
    setup_matplotlib()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    
    # Get data
    positions_df = results['positions']
    equity_curve = results['equity_curve']
    
    # Plot 1: Equity curve
    if 'timestamp' in positions_df.columns:
        dates = pd.to_datetime(positions_df['timestamp'])
        ax1.plot(dates, equity_curve, linewidth=2, color='navy', label='Portfolio Value')
        ax1.set_xlabel('Date')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax1.plot(equity_curve, linewidth=2, color='navy', label='Portfolio Value')
        ax1.set_xlabel('Time Period')
    
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title(f'{strategy_name} - Equity Curve\n'
                  f'Total Return: {results["summary"]["total_return_pct"]:.1f}% | '
                  f'Sharpe: {results["summary"]["sharpe_ratio"]:.2f} | '
                  f'Max DD: {results["summary"]["max_drawdown_pct"]:.1f}%')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add benchmark line (buy and hold)
    initial_value = equity_curve.iloc[0]
    ax1.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
    
    # Plot 2: Drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    
    if 'timestamp' in positions_df.columns:
        ax2.fill_between(dates, drawdown, 0, alpha=0.7, color='red', label='Drawdown')
        ax2.set_xlabel('Date')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.7, color='red', label='Drawdown')
        ax2.set_xlabel('Time Period')
    
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Drawdown Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Equity curve plot saved to: {output_path}")
    plt.close()


def print_performance_summary(results: dict, strategy_name: str):
    """Print formatted performance summary."""
    stats = results['stats']
    summary = results['summary']
    
    print(f"\n{'='*60}")
    print(f"  {strategy_name.upper()} STRATEGY PERFORMANCE")
    print(f"{'='*60}")
    
    print(f"\n📈 RETURNS & RISK:")
    print(f"   Total Return:     {summary['total_return_pct']:>8.2f}%")
    print(f"   CAGR:             {summary['cagr_pct']:>8.2f}%")
    print(f"   Volatility:       {stats['volatility']*100:>8.2f}%")
    print(f"   Sharpe Ratio:     {summary['sharpe_ratio']:>8.2f}")
    print(f"   Calmar Ratio:     {stats['calmar_ratio']:>8.2f}")
    
    print(f"\n📉 DRAWDOWN:")
    print(f"   Max Drawdown:     {summary['max_drawdown_pct']:>8.2f}%")
    print(f"   Max DD Duration:  {stats['max_drawdown_duration']:>8.0f} periods")
    
    print(f"\n💰 TRADING:")
    print(f"   Total Trades:     {summary['total_trades']:>8.0f}")
    print(f"   Win Rate:         {summary['win_rate_pct']:>8.2f}%")
    print(f"   Profit Factor:    {stats['profit_factor']:>8.2f}")
    print(f"   Avg Win:          {stats['avg_win']:>8.2f}")
    print(f"   Avg Loss:         {stats['avg_loss']:>8.2f}")
    print(f"   Avg Trade Dur:    {stats['avg_trade_duration']:>8.1f} periods")
    
    print(f"\n💵 CAPITAL:")
    print(f"   Initial Capital:  ${summary['initial_capital']:>8,.0f}")
    print(f"   Final Capital:    ${summary['final_capital']:>8,.0f}")
    print(f"   Total Years:      {stats['total_years']:>8.2f}")


def main():
    """Main execution function."""
    print("🚀 Baseline MA Trend Strategy Backtest")
    print("=" * 50)
    
    # Define paths
    features_path = script_dir.parent / "data" / "processed" / "btcusdt_1h_features.parquet"
    outputs_dir = script_dir.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    plot_path = outputs_dir / "baseline_ma_equity.png"
    
    # Check if features file exists
    if not features_path.exists():
        print(f"❌ Features file not found: {features_path}")
        print("   Please run 'python scripts/build_dataset.py' first.")
        return 1
    
    try:
        # Load processed features
        print(f"📥 Loading features from {features_path.name}...")
        df = pd.read_parquet(features_path)
        print(f"   Loaded {len(df):,} rows with {len(df.columns)} columns")
        
        # Check required columns for MA strategy
        required_cols = ['close', 'ma_20', 'ma_50']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return 1
        
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Generate MA trend signals
        print(f"\n🎯 Generating MA trend strategy signals...")
        signals = ma_trend_strategy(df)
        signal_stats = signals.value_counts().sort_index()
        print(f"   Signal distribution: {signal_stats.to_dict()}")
        print(f"   Active signals: {(signals != 0).sum():,} / {len(signals):,} ({(signals != 0).mean()*100:.1f}%)")
        
        # Add signals to dataframe for backtesting
        df_with_signals = df.copy()
        df_with_signals['signals'] = signals
        
        # Run backtest
        print(f"\n🔄 Running backtest...")
        results = backtest_signals(
            df=df_with_signals,
            signal_col='signals',
            fee_rate=0.0004,  # 0.04% fee
            initial_capital=10000,
            execution_price='open',  # Execute at next bar's open
            freq='1H'
        )
        
        # Print performance summary
        print_performance_summary(results, "MA Trend")
        
        # Create and save equity plot
        print(f"\n📊 Creating equity curve visualization...")
        create_equity_plot(results, "MA Trend Strategy", plot_path)
        
        # Additional analysis
        print(f"\n📋 ADDITIONAL ANALYSIS:")
        trades_df = results['trades']
        
        if len(trades_df) > 0:
            print(f"   Longest winning streak: {max([len(list(g)) for k, g in trades_df.groupby((trades_df['pnl'] <= 0).cumsum()) if k == 0], default=0)} trades")
            print(f"   Longest losing streak:  {max([len(list(g)) for k, g in trades_df.groupby((trades_df['pnl'] > 0).cumsum()) if k == 0], default=0)} trades")
            
            # Monthly returns if we have enough data
            if len(df) > 30*24:  # More than 30 days of hourly data
                positions = results['positions'].copy()
                if 'timestamp' in positions.columns:
                    positions['timestamp'] = pd.to_datetime(positions['timestamp'])
                    positions.set_index('timestamp', inplace=True)
                    monthly_returns = positions['net_returns'].resample('M').sum()
                    winning_months = (monthly_returns > 0).sum()
                    total_months = len(monthly_returns)
                    print(f"   Monthly win rate:       {winning_months}/{total_months} ({winning_months/total_months*100:.1f}%)")
        
        print(f"\n✅ Backtest complete! Check {plot_path} for equity curve.")
        return 0
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Suppress pandas warnings for cleaner output
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    
    exit_code = main()
    sys.exit(exit_code)
