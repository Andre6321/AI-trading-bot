"""
Comprehensive backtest of optimized ML strategy with:
- Optuna-tuned ensemble model
- Data-driven SL/TP levels (8%/4%)
- Volatility-adaptive risk management
- Multi-year test period
"""

import sys
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "src"))

from models.ml_dataset import build_ml_dataset


def simulate_trades_with_sltp(
    df: pd.DataFrame,
    predictions: np.ndarray,
    threshold: float = 0.55,
    sl_pct: float = 0.08,  # 8% stop loss (data-driven optimal)
    tp_pct: float = 0.04,  # 4% take profit (data-driven optimal)
    initial_capital: float = 10000,
    fee_rate: float = 0.0006,  # 0.06% Bybit fee
    use_volatility_adaptive: bool = True
) -> dict:
    """
    Simulate trading with stop-loss and take-profit based on optimized levels.
    
    Args:
        df: DataFrame with OHLC data
        predictions: Model prediction probabilities
        threshold: Prediction threshold for entry
        sl_pct: Stop loss percentage (8% optimal from data)
        tp_pct: Take profit percentage (4% optimal from data)
        initial_capital: Starting capital
        fee_rate: Trading fee rate
        use_volatility_adaptive: Use volatility-adaptive SL/TP
    
    Returns:
        Dictionary with trade results and equity curve
    """
    capital = initial_capital
    position = None
    trades = []
    equity_curve = [initial_capital]
    
    # Ensure predictions and df have same length
    min_len = min(len(df), len(predictions))
    
    for i in range(min_len):
        current_price = df['close'].iloc[i]
        pred_prob = predictions[i]
        
        # Determine volatility regime if adaptive
        if use_volatility_adaptive and 'atr_14' in df.columns:
            atr_pct = df['atr_14'].iloc[i] / current_price
            
            if atr_pct < 0.0025:  # Low volatility
                current_sl_pct = 0.05
                current_tp_pct = 0.02
            elif atr_pct < 0.005:  # Normal volatility
                current_sl_pct = 0.05
                current_tp_pct = 0.03
            else:  # High volatility
                current_sl_pct = 0.08
                current_tp_pct = 0.04
        else:
            current_sl_pct = sl_pct
            current_tp_pct = tp_pct
        
        # Check if we should enter a position
        if position is None and pred_prob > threshold:
            # Enter long position
            entry_price = current_price
            entry_fee = capital * fee_rate
            position_size = (capital - entry_fee) / entry_price
            
            position = {
                'entry_idx': i,
                'entry_price': entry_price,
                'entry_time': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i,
                'size': position_size,
                'sl_price': entry_price * (1 - current_sl_pct),
                'tp_price': entry_price * (1 + current_tp_pct),
                'sl_pct': current_sl_pct,
                'tp_pct': current_tp_pct
            }
            
            capital -= entry_fee
        
        # Check if we should exit position
        elif position is not None:
            low_price = df['low'].iloc[i] if 'low' in df.columns else current_price
            high_price = df['high'].iloc[i] if 'high' in df.columns else current_price
            
            exit_price = None
            exit_reason = None
            
            # Check stop loss hit
            if low_price <= position['sl_price']:
                exit_price = position['sl_price']
                exit_reason = 'SL'
            
            # Check take profit hit
            elif high_price >= position['tp_price']:
                exit_price = position['tp_price']
                exit_reason = 'TP'
            
            # Exit if signal changes or time-based exit (optional)
            elif pred_prob < 0.45:  # Exit threshold
                exit_price = current_price
                exit_reason = 'Signal'
            
            if exit_price is not None:
                # Calculate PnL
                gross_pnl = position['size'] * (exit_price - position['entry_price'])
                exit_fee = position['size'] * exit_price * fee_rate
                net_pnl = gross_pnl - exit_fee
                
                capital += position['size'] * exit_price - exit_fee
                
                # Record trade
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'pnl_pct': (exit_price / position['entry_price'] - 1) * 100,
                    'duration': i - position['entry_idx'],
                    'exit_reason': exit_reason,
                    'sl_pct': position['sl_pct'],
                    'tp_pct': position['tp_pct']
                })
                
                position = None
        
        equity_curve.append(capital)
    
    # Close any remaining position
    if position is not None:
        exit_price = df['close'].iloc[-1]
        gross_pnl = position['size'] * (exit_price - position['entry_price'])
        exit_fee = position['size'] * exit_price * fee_rate
        net_pnl = gross_pnl - exit_fee
        capital += position['size'] * exit_price - exit_fee
        
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else len(df) - 1,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl': net_pnl,
            'pnl_pct': (exit_price / position['entry_price'] - 1) * 100,
            'duration': len(df) - 1 - position['entry_idx'],
            'exit_reason': 'EOD',
            'sl_pct': position['sl_pct'],
            'tp_pct': position['tp_pct']
        })
        
        equity_curve[-1] = capital
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate performance metrics
    if len(trades) > 0:
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        
        total_return = (capital - initial_capital) / initial_capital * 100
        win_rate = len(wins) / len(trades_df) * 100
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf')
        
        # Calculate max drawdown
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (annualized)
        returns = equity_series.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24)  # Hourly data
        else:
            sharpe_ratio = 0
        
        # Exit reason breakdown
        exit_reasons = trades_df['exit_reason'].value_counts()
        
        summary = {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'total_return': total_return,
            'final_capital': capital,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sl_hits': exit_reasons.get('SL', 0),
            'tp_hits': exit_reasons.get('TP', 0),
            'signal_exits': exit_reasons.get('Signal', 0)
        }
    else:
        summary = {
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'final_capital': initial_capital,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sl_hits': 0,
            'tp_hits': 0,
            'signal_exits': 0
        }
    
    return {
        'trades': trades_df,
        'equity_curve': equity_curve,
        'summary': summary
    }


def main():
    """Run comprehensive backtest of optimized strategy."""
    print("=" * 70)
    print("🚀 OPTIMIZED ML STRATEGY BACKTEST")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading 5-year historical data...")
    data_path = Path("data/processed/btcusdt_1h_full_history_enhanced.parquet")
    df = pd.read_parquet(data_path)
    print(f"   Loaded {len(df):,} samples")
    
    # Load optimized model
    print("\n📦 Loading Optuna-optimized ensemble model...")
    model_path = Path("models/ensemble_btcusdt_h4_optuna_optimized.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    metadata_path = Path("models/ensemble_btcusdt_h4_optuna_optimized_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"   Model ROC AUC: {metadata['final_roc_auc']:.4f}")
    print(f"   Features: {metadata['n_features']}")
    
    # Prepare dataset
    print("\n📊 Preparing features...")
    X, y, index = build_ml_dataset(df, horizon_hours=4, threshold=0.01)
    print(f"   Dataset: {len(X):,} samples, {len(X.columns)} features")
    
    # Use last 50% of data for out-of-sample testing (2.5 years)
    test_start = len(X) // 2
    X_test = X.iloc[test_start:]
    df_test = df.iloc[test_start:].copy()
    
    print(f"\n🎯 Testing on last 50% of data ({len(X_test):,} samples)")
    print(f"   Approximately 2.5 years of out-of-sample data")
    
    # Generate predictions
    print("\n🔮 Generating predictions...")
    predictions = model.predict_proba(X_test)[:, 1]
    print(f"   Mean prediction: {predictions.mean():.3f}")
    print(f"   Signals (>0.55): {(predictions > 0.55).sum():,} ({(predictions > 0.55).mean() * 100:.1f}%)")
    
    # Run backtests with different configurations
    print("\n" + "=" * 70)
    print("📈 RUNNING BACKTESTS")
    print("=" * 70)
    
    configs = [
        {
            'name': 'Optimized SL/TP (8%/4%)',
            'sl_pct': 0.08,
            'tp_pct': 0.04,
            'use_adaptive': False
        },
        {
            'name': 'Volatility-Adaptive SL/TP',
            'sl_pct': 0.08,
            'tp_pct': 0.04,
            'use_adaptive': True
        },
        {
            'name': 'Old SL/TP (2%/4%) - Baseline',
            'sl_pct': 0.02,
            'tp_pct': 0.04,
            'use_adaptive': False
        }
    ]
    
    results = {}
    for config in configs:
        print(f"\n📊 Testing: {config['name']}")
        result = simulate_trades_with_sltp(
            df_test,
            predictions,
            threshold=0.55,
            sl_pct=config['sl_pct'],
            tp_pct=config['tp_pct'],
            use_volatility_adaptive=config['use_adaptive']
        )
        results[config['name']] = result
        
        summary = result['summary']
        print(f"   Trades: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate']:.1f}%")
        print(f"   Total Return: {summary['total_return']:.2f}%")
        print(f"   Max Drawdown: {summary['max_drawdown']:.2f}%")
        print(f"   Profit Factor: {summary['profit_factor']:.2f}")
        print(f"   TP Hits: {summary['tp_hits']} | SL Hits: {summary['sl_hits']}")
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 70)
    
    comparison_data = []
    for name, result in results.items():
        s = result['summary']
        comparison_data.append({
            'Strategy': name,
            'Return (%)': f"{s['total_return']:.2f}",
            'Win Rate (%)': f"{s['win_rate']:.1f}",
            'Profit Factor': f"{s['profit_factor']:.2f}",
            'Max DD (%)': f"{s['max_drawdown']:.2f}",
            'Sharpe': f"{s['sharpe_ratio']:.2f}",
            'Trades': s['total_trades'],
            'TP Hits': s['tp_hits'],
            'SL Hits': s['sl_hits']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(f"\n{comparison_df.to_string(index=False)}")
    
    # Create visualization
    print("\n📊 Creating visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    colors = ['green', 'blue', 'red']
    for i, (name, result) in enumerate(results.items()):
        equity = result['equity_curve']
        ax1.plot(equity, label=name, linewidth=2, color=colors[i], alpha=0.8)
    
    ax1.set_title('Equity Curves - Optimized vs Baseline Strategies', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=10000, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
    
    # Win rate comparison
    strategies = list(results.keys())
    win_rates = [results[s]['summary']['win_rate'] for s in strategies]
    returns = [results[s]['summary']['total_return'] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    ax2_twin = ax2.twinx()
    bars1 = ax2.bar(x - width/2, win_rates, width, label='Win Rate (%)', color='skyblue', alpha=0.8)
    bars2 = ax2_twin.bar(x + width/2, returns, width, label='Total Return (%)', color='lightcoral', alpha=0.8)
    
    ax2.set_ylabel('Win Rate (%)', fontsize=12)
    ax2_twin.set_ylabel('Total Return (%)', fontsize=12)
    ax2.set_title('Win Rate vs Total Return Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=15, ha='right')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path("outputs/optimized_strategy_backtest.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved to: {output_path}")
    
    print("\n✅ Backtest complete!")
    
    # Highlight best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['summary']['total_return'])
    print(f"\n🏆 BEST STRATEGY: {best_strategy[0]}")
    print(f"   Total Return: {best_strategy[1]['summary']['total_return']:.2f}%")
    print(f"   Win Rate: {best_strategy[1]['summary']['win_rate']:.1f}%")
    print(f"   Profit Factor: {best_strategy[1]['summary']['profit_factor']:.2f}")


if __name__ == "__main__":
    main()
