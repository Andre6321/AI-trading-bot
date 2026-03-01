"""
Optimize Stop-Loss and Take-Profit levels based on historical data.

Analyzes 5 years of BTC price movements to find optimal SL/TP levels that:
1. Maximize profit factor (gross profit / gross loss)
2. Optimize win rate vs reward/risk ratio
3. Account for different volatility regimes
4. Minimize false stop-outs

Usage: python scripts/optimize_sl_tp.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_data():
    """Load full history enhanced features."""
    data_path = Path(__file__).parent.parent / "data" / "processed" / "btcusdt_1h_full_history_enhanced.parquet"
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        sys.exit(1)
    
    print(f"📥 Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"✅ Loaded {len(df)} samples")
    
    # Need to reload raw data for timestamp
    raw_path = Path(__file__).parent.parent / "data" / "raw" / "btcusdt_1h_full_history.parquet"
    df_raw = pd.read_parquet(raw_path)
    
    # Merge timestamps
    df = df_raw[['timestamp', 'open', 'high', 'low', 'close']].iloc[578:].reset_index(drop=True)  # Skip NaN rows
    df = pd.concat([df, df_raw[['volume']].iloc[578:].reset_index(drop=True)], axis=1)
    
    return df


def calculate_mae_mfe(df: pd.DataFrame, horizon_hours: int = 24) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE).
    
    Args:
        df: OHLCV data
        horizon_hours: Hours to look forward
        
    Returns:
        Tuple of (mae_pct, mfe_pct) series
    """
    print(f"\n📊 Calculating MAE/MFE for {horizon_hours}h horizon...")
    
    mae_list = []
    mfe_list = []
    
    for i in tqdm(range(len(df) - horizon_hours), desc="Analyzing moves"):
        entry_price = df['close'].iloc[i]
        
        # Look at next N hours
        future_highs = df['high'].iloc[i+1:i+1+horizon_hours]
        future_lows = df['low'].iloc[i+1:i+1+horizon_hours]
        
        # Maximum Adverse Excursion (worst drawdown)
        mae = ((future_lows.min() - entry_price) / entry_price) * 100
        
        # Maximum Favorable Excursion (best gain)
        mfe = ((future_highs.max() - entry_price) / entry_price) * 100
        
        mae_list.append(mae)
        mfe_list.append(mfe)
    
    return pd.Series(mae_list), pd.Series(mfe_list)


def simulate_sl_tp_grid(
    df: pd.DataFrame,
    mae: pd.Series,
    mfe: pd.Series,
    sl_range: List[float],
    tp_range: List[float],
    horizon_hours: int = 24
) -> pd.DataFrame:
    """
    Simulate trading with different SL/TP combinations.
    
    Args:
        df: OHLCV data
        mae: Maximum Adverse Excursion series
        mfe: Maximum Favorable Excursion series
        sl_range: List of stop-loss percentages to test
        tp_range: List of take-profit percentages to test
        horizon_hours: Maximum hours to hold
        
    Returns:
        DataFrame with results for each SL/TP combination
    """
    print(f"\n🔬 Testing {len(sl_range)} SL × {len(tp_range)} TP combinations...")
    
    results = []
    
    for sl_pct in tqdm(sl_range, desc="Testing SL/TP"):
        for tp_pct in tp_range:
            wins = 0
            losses = 0
            total_profit = 0
            total_loss = 0
            premature_stops = 0  # Hit SL but later would have hit TP
            
            for i in range(len(mae)):
                # Check if stop-loss hit
                if mae.iloc[i] <= -sl_pct:
                    losses += 1
                    total_loss += sl_pct
                    
                    # Check if it was premature (later hit TP)
                    if mfe.iloc[i] >= tp_pct:
                        premature_stops += 1
                
                # Check if take-profit hit
                elif mfe.iloc[i] >= tp_pct:
                    wins += 1
                    total_profit += tp_pct
            
            # Calculate metrics
            total_trades = wins + losses
            if total_trades == 0:
                continue
            
            win_rate = wins / total_trades
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
            expectancy = (win_rate * tp_pct) - ((1 - win_rate) * sl_pct)
            premature_pct = premature_stops / losses if losses > 0 else 0
            
            results.append({
                'sl_pct': sl_pct,
                'tp_pct': tp_pct,
                'reward_risk': tp_pct / sl_pct,
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'premature_stops': premature_stops,
                'premature_pct': premature_pct,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'net_profit': total_profit - total_loss
            })
    
    return pd.DataFrame(results)


def analyze_by_volatility(df: pd.DataFrame, horizon_hours: int = 24) -> Dict:
    """
    Analyze optimal SL/TP for different volatility regimes.
    
    Args:
        df: OHLCV data with ATR
        horizon_hours: Hours forward
        
    Returns:
        Dictionary with results per volatility regime
    """
    print(f"\n🌡️  Analyzing by volatility regime...")
    
    # Calculate ATR percentage if not available
    if 'atr_pct' not in df.columns:
        df['atr_14'] = df['close'].rolling(14).apply(
            lambda x: np.mean([abs(x.iloc[i] - x.iloc[i-1]) for i in range(1, len(x))])
        )
        df['atr_pct'] = (df['atr_14'] / df['close']) * 100
    
    # Define volatility regimes
    low_vol_threshold = df['atr_pct'].quantile(0.33)
    high_vol_threshold = df['atr_pct'].quantile(0.67)
    
    results = {}
    
    for regime, (low, high) in [
        ('low', (0, low_vol_threshold)),
        ('normal', (low_vol_threshold, high_vol_threshold)),
        ('high', (high_vol_threshold, 100))
    ]:
        mask = (df['atr_pct'] >= low) & (df['atr_pct'] < high)
        df_regime = df[mask].reset_index(drop=True)
        
        if len(df_regime) < 100:
            continue
        
        print(f"   {regime.upper()} volatility: {len(df_regime)} samples")
        
        # Calculate MAE/MFE for this regime
        mae, mfe = calculate_mae_mfe(df_regime, horizon_hours)
        
        # Quick grid search for this regime
        sl_range = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        tp_range = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
        
        grid_results = simulate_sl_tp_grid(df_regime, mae, mfe, sl_range, tp_range, horizon_hours)
        
        # Find best by expectancy
        best = grid_results.nlargest(1, 'expectancy').iloc[0]
        
        results[regime] = {
            'best_sl': best['sl_pct'],
            'best_tp': best['tp_pct'],
            'win_rate': best['win_rate'],
            'profit_factor': best['profit_factor'],
            'expectancy': best['expectancy'],
            'samples': len(df_regime),
            'avg_atr': df_regime['atr_pct'].mean()
        }
    
    return results


def plot_results(results_df: pd.DataFrame, output_dir: Path):
    """Plot optimization results."""
    print(f"\n📈 Generating visualizations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Heatmap: Expectancy by SL/TP
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(['expectancy', 'profit_factor', 'win_rate']):
        pivot = results_df.pivot_table(
            values=metric,
            index='sl_pct',
            columns='tp_pct',
            aggfunc='mean'
        )
        
        im = axes[idx].imshow(pivot.values, cmap='RdYlGn', aspect='auto')
        axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Take Profit %', fontsize=12)
        axes[idx].set_ylabel('Stop Loss %', fontsize=12)
        axes[idx].set_xticks(range(len(pivot.columns)))
        axes[idx].set_xticklabels([f'{x:.1f}' for x in pivot.columns], rotation=45)
        axes[idx].set_yticks(range(len(pivot.index)))
        axes[idx].set_yticklabels([f'{x:.1f}' for x in pivot.index])
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sl_tp_optimization_heatmaps.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'sl_tp_optimization_heatmaps.png'}")
    plt.close()
    
    # 2. Scatter: Win Rate vs Profit Factor
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(
        results_df['win_rate'] * 100,
        results_df['profit_factor'],
        c=results_df['expectancy'],
        s=results_df['total_trades'] / 10,
        alpha=0.6,
        cmap='RdYlGn'
    )
    
    ax.set_xlabel('Win Rate (%)', fontsize=12)
    ax.set_ylabel('Profit Factor', fontsize=12)
    ax.set_title('Win Rate vs Profit Factor (colored by Expectancy)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Expectancy (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'win_rate_vs_profit_factor.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'win_rate_vs_profit_factor.png'}")
    plt.close()


def main():
    """Main optimization pipeline."""
    print("=" * 70)
    print("🎯 STOP-LOSS & TAKE-PROFIT OPTIMIZATION")
    print("=" * 70)
    
    # Load data
    df = load_data()
    
    # Calculate MAE/MFE for 24-hour horizon
    HORIZON_HOURS = 24
    mae, mfe = calculate_mae_mfe(df, HORIZON_HOURS)
    
    # Print MAE/MFE statistics
    print(f"\n📊 MAE/MFE Statistics ({HORIZON_HOURS}h horizon):")
    print(f"   MAE (drawdown): {mae.mean():.2f}% avg, {mae.quantile(0.05):.2f}% (5th), {mae.quantile(0.25):.2f}% (25th)")
    print(f"   MFE (max gain): {mfe.mean():.2f}% avg, {mfe.quantile(0.75):.2f}% (75th), {mfe.quantile(0.95):.2f}% (95th)")
    
    # Grid search for optimal SL/TP
    sl_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]
    tp_range = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
    
    results_df = simulate_sl_tp_grid(df, mae, mfe, sl_range, tp_range, HORIZON_HOURS)
    
    # Find top performers
    print(f"\n🏆 TOP 10 CONFIGURATIONS (by Expectancy):")
    print("=" * 70)
    
    top_10 = results_df.nlargest(10, 'expectancy')
    for idx, row in top_10.iterrows():
        print(f"\n   #{idx+1}: SL={row['sl_pct']:.1f}% / TP={row['tp_pct']:.1f}% (RR={row['reward_risk']:.2f})")
        print(f"      Win Rate: {row['win_rate']*100:.1f}% | Profit Factor: {row['profit_factor']:.2f}")
        print(f"      Expectancy: {row['expectancy']:.3f}% | Net Profit: {row['net_profit']:.1f}%")
        print(f"      Premature Stops: {row['premature_pct']*100:.1f}% | Trades: {row['total_trades']}")
    
    # Best by different criteria
    best_expectancy = results_df.nlargest(1, 'expectancy').iloc[0]
    best_profit_factor = results_df.nlargest(1, 'profit_factor').iloc[0]
    best_win_rate = results_df.nlargest(1, 'win_rate').iloc[0]
    
    print(f"\n🎯 RECOMMENDED CONFIGURATIONS:")
    print("=" * 70)
    print(f"\n💰 Best Expectancy (maximize long-term profit):")
    print(f"   SL: {best_expectancy['sl_pct']:.1f}% | TP: {best_expectancy['tp_pct']:.1f}%")
    print(f"   Expectancy: {best_expectancy['expectancy']:.3f}% | Win Rate: {best_expectancy['win_rate']*100:.1f}%")
    
    print(f"\n📊 Best Profit Factor (maximize profit/loss ratio):")
    print(f"   SL: {best_profit_factor['sl_pct']:.1f}% | TP: {best_profit_factor['tp_pct']:.1f}%")
    print(f"   Profit Factor: {best_profit_factor['profit_factor']:.2f} | Win Rate: {best_profit_factor['win_rate']*100:.1f}%")
    
    print(f"\n✅ Best Win Rate (maximize successful trades):")
    print(f"   SL: {best_win_rate['sl_pct']:.1f}% | TP: {best_win_rate['tp_pct']:.1f}%")
    print(f"   Win Rate: {best_win_rate['win_rate']*100:.1f}% | Expectancy: {best_win_rate['expectancy']:.3f}%")
    
    # Analyze by volatility
    vol_results = analyze_by_volatility(df, HORIZON_HOURS)
    
    print(f"\n🌡️  VOLATILITY-ADAPTIVE RECOMMENDATIONS:")
    print("=" * 70)
    for regime, data in vol_results.items():
        print(f"\n   {regime.upper()} Volatility (ATR ~{data['avg_atr']:.2f}%):")
        print(f"      Optimal: SL={data['best_sl']:.1f}% / TP={data['best_tp']:.1f}%")
        print(f"      Win Rate: {data['win_rate']*100:.1f}% | Expectancy: {data['expectancy']:.3f}%")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    results_df.to_csv(output_dir / 'sl_tp_optimization_results.csv', index=False)
    print(f"\n💾 Saved results to: {output_dir / 'sl_tp_optimization_results.csv'}")
    
    # Generate plots
    plot_results(results_df, output_dir)
    
    # Save recommended config
    config_update = f"""
# ==================== OPTIMIZED RISK MANAGEMENT ====================
# Based on 5 years of BTC historical data analysis

# RECOMMENDED (Best Expectancy):
STOP_LOSS_PCT={best_expectancy['sl_pct']}
TAKE_PROFIT_PCT={best_expectancy['tp_pct']}

# VOLATILITY-ADAPTIVE LEVELS:
# Low volatility:  SL={vol_results.get('low', {}).get('best_sl', 'N/A')}% / TP={vol_results.get('low', {}).get('best_tp', 'N/A')}%
# Normal volatility: SL={vol_results.get('normal', {}).get('best_sl', 'N/A')}% / TP={vol_results.get('normal', {}).get('best_tp', 'N/A')}%
# High volatility: SL={vol_results.get('high', {}).get('best_sl', 'N/A')}% / TP={vol_results.get('high', {}).get('best_tp', 'N/A')}%
"""
    
    with open(output_dir / 'recommended_config.txt', 'w') as f:
        f.write(config_update)
    
    print(f"\n✅ Optimization complete!")
    print(f"   View visualizations in: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
