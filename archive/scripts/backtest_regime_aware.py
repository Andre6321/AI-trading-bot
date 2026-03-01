"""
Backtest regime-aware trading strategy vs unified model.

Compares performance of:
1. Unified model (single model for all conditions)
2. Regime-aware strategy (separate models per regime)

Expected improvements: +10-15% win rate, +20-30% profit factor
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class RegimeAwareBacktester:
    """Backtest regime-aware vs unified strategy."""
    
    def __init__(
        self,
        data_path: str = 'data/processed/btcusdt_1h_with_regimes.parquet',
        unified_model_path: str = 'models/ensemble_btcusdt_h4_optuna_optimized.pkl',
        regime_models_dir: str = 'models'
    ):
        """Initialize backtester."""
        self.data_path = Path(data_path)
        self.unified_model_path = Path(unified_model_path)
        self.regime_models_dir = Path(regime_models_dir)
        
        # Load data
        print("📥 Loading data...")
        self.df = pd.read_parquet(self.data_path)
        print(f"   Loaded {len(self.df):,} samples")
        
        # Load models
        print("\n📦 Loading models...")
        self.unified_model = joblib.load(self.unified_model_path)
        print("   ✅ Unified model loaded")
        
        self.regime_models = {}
        for regime in ['bear', 'sideways', 'bull']:
            model_path = self.regime_models_dir / f'ensemble_{regime}_regime.pkl'
            if model_path.exists():
                self.regime_models[regime] = joblib.load(model_path)
                print(f"   ✅ {regime.capitalize()} regime model loaded")
        
        # Prepare features - exclude non-feature columns
        exclude_cols = ['target_4h', 'future_ret_4h', 'regime', 'regime_name', 
                       'timestamp', 'date', 'time', 'open', 'high', 'low', 'close', 'volume']
        self.feature_cols = [col for col in self.df.columns 
                            if col not in exclude_cols and not col.startswith('Unnamed')]
    
    def calculate_sl_tp(
        self,
        entry_price: float,
        side: str,
        atr_pct: float,
        market_regime: str = None
    ) -> Tuple[float, float]:
        """Calculate regime-aware SL/TP levels."""
        # Base volatility-adaptive levels
        if atr_pct < 0.0025:
            sl_pct = 0.05
            tp_pct = 0.02
        elif atr_pct < 0.005:
            sl_pct = 0.05
            tp_pct = 0.03
        else:
            sl_pct = 0.08
            tp_pct = 0.04
        
        # Market regime adjustments
        if market_regime == 'bull':
            tp_pct = tp_pct * 1.5  # Let winners run
            sl_pct = sl_pct * 0.9  # Tighter stops
        elif market_regime == 'bear':
            tp_pct = tp_pct * 0.7  # Take profits fast
            sl_pct = sl_pct * 1.2  # Wider stops
        elif market_regime == 'sideways':
            tp_pct = tp_pct * 0.85  # Mean reversion
            sl_pct = sl_pct * 0.85
        
        # Calculate prices
        if side == 'long':
            sl = entry_price * (1 - sl_pct)
            tp = entry_price * (1 + tp_pct)
        else:
            sl = entry_price * (1 + sl_pct)
            tp = entry_price * (1 - tp_pct)
        
        return sl, tp
    
    def simulate_trade(
        self,
        entry_idx: int,
        entry_price: float,
        side: str,
        sl: float,
        tp: float,
        max_hold_hours: int = 24
    ) -> Dict:
        """Simulate a single trade with SL/TP."""
        # Get future price data
        future_data = self.df.iloc[entry_idx:entry_idx+max_hold_hours+1]
        
        if len(future_data) <= 1:
            return {'outcome': 'expired', 'pnl': 0, 'hold_hours': 0}
        
        for i, (idx, row) in enumerate(future_data.iterrows()):
            if i == 0:
                continue  # Skip entry candle
            
            high = row['high']
            low = row['low']
            
            if side == 'long':
                # Check SL hit
                if low <= sl:
                    pnl = (sl / entry_price - 1) * 100
                    return {'outcome': 'sl', 'pnl': pnl, 'hold_hours': i, 'exit_price': sl}
                # Check TP hit
                if high >= tp:
                    pnl = (tp / entry_price - 1) * 100
                    return {'outcome': 'tp', 'pnl': pnl, 'hold_hours': i, 'exit_price': tp}
            else:  # short
                # Check SL hit
                if high >= sl:
                    pnl = (1 - sl / entry_price) * 100
                    return {'outcome': 'sl', 'pnl': pnl, 'hold_hours': i, 'exit_price': sl}
                # Check TP hit
                if low <= tp:
                    pnl = (1 - tp / entry_price) * 100
                    return {'outcome': 'tp', 'pnl': pnl, 'hold_hours': i, 'exit_price': tp}
        
        # Exit at market after max hold
        exit_price = future_data.iloc[-1]['close']
        if side == 'long':
            pnl = (exit_price / entry_price - 1) * 100
        else:
            pnl = (1 - exit_price / entry_price) * 100
        
        return {'outcome': 'market', 'pnl': pnl, 'hold_hours': max_hold_hours, 'exit_price': exit_price}
    
    def backtest_unified(
        self,
        confidence_threshold: float = 0.6,
        start_idx: int = 0,
        end_idx: int = None
    ) -> Tuple[List[Dict], Dict]:
        """Backtest unified model strategy."""
        print("\n" + "="*70)
        print("🔹 BACKTESTING UNIFIED MODEL")
        print("="*70)
        
        if end_idx is None:
            end_idx = len(self.df)
        
        trades = []
        test_data = self.df.iloc[start_idx:end_idx]
        
        for i, (idx, row) in enumerate(test_data.iterrows()):
            if i % 1000 == 0:
                print(f"   Processing: {i:,}/{len(test_data):,} ({i/len(test_data)*100:.1f}%)")
            
            # Get features
            features = row[self.feature_cols].values.reshape(1, -1)
            
            # Predict
            try:
                pred_proba = self.unified_model.predict_proba(features)[0]
                pred = self.unified_model.predict(features)[0]
                confidence = pred_proba[pred]
            except:
                continue
            
            # Check confidence
            if confidence < confidence_threshold:
                continue
            
            # Only take long positions
            if pred == 0:  # Down prediction
                continue
            
            # Calculate SL/TP (baseline: no regime adjustment)
            entry_price = row['close']
            atr_pct = row.get('atr_pct', 0.005)
            sl, tp = self.calculate_sl_tp(entry_price, 'long', atr_pct, market_regime=None)
            
            # Simulate trade
            result = self.simulate_trade(
                entry_idx=start_idx + i,
                entry_price=entry_price,
                side='long',
                sl=sl,
                tp=tp
            )
            
            trades.append({
                'entry_idx': start_idx + i,
                'entry_price': entry_price,
                'confidence': confidence,
                'regime': row.get('regime_name', 'unknown'),
                **result
            })
        
        # Calculate statistics
        stats = self._calculate_statistics(trades, "Unified Model")
        
        return trades, stats
    
    def backtest_regime_aware(
        self,
        confidence_threshold: float = 0.6,
        start_idx: int = 0,
        end_idx: int = None
    ) -> Tuple[List[Dict], Dict]:
        """Backtest regime-aware strategy."""
        print("\n" + "="*70)
        print("🎯 BACKTESTING REGIME-AWARE STRATEGY")
        print("="*70)
        
        if end_idx is None:
            end_idx = len(self.df)
        
        trades = []
        test_data = self.df.iloc[start_idx:end_idx]
        
        for i, (idx, row) in enumerate(test_data.iterrows()):
            if i % 1000 == 0:
                print(f"   Processing: {i:,}/{len(test_data):,} ({i/len(test_data)*100:.1f}%)")
            
            # Get regime
            regime = row.get('regime_name', 'sideways').lower()
            
            # Get appropriate model
            model = self.regime_models.get(regime)
            if model is None:
                continue
            
            # Get features
            features = row[self.feature_cols].values.reshape(1, -1)
            
            # Predict
            try:
                pred_proba = model.predict_proba(features)[0]
                pred = model.predict(features)[0]
                confidence = pred_proba[pred]
            except:
                continue
            
            # Check confidence
            if confidence < confidence_threshold:
                continue
            
            # Only take long positions
            if pred == 0:  # Down prediction
                continue
            
            # Calculate regime-aware SL/TP
            entry_price = row['close']
            atr_pct = row.get('atr_pct', 0.005)
            sl, tp = self.calculate_sl_tp(entry_price, 'long', atr_pct, market_regime=regime)
            
            # Simulate trade
            result = self.simulate_trade(
                entry_idx=start_idx + i,
                entry_price=entry_price,
                side='long',
                sl=sl,
                tp=tp
            )
            
            trades.append({
                'entry_idx': start_idx + i,
                'entry_price': entry_price,
                'confidence': confidence,
                'regime': regime,
                'model': f'{regime}_specific',
                **result
            })
        
        # Calculate statistics
        stats = self._calculate_statistics(trades, "Regime-Aware")
        
        return trades, stats
    
    def _calculate_statistics(self, trades: List[Dict], strategy_name: str) -> Dict:
        """Calculate trading statistics."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'profit_factor': 0
            }
        
        df_trades = pd.DataFrame(trades)
        
        # Basic stats
        total_trades = len(trades)
        winners = df_trades[df_trades['pnl'] > 0]
        losers = df_trades[df_trades['pnl'] < 0]
        
        win_rate = len(winners) / total_trades * 100
        avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
        
        total_profit = winners['pnl'].sum() if len(winners) > 0 else 0
        total_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # Outcome breakdown
        tp_count = (df_trades['outcome'] == 'tp').sum()
        sl_count = (df_trades['outcome'] == 'sl').sum()
        market_count = (df_trades['outcome'] == 'market').sum()
        
        # Regime breakdown
        regime_stats = {}
        for regime in ['bear', 'sideways', 'bull']:
            regime_trades = df_trades[df_trades['regime'] == regime]
            if len(regime_trades) > 0:
                regime_stats[regime] = {
                    'count': len(regime_trades),
                    'win_rate': (regime_trades['pnl'] > 0).mean() * 100,
                    'avg_pnl': regime_trades['pnl'].mean()
                }
        
        stats = {
            'strategy': strategy_name,
            'total_trades': total_trades,
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': df_trades['pnl'].sum(),
            'avg_pnl': df_trades['pnl'].mean(),
            'tp_count': tp_count,
            'sl_count': sl_count,
            'market_count': market_count,
            'avg_hold_hours': df_trades['hold_hours'].mean(),
            'regime_breakdown': regime_stats
        }
        
        # Print results
        print(f"\n📊 {strategy_name} Results:")
        print(f"   Total Trades: {total_trades:,}")
        print(f"   Win Rate: {win_rate:.2f}%")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print(f"   Avg PnL: {stats['avg_pnl']:.2f}%")
        print(f"   Total PnL: {stats['total_pnl']:.2f}%")
        print(f"\n   Outcomes:")
        print(f"     TP hits: {tp_count} ({tp_count/total_trades*100:.1f}%)")
        print(f"     SL hits: {sl_count} ({sl_count/total_trades*100:.1f}%)")
        print(f"     Market exits: {market_count} ({market_count/total_trades*100:.1f}%)")
        print(f"     Avg hold: {stats['avg_hold_hours']:.1f} hours")
        
        print(f"\n   Regime Breakdown:")
        for regime, data in regime_stats.items():
            print(f"     {regime.upper()}: {data['count']} trades, "
                  f"{data['win_rate']:.1f}% WR, {data['avg_pnl']:+.2f}% avg")
        
        return stats
    
    def compare_strategies(self, unified_stats: Dict, regime_stats: Dict):
        """Compare unified vs regime-aware performance."""
        print("\n" + "="*70)
        print("📈 STRATEGY COMPARISON")
        print("="*70)
        
        print(f"\n{'Metric':<25} {'Unified':<15} {'Regime-Aware':<15} {'Improvement':<15}")
        print("-" * 70)
        
        metrics = [
            ('Total Trades', 'total_trades', 'd'),
            ('Win Rate %', 'win_rate', '.2f'),
            ('Profit Factor', 'profit_factor', '.2f'),
            ('Avg PnL %', 'avg_pnl', '.2f'),
            ('Total PnL %', 'total_pnl', '.2f'),
            ('TP Hit Rate %', 'tp_rate', '.1f'),
            ('SL Hit Rate %', 'sl_rate', '.1f'),
        ]
        
        # Calculate rates
        unified_stats['tp_rate'] = unified_stats['tp_count'] / unified_stats['total_trades'] * 100 if unified_stats['total_trades'] > 0 else 0
        unified_stats['sl_rate'] = unified_stats['sl_count'] / unified_stats['total_trades'] * 100 if unified_stats['total_trades'] > 0 else 0
        regime_stats['tp_rate'] = regime_stats['tp_count'] / regime_stats['total_trades'] * 100 if regime_stats['total_trades'] > 0 else 0
        regime_stats['sl_rate'] = regime_stats['sl_count'] / regime_stats['total_trades'] * 100 if regime_stats['total_trades'] > 0 else 0
        
        for label, key, fmt in metrics:
            unified_val = unified_stats.get(key, 0)
            regime_val = regime_stats.get(key, 0)
            
            if unified_val > 0:
                if key == 'sl_rate':  # Lower is better for SL
                    improvement = ((unified_val - regime_val) / unified_val * 100)
                    improvement_str = f"{improvement:+.1f}%"
                else:
                    improvement = ((regime_val - unified_val) / unified_val * 100)
                    improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
            
            print(f"{label:<25} {unified_val:<15{fmt}} {regime_val:<15{fmt}} {improvement_str:<15}")
        
        print("\n" + "="*70)
        print("🎯 KEY INSIGHTS")
        print("="*70)
        
        wr_improvement = regime_stats['win_rate'] - unified_stats['win_rate']
        pf_improvement = (regime_stats['profit_factor'] / unified_stats['profit_factor'] - 1) * 100 if unified_stats['profit_factor'] > 0 else 0
        
        print(f"\n✅ Win Rate Improvement: {wr_improvement:+.2f}%")
        print(f"✅ Profit Factor Improvement: {pf_improvement:+.2f}%")
        print(f"✅ Total PnL Improvement: {regime_stats['total_pnl'] - unified_stats['total_pnl']:+.2f}%")
        
        if wr_improvement > 5:
            print("\n🎉 SIGNIFICANT IMPROVEMENT from regime-aware strategy!")
        elif wr_improvement > 0:
            print("\n👍 Positive improvement from regime-aware strategy")
        else:
            print("\n⚠️ Unified model performing better - check regime models")


def create_stunning_visualizations(unified_trades, regime_trades, unified_stats, regime_stats):
    """Create impressive visualizations for presentation."""
    output_dir = Path('outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('dark_background')
    sns.set_palette("husl")
    
    # 1. CUMULATIVE RETURNS COMPARISON (MAIN SHOWCASE)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    
    unified_df = pd.DataFrame(unified_trades)
    regime_df = pd.DataFrame(regime_trades)
    
    if len(unified_df) > 0 and len(regime_df) > 0:
        unified_df['cumulative_pnl'] = unified_df['pnl_pct'].cumsum()
        regime_df['cumulative_pnl'] = regime_df['pnl_pct'].cumsum()
        
        ax.plot(unified_df.index, unified_df['cumulative_pnl'], 
                label='Standard Model', linewidth=2.5, color='#FF6B6B', alpha=0.8)
        ax.plot(regime_df.index, regime_df['cumulative_pnl'], 
                label='AI Regime-Aware Model', linewidth=3, color='#4ECDC4')
        
        ax.fill_between(unified_df.index, unified_df['cumulative_pnl'], 
                        alpha=0.2, color='#FF6B6B')
        ax.fill_between(regime_df.index, regime_df['cumulative_pnl'], 
                        alpha=0.3, color='#4ECDC4')
        
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_xlabel('Trade Number', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)', fontsize=14, fontweight='bold')
        ax.set_title('🚀 AI-Powered Trading Performance Comparison', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.2, linestyle='--')
        
        # Add performance metrics
        final_regime = regime_df['cumulative_pnl'].iloc[-1]
        final_unified = unified_df['cumulative_pnl'].iloc[-1]
        improvement = final_regime - final_unified
        
        textstr = f'AI Advantage: +{improvement:.2f}%\n'
        textstr += f'Win Rate Boost: +{regime_stats["win_rate"] - unified_stats["win_rate"]:.1f}%\n'
        textstr += f'Profit Factor: {regime_stats["profit_factor"]:.2f}x'
        
        props = dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='#4ECDC4', linewidth=2)
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=13,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=props, color='#4ECDC4', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_showcase.png', dpi=100, bbox_inches='tight', 
                   facecolor='#1a1a1a')
        plt.close()
    
    # 2. PERFORMANCE METRICS DASHBOARD
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=100)
    fig.patch.set_facecolor('#1a1a1a')
    
    metrics = [
        ('Win Rate', unified_stats['win_rate'], regime_stats['win_rate'], '%'),
        ('Profit Factor', unified_stats['profit_factor'], regime_stats['profit_factor'], 'x'),
        ('Total Return', unified_stats['total_pnl'], regime_stats['total_pnl'], '%'),
        ('Avg Win', unified_stats['avg_win'], regime_stats['avg_win'], '%'),
        ('Max Win', unified_stats['max_win'], regime_stats['max_win'], '%'),
        ('Sharpe Ratio', unified_stats['sharpe_ratio'], regime_stats['sharpe_ratio'], '')
    ]
    
    for idx, (ax, (name, unified_val, regime_val, unit)) in enumerate(zip(axes.flat, metrics)):
        x = ['Standard\nModel', 'AI Regime-\nAware']
        y = [unified_val, regime_val]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(x, y, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add value labels on bars
        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}{unit}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold', color='white')
        
        ax.set_title(name, fontsize=14, fontweight='bold', pad=10)
        ax.set_facecolor('#2a2a2a')
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        ax.tick_params(labelsize=10)
        
        # Highlight improvement
        if regime_val > unified_val:
            improvement_pct = ((regime_val - unified_val) / unified_val * 100) if unified_val != 0 else 0
            ax.text(0.5, 0.95, f'↑ +{improvement_pct:.1f}%', 
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=11, color='#4ECDC4', fontweight='bold')
    
    plt.suptitle('📊 AI Model Performance Dashboard', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_dashboard.png', dpi=100, bbox_inches='tight',
               facecolor='#1a1a1a')
    plt.close()
    
    # 3. REGIME-SPECIFIC PERFORMANCE
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    fig.patch.set_facecolor('#1a1a1a')
    
    if 'regime' in regime_df.columns:
        regime_performance = regime_df.groupby('regime')['pnl_pct'].agg(['mean', 'count', 'sum'])
        regime_performance = regime_performance.sort_values('sum', ascending=False)
        
        x = np.arange(len(regime_performance))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, regime_performance['mean'], width, 
                      label='Avg Return per Trade', color='#4ECDC4', alpha=0.8, edgecolor='white', linewidth=2)
        bars2 = ax.bar(x + width/2, regime_performance['sum']/10, width,
                      label='Total Return (÷10)', color='#95E1D3', alpha=0.8, edgecolor='white', linewidth=2)
        
        ax.set_xlabel('Market Regime', fontsize=13, fontweight='bold')
        ax.set_ylabel('Return (%)', fontsize=13, fontweight='bold')
        ax.set_title('🎯 AI Performance Across Market Conditions', fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{reg.capitalize()}\n({int(regime_performance.loc[reg, "count"])} trades)' 
                           for reg in regime_performance.index], fontsize=11)
        ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        ax.set_facecolor('#2a2a2a')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%', ha='center', va='bottom',
                       fontsize=10, fontweight='bold', color='white')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'regime_performance.png', dpi=100, bbox_inches='tight',
                   facecolor='#1a1a1a')
        plt.close()
    
    # 4. WIN/LOSS DISTRIBUTION
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=100)
    fig.patch.set_facecolor('#1a1a1a')
    
    # Unified model
    wins_u = unified_df[unified_df['pnl_pct'] > 0]['pnl_pct']
    losses_u = unified_df[unified_df['pnl_pct'] <= 0]['pnl_pct']
    
    ax1.hist(wins_u, bins=30, color='#4ECDC4', alpha=0.7, label='Wins', edgecolor='white')
    ax1.hist(losses_u, bins=30, color='#FF6B6B', alpha=0.7, label='Losses', edgecolor='white')
    ax1.axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.5)
    ax1.set_title('Standard Model - Trade Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Return (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.2, linestyle='--')
    ax1.set_facecolor('#2a2a2a')
    
    # Regime-aware model
    wins_r = regime_df[regime_df['pnl_pct'] > 0]['pnl_pct']
    losses_r = regime_df[regime_df['pnl_pct'] <= 0]['pnl_pct']
    
    ax2.hist(wins_r, bins=30, color='#4ECDC4', alpha=0.7, label='Wins', edgecolor='white')
    ax2.hist(losses_r, bins=30, color='#FF6B6B', alpha=0.7, label='Losses', edgecolor='white')
    ax2.axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_title('AI Regime-Aware - Trade Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Return (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.2, linestyle='--')
    ax2.set_facecolor('#2a2a2a')
    
    plt.suptitle('📈 Trade Distribution Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'trade_distribution.png', dpi=100, bbox_inches='tight',
               facecolor='#1a1a1a')
    plt.close()
    
    print(f"   ✅ Visualizations saved to {output_dir}/")


def main():
    """Run backtest comparison."""
    print("="*70)
    print("🚀 REGIME-AWARE STRATEGY BACKTEST")
    print("="*70)
    
    # Initialize backtester
    backtester = RegimeAwareBacktester()
    
    # Split data: use last 30% for testing
    total_samples = len(backtester.df)
    train_size = int(total_samples * 0.7)
    
    print(f"\nTest period: Last {total_samples - train_size:,} samples ({(total_samples - train_size)/24:.0f} days)")
    
    # Backtest unified
    unified_trades, unified_stats = backtester.backtest_unified(
        confidence_threshold=0.6,
        start_idx=train_size
    )
    
    # Backtest regime-aware
    regime_trades, regime_stats = backtester.backtest_regime_aware(
        confidence_threshold=0.6,
        start_idx=train_size
    )
    
    # Compare
    backtester.compare_strategies(unified_stats, regime_stats)
    
    # Save results
    print("\n💾 Saving results...")
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    pd.DataFrame(unified_trades).to_csv(output_dir / 'backtest_unified_trades.csv', index=False)
    pd.DataFrame(regime_trades).to_csv(output_dir / 'backtest_regime_trades.csv', index=False)
    
    with open(output_dir / 'backtest_comparison.json', 'w') as f:
        json.dump({
            'unified': unified_stats,
            'regime_aware': regime_stats
        }, f, indent=2)
    
    print("   ✅ Results saved to outputs/")
    
    # CREATE STUNNING VISUALIZATIONS
    print("\n🎨 Creating impressive visualizations...")
    create_stunning_visualizations(unified_trades, regime_trades, unified_stats, regime_stats)
    
    print("\n" + "="*70)
    print("✨ PRESENTATION-READY VISUALIZATIONS CREATED!")
    print("="*70)
    print("\n📊 Check outputs/visualizations/ for:")
    print("   • performance_showcase.png - Main comparison chart")
    print("   • metrics_dashboard.png - Key metrics overview")
    print("   • regime_performance.png - Performance by market condition")
    print("   • trade_distribution.png - Win/loss analysis")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
