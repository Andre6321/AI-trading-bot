"""
ML Strategy Backtesting with XGBoost Predictions.

This script:
1. Loads processed features and trained XGBoost model (xgb_h4.pkl)
2. Generates prediction probabilities for the test period
3. Creates trading signals combining ML predictions with trend filters
4. Runs vectorized backtest using backtest_signals
5. Evaluates performance and creates equity curve visualization

Signal Logic:
- signal = 1 if p_up > 0.55 AND MA20 > MA50 (trend filter)
- signal = 0 otherwise (flat position)

Usage: python scripts/backtest_ml_strategy.py
"""
import os
import sys
from pathlib import Path
import pickle
import warnings
from datetime import datetime

# Add src to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Our modules
from models.ml_dataset import build_ml_dataset
from backtest.backtester import backtest_signals, compare_strategies
from backtest.strategies import ma_trend_strategy


def load_trained_model(model_path: Path) -> dict:
    """
    Load trained ensemble model and metadata.
    
    Args:
        model_path: Path to the saved model pickle file
    
    Returns:
        Dictionary containing model and metadata
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load metadata
    metadata_path = model_path.with_name(model_path.stem + '_metadata.json')
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"📦 Loaded optimized ensemble model from {model_path.name}")
        print(f"   Training date: {metadata.get('training_date', 'N/A')}")
        print(f"   Features: {metadata.get('n_features', 'N/A')}")
        print(f"   Final ROC AUC: {metadata.get('final_roc_auc', 'N/A'):.4f}")
        
        return {
            'model': model,
            'feature_columns': metadata.get('features', []),
            'training_info': metadata,
            'evaluation_metrics': {'roc_auc': metadata.get('final_roc_auc', 0)}
        }
    else:
        # Fallback for old format
        print(f"📦 Loaded model from {model_path.name}")
        print(f"   Warning: Metadata file not found")
        
        return {
            'model': model,
            'feature_columns': [],
            'training_info': {'training_date': 'Unknown'},
            'evaluation_metrics': {'roc_auc': 0}
        }


def generate_ml_signals(df: pd.DataFrame, 
                       model_data: dict,
                       prob_threshold: float = 0.55,
                       use_trend_filter: bool = True) -> pd.Series:
    """
    Generate trading signals using ML predictions and trend filter.
    
    Args:
        df: Features DataFrame
        model_data: Loaded model dictionary
        prob_threshold: Minimum probability for positive signal
        use_trend_filter: Whether to apply MA trend filter
    
    Returns:
        Series with trading signals (0 or 1)
    """
    model = model_data['model']
    feature_columns = model_data['feature_columns']
    
    # Validate that we have the required features
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Extract features for prediction
    X = df[feature_columns].copy()
    
    # Handle any missing values (forward fill for time series)
    X = X.fillna(method='ffill').fillna(0)
    
    # Generate predictions
    print(f"🔮 Generating ML predictions...")
    print(f"   Probability threshold: {prob_threshold}")
    print(f"   Trend filter: {'Enabled' if use_trend_filter else 'Disabled'}")
    
    # Get prediction probabilities
    pred_proba = model.predict_proba(X)[:, 1]  # Probability of class 1 (up move)
    
    # Initialize signals
    signals = pd.Series(0, index=df.index)
    
    # ML signal condition: probability above threshold
    ml_condition = pred_proba > prob_threshold
    
    if use_trend_filter:
        # Trend filter: MA20 > MA50 (bullish trend)
        if 'ma_20' in df.columns and 'ma_50' in df.columns:
            trend_condition = df['ma_20'] > df['ma_50']
            # Combined condition: ML prediction AND trend filter
            final_condition = ml_condition & trend_condition
            print(f"   Using trend filter (MA20 > MA50)")
        else:
            print(f"   Warning: MA columns not found, skipping trend filter")
            final_condition = ml_condition
    else:
        final_condition = ml_condition
    
    # Set signals
    signals[final_condition] = 1
    
    # Signal statistics
    ml_signals_count = ml_condition.sum()
    final_signals_count = final_condition.sum()
    
    print(f"   ML signals (p > {prob_threshold}): {ml_signals_count:,} ({ml_signals_count/len(df)*100:.1f}%)")
    print(f"   Final signals (with filter): {final_signals_count:,} ({final_signals_count/len(df)*100:.1f}%)")
    
    if use_trend_filter and 'ma_20' in df.columns:
        trend_filter_reduction = (ml_signals_count - final_signals_count) / ml_signals_count * 100 if ml_signals_count > 0 else 0
        print(f"   Trend filter reduction: {trend_filter_reduction:.1f}%")
    
    return signals, pred_proba


def create_strategy_comparison_plot(results_dict: dict, output_path: Path):
    """
    Create comparison plot of multiple strategies.
    
    Args:
        results_dict: Dictionary of strategy results
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1])
    
    colors = ['navy', 'darkred', 'green', 'purple']
    
    for i, (strategy_name, results) in enumerate(results_dict.items()):
        equity_curve = results['equity_curve']
        positions_df = results['positions']
        
        # Plot equity curves
        if 'timestamp' in positions_df.columns:
            dates = pd.to_datetime(positions_df['timestamp'])
            ax1.plot(dates, equity_curve, linewidth=2, 
                    color=colors[i % len(colors)], label=strategy_name)
        else:
            ax1.plot(equity_curve, linewidth=2, 
                    color=colors[i % len(colors)], label=strategy_name)
    
    # Equity curve formatting
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Strategy Comparison - Equity Curves')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    if 'timestamp' in list(results_dict.values())[0]['positions'].columns:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Drawdown comparison for first strategy (ML strategy)
    ml_results = list(results_dict.values())[0]
    equity_curve = ml_results['equity_curve']
    positions_df = ml_results['positions']
    
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    
    if 'timestamp' in positions_df.columns:
        dates = pd.to_datetime(positions_df['timestamp'])
        ax2.fill_between(dates, drawdown, 0, alpha=0.7, color='red', label='ML Strategy Drawdown')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.7, color='red')
    
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.set_title('ML Strategy Drawdown')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Strategy comparison plot saved to: {output_path}")
    plt.close()


def create_prediction_analysis_plot(df: pd.DataFrame, pred_proba: np.ndarray, 
                                   signals: pd.Series, output_path: Path):
    """
    Create analysis plots for ML predictions.
    
    Args:
        df: Features DataFrame with price data
        pred_proba: Prediction probabilities
        signals: Trading signals
        output_path: Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Price with signals
    if 'timestamp' in df.columns:
        dates = pd.to_datetime(df['timestamp'])
        ax1.plot(dates, df['close'], linewidth=1, color='blue', alpha=0.7, label='Price')
        
        # Mark signal points
        signal_dates = dates[signals == 1]
        signal_prices = df['close'][signals == 1]
        ax1.scatter(signal_dates, signal_prices, color='red', s=10, alpha=0.7, label='ML Signals')
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax1.plot(df['close'], linewidth=1, color='blue', alpha=0.7, label='Price')
        ax1.scatter(signals[signals == 1].index, df['close'][signals == 1], 
                   color='red', s=10, alpha=0.7, label='ML Signals')
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Price Chart with ML Signals')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction probability distribution
    ax2.hist(pred_proba, bins=50, alpha=0.7, color='skyblue', density=True)
    ax2.axvline(x=0.55, color='red', linestyle='--', linewidth=2, label='Signal Threshold (0.55)')
    ax2.set_xlabel('Prediction Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('ML Prediction Probability Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction probability over time
    if 'timestamp' in df.columns:
        ax3.plot(dates, pred_proba, linewidth=1, color='green', alpha=0.8)
        ax3.axhline(y=0.55, color='red', linestyle='--', linewidth=1, label='Threshold')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax3.plot(pred_proba, linewidth=1, color='green', alpha=0.8)
        ax3.axhline(y=0.55, color='red', linestyle='--', linewidth=1, label='Threshold')
    
    ax3.set_ylabel('Prediction Probability')
    ax3.set_title('ML Predictions Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Signal frequency over time (monthly)
    if 'timestamp' in df.columns:
        df_with_signals = df.copy()
        df_with_signals['signals'] = signals
        df_with_signals['timestamp'] = pd.to_datetime(df_with_signals['timestamp'])
        df_with_signals.set_index('timestamp', inplace=True)
        
        monthly_signals = df_with_signals['signals'].resample('M').mean()
        ax4.plot(monthly_signals.index, monthly_signals.values * 100, marker='o', linewidth=2)
        ax4.set_ylabel('Signal Rate (%)')
        ax4.set_title('Monthly Signal Frequency')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    else:
        # Simple signal distribution if no timestamps
        signal_counts = signals.value_counts()
        ax4.bar(['No Signal', 'Signal'], [signal_counts.get(0, 0), signal_counts.get(1, 0)])
        ax4.set_ylabel('Count')
        ax4.set_title('Signal Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Prediction analysis plot saved to: {output_path}")
    plt.close()


def main():
    """Main ML strategy backtesting pipeline."""
    print("🤖 ML Strategy Backtesting Pipeline")
    print("=" * 50)
    
    # Define paths
    features_path = script_dir.parent / "data" / "processed" / "btcusdt_1h_full_history_enhanced.parquet"
    model_path = script_dir.parent / "models" / "ensemble_btcusdt_h4_optuna_optimized.pkl"
    outputs_dir = script_dir.parent / "outputs"
    
    # Ensure output directory exists
    outputs_dir.mkdir(exist_ok=True)
    
    # Check if required files exist
    if not features_path.exists():
        print(f"❌ Features file not found: {features_path}")
        print("   Please run 'python scripts/build_dataset.py' first.")
        return 1
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("   Please run 'python scripts/train_xgboost_model.py' first.")
        return 1
    
    try:
        # Load features
        print(f"📥 Loading features...")
        df = pd.read_parquet(features_path)
        print(f"   Loaded {len(df):,} rows with {len(df.columns)} columns")
        
        # Load trained model
        model_data = load_trained_model(model_path)
        
        # Build ML dataset to get the same splits as training
        print(f"\n🎯 Preparing ML dataset...")
        X, y, aligned_index = build_ml_dataset(
            df=df,
            horizon_hours=4,  # Same as training
            threshold=0.01,   # Same as training
            validate_features=False  # Skip validation for speed
        )
        
        # Get test period (same split as training)
        print(f"\n📊 Creating test period split...")
        n_samples = len(X)
        test_start = int(n_samples * 0.85)  # Last 15% (same as training)
        
        # Extract test period data
        test_indices = range(test_start, n_samples)
        df_test = df.iloc[test_indices].copy().reset_index()
        
        # Ensure timestamp column exists
        if 'timestamp' not in df_test.columns:
            if df_test.index.name == 'timestamp' or isinstance(df_test.index, pd.DatetimeIndex):
                df_test['timestamp'] = df_test.index
            else:
                # Generate sequential timestamps if not available
                df_test['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df_test), freq='1H')
        
        print(f"   Test period: {len(df_test):,} samples")
        if 'timestamp' in df_test.columns:
            print(f"   Date range: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")
        
        # Generate ML signals for test period
        print(f"\n🔮 Generating ML trading signals...")
        ml_signals, pred_proba = generate_ml_signals(
            df_test, 
            model_data,
            prob_threshold=0.55,
            use_trend_filter=True
        )
        
        # Add signals to test dataframe
        df_test_with_signals = df_test.copy()
        df_test_with_signals['ml_signals'] = ml_signals
        df_test_with_signals['pred_proba'] = pred_proba
        
        # Generate baseline MA signals for comparison
        print(f"\n📈 Generating baseline MA signals for comparison...")
        ma_signals = ma_trend_strategy(df_test)
        df_test_with_signals['ma_signals'] = ma_signals
        
        # Run backtests
        print(f"\n🔄 Running strategy backtests...")
        
        # ML Strategy backtest
        ml_results = backtest_signals(
            df=df_test_with_signals,
            signal_col='ml_signals',
            fee_rate=0.0004,
            initial_capital=10000,
            execution_price='open',
            freq='1H'
        )
        
        # MA Strategy backtest (for comparison)
        ma_results = backtest_signals(
            df=df_test_with_signals,
            signal_col='ma_signals',
            fee_rate=0.0004,
            initial_capital=10000,
            execution_price='open',
            freq='1H'
        )
        
        # Print performance comparison
        print(f"\n📊 STRATEGY PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        
        strategies = {
            'ML Strategy (XGBoost + Trend)': ml_results,
            'MA Trend Strategy': ma_results
        }
        
        comparison_df = compare_strategies(strategies)
        print(f"\n{comparison_df.to_string(index=False)}")
        
        # Detailed ML strategy results
        print(f"\n🤖 ML STRATEGY DETAILED RESULTS")
        print(f"{'='*60}")
        
        ml_summary = ml_results['summary']
        ml_stats = ml_results['stats']
        
        print(f"\n💰 PERFORMANCE:")
        print(f"   Total Return:     {ml_summary['total_return_pct']:>8.2f}%")
        print(f"   CAGR:             {ml_summary['cagr_pct']:>8.2f}%")
        print(f"   Sharpe Ratio:     {ml_summary['sharpe_ratio']:>8.2f}")
        print(f"   Max Drawdown:     {ml_summary['max_drawdown_pct']:>8.2f}%")
        print(f"   Win Rate:         {ml_summary['win_rate_pct']:>8.2f}%")
        print(f"   Total Trades:     {ml_summary['total_trades']:>8.0f}")
        
        print(f"\n🔮 ML SIGNALS:")
        print(f"   Signal Rate:      {ml_signals.mean()*100:>8.1f}%")
        print(f"   Avg Probability:  {pred_proba.mean():>8.3f}")
        print(f"   High Conf Rate:   {(pred_proba > 0.7).mean()*100:>8.1f}%")
        
        # Create visualizations
        print(f"\n📊 Creating visualizations...")
        
        # Strategy comparison plot
        comparison_plot_path = outputs_dir / "ml_strategy_comparison.png"
        create_strategy_comparison_plot(strategies, comparison_plot_path)
        
        # ML prediction analysis plot
        analysis_plot_path = outputs_dir / "ml_prediction_analysis.png"
        create_prediction_analysis_plot(df_test, pred_proba, ml_signals, analysis_plot_path)
        
        print(f"\n✅ ML strategy backtesting completed!")
        print(f"   ML Strategy Return: {ml_summary['total_return_pct']:.2f}%")
        print(f"   MA Strategy Return: {ma_results['summary']['total_return_pct']:.2f}%")
        print(f"   Performance Diff:   {ml_summary['total_return_pct'] - ma_results['summary']['total_return_pct']:+.2f}%")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during ML backtesting: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    exit_code = main()
    sys.exit(exit_code)
