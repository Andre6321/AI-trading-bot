"""
Market Regime Detection using Hidden Markov Model (HMM).

Classifies market into three regimes:
- Bull Market: Strong uptrend, high momentum
- Bear Market: Strong downtrend, high volatility
- Sideways/Consolidation: Range-bound, low momentum

This allows training separate models for each regime to improve predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate features for regime detection.
    
    Features:
    - Returns (various timeframes)
    - Volatility (rolling std)
    - Trend strength (ADX-like)
    - Volume changes
    - Price momentum
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with regime features
    """
    features = pd.DataFrame(index=df.index)
    
    # Returns at different timeframes
    features['ret_1h'] = df['close'].pct_change(1)
    features['ret_4h'] = df['close'].pct_change(4)
    features['ret_24h'] = df['close'].pct_change(24)
    features['ret_7d'] = df['close'].pct_change(168)  # 7 days
    
    # Volatility (rolling standard deviation of returns)
    features['vol_24h'] = features['ret_1h'].rolling(24).std()
    features['vol_7d'] = features['ret_1h'].rolling(168).std()
    
    # Trend strength: difference between fast and slow MA
    ma_fast = df['close'].rolling(20).mean()
    ma_slow = df['close'].rolling(50).mean()
    features['ma_diff'] = (ma_fast - ma_slow) / ma_slow
    
    # Momentum
    features['rsi'] = calculate_rsi(df['close'], 14)
    features['momentum_24h'] = df['close'] / df['close'].shift(24) - 1
    
    # Volume changes
    if 'volume' in df.columns:
        features['vol_change'] = df['volume'].pct_change(24)
        features['vol_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Price range (high-low spread)
    if 'high' in df.columns and 'low' in df.columns:
        features['hl_range'] = (df['high'] - df['low']) / df['close']
    
    # Trend direction: percentage above/below MA
    features['price_vs_ma50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).mean()
    
    return features.dropna()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def train_regime_hmm(
    features: pd.DataFrame,
    n_states: int = 3,
    n_iter: int = 1000
) -> Tuple[hmm.GaussianHMM, StandardScaler]:
    """
    Train Hidden Markov Model for regime detection.
    
    Args:
        features: Feature DataFrame
        n_states: Number of market regimes (default 3: bull/bear/sideways)
        n_iter: Number of EM iterations
        
    Returns:
        Trained HMM model and scaler
    """
    print(f"\n🔬 Training HMM with {n_states} market regimes...")
    
    # Replace infinity and NaN values
    features_clean = features.replace([np.inf, -np.inf], np.nan)
    features_clean = features_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_clean)
    
    # Train HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42
    )
    
    model.fit(features_scaled)
    
    print(f"   Training complete!")
    print(f"   Converged: {model.monitor_.converged}")
    print(f"   Log likelihood: {model.score(features_scaled):.2f}")
    
    return model, scaler


def predict_regimes(
    model: hmm.GaussianHMM,
    scaler: StandardScaler,
    features: pd.DataFrame
) -> np.ndarray:
    """
    Predict market regimes using trained HMM.
    
    Args:
        model: Trained HMM model
        scaler: Fitted scaler
        features: Feature DataFrame
        
    Returns:
        Array of regime predictions
    """
    # Replace infinity and NaN values
    features_clean = features.replace([np.inf, -np.inf], np.nan)
    features_clean = features_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    features_scaled = scaler.transform(features_clean)
    regimes = model.predict(features_scaled)
    return regimes


def label_regimes(
    regimes: np.ndarray,
    features: pd.DataFrame
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Label regimes as Bull/Bear/Sideways based on characteristics.
    
    Args:
        regimes: Array of regime labels from HMM
        features: Feature DataFrame
        
    Returns:
        Relabeled regimes and regime name mapping
    """
    # Calculate mean characteristics for each regime
    regime_stats = {}
    for regime in range(3):
        mask = regimes == regime
        stats = {
            'mean_return': features.loc[mask, 'ret_24h'].mean(),
            'mean_vol': features.loc[mask, 'vol_24h'].mean(),
            'mean_momentum': features.loc[mask, 'momentum_24h'].mean(),
            'count': mask.sum()
        }
        regime_stats[regime] = stats
    
    # Sort regimes by mean return
    sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1]['mean_return'])
    
    # Map regimes: lowest return = bear, middle = sideways, highest = bull
    regime_mapping = {
        sorted_regimes[0][0]: 0,  # Bear
        sorted_regimes[1][0]: 1,  # Sideways
        sorted_regimes[2][0]: 2   # Bull
    }
    
    regime_names = {
        0: "BEAR",
        1: "SIDEWAYS",
        2: "BULL"
    }
    
    # Relabel
    relabeled = np.array([regime_mapping[r] for r in regimes])
    
    # Print regime statistics
    print(f"\n📊 REGIME STATISTICS:")
    for regime_id, name in regime_names.items():
        original_id = [k for k, v in regime_mapping.items() if v == regime_id][0]
        stats = regime_stats[original_id]
        print(f"\n   {name} Market:")
        print(f"      Samples: {stats['count']:,} ({stats['count']/len(regimes)*100:.1f}%)")
        print(f"      Avg 24h Return: {stats['mean_return']*100:+.2f}%")
        print(f"      Avg Volatility: {stats['mean_vol']*100:.2f}%")
        print(f"      Avg Momentum: {stats['mean_momentum']*100:+.2f}%")
    
    return relabeled, regime_names


def plot_regime_timeline(
    df: pd.DataFrame,
    regimes: np.ndarray,
    regime_names: Dict[int, str],
    output_path: Path
):
    """
    Visualize market regimes over time.
    
    Args:
        df: DataFrame with price data
        regimes: Array of regime labels
        regime_names: Mapping of regime IDs to names
        output_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, height_ratios=[3, 1])
    
    # Plot price with regime coloring
    regime_colors = {0: 'red', 1: 'gray', 2: 'green'}
    
    # Create segments for each regime
    for i in range(len(df)):
        if i > 0:
            regime = regimes[i]
            color = regime_colors[regime]
            ax1.plot(
                df.index[i-1:i+1],
                df['close'].iloc[i-1:i+1],
                color=color,
                linewidth=1.5,
                alpha=0.7
            )
    
    ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax1.set_title('BTC Price with Market Regime Detection', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=regime_colors[0], label='BEAR Market'),
        Patch(facecolor=regime_colors[1], label='SIDEWAYS Market'),
        Patch(facecolor=regime_colors[2], label='BULL Market')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Plot regime timeline
    regime_timeline = pd.Series(regimes, index=df.index)
    colors_mapped = regime_timeline.map(regime_colors)
    
    for i in range(len(regime_timeline)):
        ax2.axvspan(
            regime_timeline.index[i],
            regime_timeline.index[min(i+1, len(regime_timeline)-1)],
            color=colors_mapped.iloc[i],
            alpha=0.7
        )
    
    ax2.set_ylabel('Market Regime', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['BEAR', 'SIDEWAYS', 'BULL'])
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Regime timeline plot saved to: {output_path}")
    plt.close()


def main():
    """Main regime detection pipeline."""
    print("=" * 70)
    print("🎯 MARKET REGIME DETECTION WITH HMM")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading historical data...")
    data_path = Path("data/processed/btcusdt_1h_full_history_enhanced.parquet")
    df = pd.read_parquet(data_path)
    print(f"   Loaded {len(df):,} samples")
    
    # Calculate regime features
    print("\n🔧 Calculating regime detection features...")
    regime_features = calculate_regime_features(df)
    print(f"   Created {len(regime_features.columns)} regime features")
    print(f"   Features: {list(regime_features.columns[:5])}...")
    
    # Align dataframes
    df_aligned = df.loc[regime_features.index]
    
    # Train HMM
    model, scaler = train_regime_hmm(regime_features, n_states=3, n_iter=1000)
    
    # Predict regimes
    print("\n🔮 Predicting market regimes...")
    regimes = predict_regimes(model, scaler, regime_features)
    
    # Label regimes
    regimes_labeled, regime_names = label_regimes(regimes, regime_features)
    
    # Save regime data
    print("\n💾 Saving regime predictions...")
    regime_df = df_aligned.copy()
    regime_df['regime'] = regimes_labeled
    regime_df['regime_name'] = [regime_names[r] for r in regimes_labeled]
    
    output_path = Path("data/processed/btcusdt_1h_with_regimes.parquet")
    regime_df.to_parquet(output_path)
    print(f"   Saved to: {output_path}")
    
    # Save model
    import pickle
    model_path = Path("models/market_regime_hmm.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'regime_names': regime_names}, f)
    print(f"   Model saved to: {model_path}")
    
    # Visualize
    print("\n📊 Creating visualizations...")
    plot_output = Path("outputs/market_regime_detection.png")
    plot_regime_timeline(df_aligned, regimes_labeled, regime_names, plot_output)
    
    # Transition analysis
    print("\n🔄 REGIME TRANSITION ANALYSIS:")
    transitions = pd.DataFrame({
        'from': regimes_labeled[:-1],
        'to': regimes_labeled[1:]
    })
    
    transition_matrix = pd.crosstab(
        transitions['from'],
        transitions['to'],
        normalize='index'
    ) * 100
    
    transition_matrix.index = [regime_names[i] for i in transition_matrix.index]
    transition_matrix.columns = [regime_names[i] for i in transition_matrix.columns]
    
    print(f"\n   Regime Transition Probabilities (%):")
    print(transition_matrix.round(1))
    
    print("\n✅ Market regime detection complete!")
    print(f"\n💡 Next steps:")
    print(f"   1. Use regime labels to train separate models per regime")
    print(f"   2. Or use regime as additional feature in main model")
    print(f"   3. Apply regime-specific trading strategies")


if __name__ == "__main__":
    main()
