"""
Build dataset with order book features for training.

Since historical order book data is not available, we:
1. Use existing enhanced features (50+ indicators)
2. Add synthetic order book proxies based on price/volume action
3. For live trading, replace with real order book features

Synthetic OB features (based on observable price action):
- Estimated spread from high-low range
- Volume imbalance from buy/sell volume ratio
- Price clustering detection
- Support/resistance levels as depth proxies
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))


def add_synthetic_orderbook_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic order book features based on price action.
    
    These are proxies for order book characteristics that can be
    inferred from historical OHLCV data. In live trading, these
    will be replaced with real order book features.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with synthetic OB features
    """
    df = df.copy()
    
    print("   Adding synthetic order book features...")
    
    # 1. SPREAD PROXY: Use intrabar high-low range as spread estimate
    df['ob_spread_pct'] = ((df['high'] - df['low']) / df['close']) * 100
    df['ob_spread_ma5'] = df['ob_spread_pct'].rolling(5).mean()
    df['ob_spread_volatility'] = df['ob_spread_pct'].rolling(20).std()
    
    # Spread regime (tight vs wide)
    spread_percentile = df['ob_spread_pct'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 1 else np.nan
    )
    df['ob_spread_regime'] = (spread_percentile > 0.7).astype(int)  # 1 = wide spread
    
    # 2. IMBALANCE PROXY: Volume-weighted price movement
    # If close > open with high volume = buy pressure
    # If close < open with high volume = sell pressure
    df['price_change_pct'] = df['close'].pct_change() * 100
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Imbalance = directional volume
    df['ob_imbalance_proxy'] = np.sign(df['price_change_pct']) * df['volume_ratio']
    df['ob_imbalance_ma5'] = df['ob_imbalance_proxy'].rolling(5).mean()
    df['ob_imbalance_ma10'] = df['ob_imbalance_proxy'].rolling(10).mean()
    df['ob_imbalance_ma20'] = df['ob_imbalance_proxy'].rolling(20).mean()
    
    # Buy/sell volume split (approximation using candle color)
    df['ob_buy_volume_pct'] = ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)).clip(0, 1) * 100
    df['ob_sell_volume_pct'] = 100 - df['ob_buy_volume_pct']
    
    # Volume imbalance ratio
    df['ob_volume_imbalance_ratio'] = df['ob_buy_volume_pct'] / (df['ob_sell_volume_pct'] + 1e-10)
    
    # 3. LARGE ORDER PROXY: Detect unusual volume spikes at specific price levels
    # Large spike + narrow range = potential whale order
    df['volume_spike'] = df['volume'] / df['volume'].rolling(20).mean()
    df['price_range_pct'] = ((df['high'] - df['low']) / df['close']) * 100
    
    # Large order indicator: High volume + low range = absorption
    df['ob_large_order_signal'] = (
        (df['volume_spike'] > 2.0) & (df['price_range_pct'] < df['price_range_pct'].rolling(20).mean())
    ).astype(int)
    
    df['ob_large_order_count'] = df['ob_large_order_signal'].rolling(10).sum()
    
    # 4. DEPTH PROXY: Support/resistance strength
    # Use pivot points and price clustering as depth indicators
    
    # Rolling highs/lows (S/R levels)
    df['swing_high_20'] = df['high'].rolling(20).max()
    df['swing_low_20'] = df['low'].rolling(20).min()
    
    # Distance to S/R (closer = stronger support/resistance)
    df['ob_distance_to_resistance'] = ((df['swing_high_20'] - df['close']) / df['close']) * 100
    df['ob_distance_to_support'] = ((df['close'] - df['swing_low_20']) / df['close']) * 100
    
    # Depth strength: How many times price tested and held at level
    # Count touches within 0.5% of swing levels
    df['ob_resistance_touches'] = 0
    df['ob_support_touches'] = 0
    
    for i in range(20, len(df)):
        resistance = df['swing_high_20'].iloc[i]
        support = df['swing_low_20'].iloc[i]
        
        # Count touches in last 50 bars
        window = df.iloc[max(0, i-50):i]
        df.loc[df.index[i], 'ob_resistance_touches'] = ((window['high'] >= resistance * 0.995) & 
                                                          (window['high'] <= resistance * 1.005)).sum()
        df.loc[df.index[i], 'ob_support_touches'] = ((window['low'] <= support * 1.005) & 
                                                       (window['low'] >= support * 0.995)).sum()
    
    # Depth strength = more touches = stronger level
    df['ob_depth_strength'] = df['ob_resistance_touches'] + df['ob_support_touches']
    
    # 5. CLUSTERING PROXY: Price at round levels
    # Check if current price is near psychological levels (multiples of 1000, 5000, 10000)
    def distance_to_round_level(price):
        """Calculate distance to nearest round level."""
        levels = [100, 500, 1000, 5000, 10000]
        min_dist = float('inf')
        
        for level in levels:
            remainder = price % level
            dist = min(remainder, level - remainder)
            min_dist = min(min_dist, dist)
        
        return min_dist / price * 100  # As percentage
    
    df['ob_round_level_distance'] = df['close'].apply(distance_to_round_level)
    
    # Near round level flag (within 0.5%)
    df['ob_near_round_level'] = (df['ob_round_level_distance'] < 0.5).astype(int)
    
    # Volume clustering: Higher volume at round levels
    df['ob_round_level_volume'] = df['volume'] * df['ob_near_round_level']
    df['ob_round_level_volume_ma10'] = df['ob_round_level_volume'].rolling(10).mean()
    
    # 6. LIQUIDITY FEATURES
    # Liquidity score: Inverse of spread, scaled by volume
    df['ob_liquidity_score'] = (df['volume_ratio'] / (df['ob_spread_pct'] + 1e-10))
    df['ob_liquidity_ma20'] = df['ob_liquidity_score'].rolling(20).mean()
    
    # Liquidity regime
    liquidity_percentile = df['ob_liquidity_score'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 1 else np.nan
    )
    df['ob_liquidity_regime'] = (liquidity_percentile > 0.7).astype(int)  # 1 = high liquidity
    
    # 7. MICROSTRUCTURE SIGNALS
    # Price impact: How much does volume move price?
    # High impact = low liquidity, Low impact = high liquidity
    df['price_move_abs'] = abs(df['close'] - df['open'])
    df['ob_price_impact'] = df['price_move_abs'] / (df['volume'] + 1e-10)
    df['ob_price_impact_ma20'] = df['ob_price_impact'].rolling(20).mean()
    
    # Tick intensity: Rate of price changes
    df['ob_tick_intensity'] = abs(df['close'].diff()).rolling(10).sum()
    
    # Order flow imbalance strength
    df['ob_flow_strength'] = abs(df['ob_imbalance_ma5']) * df['volume_ratio']
    
    print(f"   Added {sum('ob_' in col for col in df.columns)} order book features")
    
    return df


def build_dataset_with_orderbook():
    """Build complete dataset with order book features."""
    print("🚀 Building Dataset with Order Book Features")
    print("=" * 60)
    
    try:
        # 1. Load enhanced features
        features_path = Path(__file__).parent.parent / "data" / "processed" / "btcusdt_1h_enhanced_features.parquet"
        
        if not features_path.exists():
            print(f"❌ Enhanced features not found at: {features_path}")
            print("Please run 'python scripts/build_enhanced_features.py' first.")
            return 1
        
        print(f"📥 Loading enhanced features from: {features_path}")
        df = pd.read_parquet(features_path)
        print(f"   Loaded {len(df)} rows with {len(df.columns)} features")
        
        # 2. Add synthetic order book features
        print(f"\n🔧 Adding order book features...")
        df = add_synthetic_orderbook_features(df)
        
        # 3. Drop NaN rows
        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)
        print(f"\n🧹 Dropped {rows_before - rows_after} rows with NaN values")
        
        # 4. Save final dataset
        output_path = Path(__file__).parent.parent / "data" / "processed" / "btcusdt_1h_with_orderbook.parquet"
        
        print(f"\n💾 Saving dataset to: {output_path}")
        df.to_parquet(output_path, index=False)
        
        print(f"\n✅ Dataset with order book features completed!")
        print(f"   Rows: {len(df)}")
        print(f"   Total features: {len(df.columns)}")
        
        # Show feature breakdown
        print(f"\n📊 Feature categories:")
        ob_features = [col for col in df.columns if 'ob_' in col]
        other_features = [col for col in df.columns if 'ob_' not in col and col not in ['timestamp', 'target']]
        
        print(f"   Order book features: {len(ob_features)}")
        print(f"   Other features: {len(other_features)}")
        
        # Show sample OB features
        print(f"\n📋 Order book feature categories:")
        spread_features = [col for col in ob_features if 'spread' in col]
        imbalance_features = [col for col in ob_features if 'imbalance' in col]
        depth_features = [col for col in ob_features if 'depth' in col or 'support' in col or 'resistance' in col]
        liquidity_features = [col for col in ob_features if 'liquidity' in col]
        flow_features = [col for col in ob_features if 'flow' in col or 'impact' in col]
        
        print(f"   Spread indicators: {len(spread_features)}")
        print(f"   Imbalance indicators: {len(imbalance_features)}")
        print(f"   Depth indicators: {len(depth_features)}")
        print(f"   Liquidity indicators: {len(liquidity_features)}")
        print(f"   Order flow indicators: {len(flow_features)}")
        
        # File size
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"\n📁 Output file size: {file_size:.2f} MB")
        
        # Show sample statistics
        print(f"\n📈 Sample order book feature statistics:")
        sample_features = [
            'ob_spread_pct', 
            'ob_imbalance_ratio', 
            'ob_liquidity_score',
            'ob_depth_strength',
            'ob_flow_strength'
        ]
        
        for feat in sample_features:
            if feat in df.columns:
                mean_val = df[feat].mean()
                std_val = df[feat].std()
                print(f"   {feat}: μ={mean_val:.3f}, σ={std_val:.3f}")
        
        print(f"\n💡 NOTE: These are synthetic order book features inferred from price action.")
        print(f"   In live trading, these will be replaced with real-time order book data.")
        print(f"   See: src/bybit_ai_trader/features/order_book_features.py")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(build_dataset_with_orderbook())
