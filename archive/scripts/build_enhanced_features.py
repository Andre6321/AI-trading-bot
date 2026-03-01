"""
Enhanced feature building pipeline script.

This script:
1. Uses existing raw OHLCV data
2. Builds comprehensive enhanced features (50+ indicators)
3. Saves processed features for advanced ML model training

Output files:
- data/processed/btcusdt_1h_enhanced_features.parquet

Usage: python scripts/build_enhanced_features.py
"""
import sys
from pathlib import Path
import pandas as pd

# Add src to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from bybit_ai_trader.research.features_enhanced import build_enhanced_features


def main():
    """Main enhanced feature building pipeline."""
    print("🚀 Enhanced Feature Building Pipeline")
    print("=" * 60)
    
    try:
        # Use existing raw data
        raw_data_path = Path(__file__).parent.parent / "data" / "raw" / "bybit_btcusdt_1h.parquet"
        
        if not raw_data_path.exists():
            print(f"❌ Raw data not found at: {raw_data_path}")
            print("Please run 'python scripts/build_features.py' first to download data.")
            return 1
        
        print(f"📥 Loading raw data from: {raw_data_path}")
        df_raw = pd.read_parquet(raw_data_path)
        print(f"   Loaded {len(df_raw)} rows")
        
        # Build enhanced features
        print(f"\n🔧 Building enhanced features...")
        print("   This will add:")
        print("   - Multi-timeframe features (1H, 4H, 1D)")
        print("   - Volatility indicators (BB, ATR percentile)")
        print("   - Volume analysis (VWAP, OBV)")
        print("   - Trend strength (ADX, MACD, SAR, CCI, Williams %R)")
        print("   - Market structure (swing highs/lows, S/R levels)")
        print()
        
        df_enhanced = build_enhanced_features(df_raw)
        
        # Save enhanced features
        output_path = Path(__file__).parent.parent / "data" / "processed" / "btcusdt_1h_enhanced_features.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving enhanced features to: {output_path}")
        df_enhanced.to_parquet(output_path, index=False)
        
        print(f"\n✅ Enhanced feature engineering completed!")
        print(f"   Input: {len(df_raw)} rows")
        print(f"   Output: {len(df_enhanced)} rows")
        print(f"   Features: {len(df_enhanced.columns)} columns")
        print(f"   Dropped {len(df_raw) - len(df_enhanced)} rows with NaN values")
        
        # Show feature summary
        print(f"\n📊 Feature categories:")
        feature_cols = df_enhanced.columns.tolist()
        
        # Count feature categories
        multi_tf = [c for c in feature_cols if any(x in c for x in ['_1h_', '_4h_', '_1d_', 'trend_alignment'])]
        volatility = [c for c in feature_cols if any(x in c for x in ['bb_', 'atr_percentile', 'volatility_regime'])]
        volume = [c for c in feature_cols if any(x in c for x in ['vwap', 'obv', 'volume_spike', 'vpt'])]
        trend = [c for c in feature_cols if any(x in c for x in ['adx', 'macd', 'sar', 'cci', 'willr', 'momentum'])]
        structure = [c for c in feature_cols if any(x in c for x in ['swing_', 'support', 'resistance', 'candle_'])]
        
        print(f"   Multi-timeframe: {len(multi_tf)} features")
        print(f"   Volatility: {len(volatility)} features")
        print(f"   Volume: {len(volume)} features")
        print(f"   Trend strength: {len(trend)} features")
        print(f"   Market structure: {len(structure)} features")
        
        # Verify output file
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"\n📁 Output file size: {file_size:.2f} MB")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error in enhanced feature building pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
