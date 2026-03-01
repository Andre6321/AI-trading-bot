"""
Build enhanced features from full historical dataset.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from bybit_ai_trader.research.features_enhanced import build_enhanced_features


def main():
    """Build features from full history."""
    print("🚀 Enhanced Feature Building Pipeline (FULL HISTORY)")
    print("=" * 60)
    
    # Load full history
    raw_data_path = Path(__file__).parent.parent / "data" / "raw" / "btcusdt_1h_full_history.parquet"
    
    if not raw_data_path.exists():
        print(f"❌ Full history data not found at: {raw_data_path}")
        print("Please run 'python src/data/download_multi_source.py' first.")
        return 1
    
    print(f"📥 Loading full history from: {raw_data_path}")
    df_raw = pd.read_parquet(raw_data_path)
    
    # Ensure timezone-naive timestamps (some sources provide UTC-aware)
    if hasattr(df_raw['timestamp'].dt, 'tz') and df_raw['timestamp'].dt.tz is not None:
        df_raw['timestamp'] = df_raw['timestamp'].dt.tz_localize(None)
    
    print(f"   Loaded {len(df_raw)} rows ({(df_raw['timestamp'].max() - df_raw['timestamp'].min()).days} days)")
    
    # Build enhanced features
    print(f"\n🔧 Building enhanced features...")
    df_enhanced = build_enhanced_features(df_raw)
    
    # Save enhanced features
    output_path = Path(__file__).parent.parent / "data" / "processed" / "btcusdt_1h_full_history_enhanced.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving enhanced features to: {output_path}")
    df_enhanced.to_parquet(output_path, index=True)
    
    print(f"\n✅ Enhanced feature engineering completed!")
    print(f"   Input: {len(df_raw)} rows")
    print(f"   Output: {len(df_enhanced)} rows ({len(df_enhanced.columns)} features)")
    print(f"   Dropped: {len(df_raw) - len(df_enhanced)} rows with NaN")
    print(f"   Date range: {df_enhanced.index[0] if len(df_enhanced) > 0 else 'N/A'}")
    
    # File size
    if output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"\n📁 Output file size: {file_size:.2f} MB")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
