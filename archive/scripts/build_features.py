"""
Feature building pipeline script.

This script:
1. Downloads Bybit BTCUSDT 1h data with funding rates
2. Builds comprehensive features from the raw data
3. Saves processed features for ML model training

Output files:
- data/raw/bybit_btcusdt_1h_with_funding.parquet
- data/processed/btcusdt_1h_features.parquet

Usage: python scripts/build_features.py
"""
import sys
from pathlib import Path

# Add src to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from bybit_ai_trader.research.data_download import download_bybit_btcusdt_1h
from bybit_ai_trader.research.features import build_and_save_features


def main():
    """Main feature building pipeline."""
    print("🚀 Feature Building Pipeline")
    print("=" * 50)
    
    try:
        # Step 1: Download raw data with funding rates
        print("📥 Step 1: Downloading Bybit BTCUSDT data...")
        ohlcv_path, funding_path = download_bybit_btcusdt_1h(
            years=3,
            include_funding=True
        )
        
        if funding_path is None:
            print("⚠️  Warning: No funding data available, using OHLCV only")
            input_path = ohlcv_path
        else:
            print(f"✅ Using funding-enriched data: {funding_path}")
            input_path = funding_path
        
        # Step 2: Build features from raw data
        print(f"\n🔧 Step 2: Building features...")
        features_path = build_and_save_features(raw_path=input_path)
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📁 Output files:")
        print(f"   Raw data: {input_path}")
        print(f"   Features: {features_path}")
        
        # Verify output files
        if input_path.exists():
            raw_size = input_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   Raw file size: {raw_size:.2f} MB")
        
        if features_path.exists():
            features_size = features_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   Features file size: {features_size:.2f} MB")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error in feature building pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
