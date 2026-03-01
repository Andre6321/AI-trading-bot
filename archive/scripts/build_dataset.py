"""
Build complete dataset from raw Bybit data to ML-ready features.

This script:
1. Loads data/raw/bybit_btcusdt_perp_1h_with_sentiment.parquet
2. Applies feature engineering using build_features()
3. Saves result to data/processed/btcusdt_1h_features.parquet

Usage: python scripts/build_dataset.py
"""
import os
import sys
from pathlib import Path

# Add src to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

import pandas as pd
from features.feature_builder import build_features, get_feature_groups


def main():
    """Main dataset building pipeline."""
    print("=== Bybit BTCUSDT Dataset Builder ===\n")
    
    # Define paths
    raw_data_path = script_dir.parent / "data" / "raw" / "bybit_btcusdt_perp_1h_with_sentiment.parquet"
    processed_dir = script_dir.parent / "data" / "processed"
    output_path = processed_dir / "btcusdt_1h_features.parquet"
    
    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if raw data exists
    if not raw_data_path.exists():
        print(f"❌ Raw data file not found: {raw_data_path}")
        print("   Please run 'python src/data/download_bybit_data.py' first to download the data.")
        return 1
    
    try:
        # Load raw data
        print("📥 Loading raw data...")
        raw_df = pd.read_parquet(raw_data_path)
        print(f"   Loaded {len(raw_df):,} rows from {raw_data_path.name}")
        print(f"   Date range: {raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}")
        print(f"   Columns: {list(raw_df.columns)}\n")
        
        # Build features
        print("🔧 Building features...")
        features_df = build_features(raw_df)
        print(f"   Feature engineering complete!\n")
        
        # Show feature summary
        feature_groups = get_feature_groups()
        print("📊 Feature Summary:")
        total_features = 0
        for group_name, feature_list in feature_groups.items():
            available_features = [f for f in feature_list if f in features_df.columns]
            total_features += len(available_features)
            print(f"   {group_name}: {len(available_features)} features")
            if len(available_features) != len(feature_list):
                missing = set(feature_list) - set(available_features)
                print(f"     Missing: {missing}")
        
        print(f"   Total features: {total_features}")
        print(f"   Final dataset shape: {features_df.shape}\n")
        
        # Save processed features
        print("💾 Saving processed dataset...")
        features_df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"   ✅ Saved to: {output_path}")
        
        # Show basic statistics
        print("\n📈 Dataset Statistics:")
        print("   Key features:")
        stats_cols = []
        for col in ['returns_1h', 'returns_24h', 'rsi_14', 'volatility_24h', 'funding_z_score', 'oi_z_score']:
            if col in features_df.columns:
                stats_cols.append(col)
        
        if stats_cols:
            print(features_df[stats_cols].describe().round(4))
        
        # Check for any remaining NaN values
        nan_counts = features_df.isnull().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        if len(cols_with_nans) > 0:
            print(f"\n⚠️  Columns with NaN values:")
            for col, count in cols_with_nans.items():
                pct = count / len(features_df) * 100
                print(f"   {col}: {count:,} ({pct:.1f}%)")
        else:
            print("\n✅ No NaN values found in final dataset")
        
        print(f"\n🎉 Dataset building complete!")
        print(f"   Input: {len(raw_df):,} rows")
        print(f"   Output: {len(features_df):,} rows")
        print(f"   Features: {total_features}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error building dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
