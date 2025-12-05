"""
Example usage of the feature builder module.

This script demonstrates how to load raw Bybit data and transform it into ML features.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from src.features.feature_builder import build_features, get_feature_groups


def main():
    # Example of how to use the feature builder
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'bybit_btcusdt_perp_1h_with_sentiment.parquet')
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please run src/data/download_bybit_data.py first to download the data.")
        return
    
    print("Loading raw data...")
    raw_data = pd.read_parquet(data_path)
    print(f"Loaded {len(raw_data)} rows of raw data")
    print("Columns:", raw_data.columns.tolist())
    
    print("\nBuilding features...")
    features_df = build_features(raw_data)
    
    print(f"\nFeatures built successfully!")
    print(f"Shape: {features_df.shape}")
    
    # Show feature groups
    groups = get_feature_groups()
    print(f"\nFeature groups:")
    for group_name, features in groups.items():
        available_features = [f for f in features if f in features_df.columns]
        print(f"  {group_name}: {len(available_features)} features")
    
    # Save features to processed data folder
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'btcusdt_features_1h.parquet')
    features_df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"\nSaved features to: {output_path}")
    
    # Basic statistics
    print(f"\nBasic statistics:")
    print(features_df[['returns_1h', 'rsi_14', 'volatility_24h', 'funding_z_score']].describe())


if __name__ == "__main__":
    main()
