"""
Add target column to regime-labeled data for training.
"""

import pandas as pd
import numpy as np

print("📥 Loading regime data...")
df = pd.read_parquet('data/processed/btcusdt_1h_with_regimes.parquet')
print(f"   Loaded {len(df):,} samples")
print(f"   Columns: {len(df.columns)}")

# Create target: will price go up in next 4 hours?
print("\n🎯 Creating target (4-hour forward return)...")

# Calculate 4-hour forward return
df['future_ret_4h'] = df['close'].shift(-4) / df['close'] - 1

# Binary target: 1 if price goes up, 0 if down
df['target_4h'] = (df['future_ret_4h'] > 0).astype(int)

# Remove last 4 rows (no future data)
df = df[df['target_4h'].notna()].copy()

print(f"   Valid samples: {len(df):,}")
print(f"\n   Class distribution:")
print(f"     UP (1):   {(df['target_4h']==1).sum():>6} ({(df['target_4h']==1).sum()/len(df)*100:.1f}%)")
print(f"     DOWN (0): {(df['target_4h']==0).sum():>6} ({(df['target_4h']==0).sum()/len(df)*100:.1f}%)")

# Save updated data
print("\n💾 Saving updated data...")
df.to_parquet('data/processed/btcusdt_1h_with_regimes.parquet')
print("   ✅ Saved!")

print("\n📊 Regime-specific class distribution:")
for regime in ['BEAR', 'SIDEWAYS', 'BULL']:
    regime_df = df[df['regime_name'] == regime]
    up_pct = (regime_df['target_4h']==1).sum() / len(regime_df) * 100
    print(f"\n{regime:>10}: {len(regime_df):>6} samples")
    print(f"             UP rate: {up_pct:>5.1f}%")
