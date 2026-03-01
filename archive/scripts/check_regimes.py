import pandas as pd

df = pd.read_parquet('data/processed/btcusdt_1h_with_regimes.parquet')

print(f'Total samples: {len(df)}')
print(f'\nLast 5 columns: {list(df.columns)[-5:]}')
print(f'\nRegime distribution:')
print(df["regime_name"].value_counts())
print(f'\nAvailable columns: {list(df.columns[:10])}...')
print(f'\nRegime statistics by name:')
for regime in df["regime_name"].unique():
    regime_data = df[df["regime_name"] == regime]
    print(f"\n{regime}:")
    print(f"  Count: {len(regime_data)} ({len(regime_data)/len(df)*100:.1f}%)")

print(f'\nRecent 10 samples:')
print(df[["timestamp", "close", "regime", "regime_name"]].tail(10))
