"""Fix timestamps in processed data and merge funding rates."""
import pandas as pd
import numpy as np

# Load raw data for timestamps
raw = pd.read_parquet("data/raw/btcusdt_1h_full_history.parquet")
proc = pd.read_parquet("data/processed/btcusdt_1h_full_history_enhanced.parquet")
print(f"Processed: {len(proc)} rows, Raw: {len(raw)} rows")

# Match by close price
proc_first_close = proc["close"].iloc[0]
raw_match = raw[raw["close"] == proc_first_close]
print(f"Proc first close ({proc_first_close}) found at raw index: {raw_match.index.tolist()[:5]}")

if len(raw_match) == 0:
    print("ERROR: Cannot match close prices!")
    exit(1)

start_idx = raw_match.index[0]
end_idx = start_idx + len(proc)
print(f"Using raw[{start_idx}:{end_idx}]")

if end_idx > len(raw):
    # Truncate processed data to available raw timestamps
    available = len(raw) - start_idx
    print(f"WARNING: Raw shorter. Using {available} rows instead of {len(proc)}")
    proc = proc.iloc[:available]
    end_idx = len(raw)

timestamps = raw["timestamp"].iloc[start_idx:end_idx].values
ts_index = pd.DatetimeIndex(timestamps)
if ts_index.tz is not None:
    ts_index = ts_index.tz_localize(None)
proc.index = ts_index
proc.index.name = "timestamp"
print(f"Fixed index: {proc.index[0]} → {proc.index[-1]}")

# Now merge funding data
fund = pd.read_parquet("data/raw/btcusdt_funding_rates.parquet")
fund_ts = pd.DatetimeIndex(fund["timestamp"])
if fund_ts.tz is not None:
    fund_ts = fund_ts.tz_localize(None)
fund_indexed = fund.set_index(fund_ts)["funding_rate"]
fund_indexed = fund_indexed.sort_index()

# Remove any old funding columns first
for col in list(proc.columns):
    if "funding" in col.lower():
        proc.drop(col, axis=1, inplace=True)

proc["funding_rate"] = fund_indexed.reindex(proc.index, method="ffill")
n_filled = proc["funding_rate"].notna().sum()
print(f"Funding filled: {n_filled}/{len(proc)} bars")

# Build funding features
if n_filled > 100:
    fr = proc["funding_rate"]
    proc["funding_cum_8h"] = fr.rolling(8, min_periods=1).sum()
    proc["funding_cum_24h"] = fr.rolling(24, min_periods=1).sum()
    proc["funding_cum_72h"] = fr.rolling(72, min_periods=1).sum()
    fr_mean = fr.rolling(168, min_periods=24).mean()
    fr_std = fr.rolling(168, min_periods=24).std()
    proc["funding_zscore"] = (fr - fr_mean) / (fr_std + 1e-10)
    proc["funding_mr_signal"] = -proc["funding_zscore"]
    proc["funding_accel"] = fr.diff().rolling(8, min_periods=1).mean()
    proc["funding_extreme_long"] = (proc["funding_zscore"] > 2.0).astype(float)
    proc["funding_extreme_short"] = (proc["funding_zscore"] < -2.0).astype(float)
    print("Built 8 funding features")

# Save
proc.to_parquet("data/processed/btcusdt_1h_full_history_enhanced.parquet")
print(f"Saved with {len(proc.columns)} columns, {len(proc)} rows")

# Stats
if "funding_rate" in proc.columns:
    fr = proc["funding_rate"].dropna()
    print(f"\nFunding stats:")
    print(f"  Count: {len(fr):,}")
    print(f"  Mean: {fr.mean():.6f}")
    print(f"  >0 (longs pay): {(fr > 0).mean():.1%}")
    print(f"  Annualised: {fr.mean() * 3 * 365 * 100:.1f}%")
