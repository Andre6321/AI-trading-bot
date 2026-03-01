"""
Fetch historical funding rates from Bybit and merge with candle data.

Bybit V5 API: /v5/market/funding/history
Funding is settled every 8 hours on BTC perpetuals.
We interpolate to 1h bars and merge with existing feature parquet.

Usage:
    python scripts/fetch_funding_rates.py
"""
import os, sys, time
os.environ["PYTHONIOENCODING"] = "utf-8"

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).parent.parent


def fetch_funding_history(
    symbol: str = "BTCUSDT",
    start_ts_ms: int = None,
    end_ts_ms: int = None,
    limit: int = 200,
) -> list:
    """Fetch one page of funding rate history from Bybit V5 (public, no auth)."""
    url = "https://api.bybit.com/v5/market/funding/history"
    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": limit,
    }
    if start_ts_ms:
        params["startTime"] = start_ts_ms
    if end_ts_ms:
        params["endTime"] = end_ts_ms

    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error: {data}")
    return data["result"]["list"]


def fetch_all_funding(symbol: str = "BTCUSDT",
                      start_date: str = "2020-01-01") -> pd.DataFrame:
    """
    Paginate backwards to fetch full funding rate history.
    Bybit returns newest-first, max 200 per page.
    """
    print(f"[FUNDING] Fetching funding rate history for {symbol}...")
    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    now_ms = int(datetime.utcnow().timestamp() * 1000)

    all_records = []
    cursor_end = now_ms
    page = 0

    while cursor_end > start_ms:
        page += 1
        records = fetch_funding_history(symbol, end_ts_ms=cursor_end, limit=200)
        if not records:
            break

        all_records.extend(records)

        # Bybit returns newest first; last element is oldest in page
        oldest_ts = int(records[-1]["fundingRateTimestamp"])
        if oldest_ts >= cursor_end:
            break  # no progress
        cursor_end = oldest_ts - 1

        if page % 10 == 0:
            print(f"   Page {page}: {len(all_records):,} records so far "
                  f"(oldest: {pd.Timestamp(oldest_ts, unit='ms')})")
        time.sleep(0.15)  # rate-limit courtesy

    if not all_records:
        print("   [WARN] No funding records fetched!")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["fundingRateTimestamp"].astype(int), unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "funding_rate"]].drop_duplicates("timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"   Fetched {len(df):,} funding records")
    print(f"   Range: {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")
    print(f"   Mean funding: {df['funding_rate'].mean():.6f}  "
          f"(annualised: {df['funding_rate'].mean()*3*365*100:.1f}%)")
    return df


def merge_funding_with_candles(candle_path: Path, funding_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 8h funding rates into 1h candle DataFrame by forward-filling.
    Also builds derived funding alpha features.
    """
    print(f"\n[MERGE] Merging funding into candle data...")
    df = pd.read_parquet(candle_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    print(f"   Candles: {len(df):,} rows ({df.index[0]} → {df.index[-1]})")

    # Set funding timestamp as index for merge
    funding_df = funding_df.set_index("timestamp").sort_index()

    # Merge via reindex + forward-fill (funding persists until next settlement)
    df["funding_rate"] = funding_df["funding_rate"].reindex(df.index, method="ffill")

    n_filled = df["funding_rate"].notna().sum()
    n_missing = df["funding_rate"].isna().sum()
    print(f"   Funding filled: {n_filled:,} bars  |  Missing: {n_missing:,} bars")

    # Build funding-derived features
    if n_filled > 100:
        fr = df["funding_rate"]
        df["funding_cum_8h"] = fr.rolling(8, min_periods=1).sum()
        df["funding_cum_24h"] = fr.rolling(24, min_periods=1).sum()
        df["funding_cum_72h"] = fr.rolling(72, min_periods=1).sum()

        # Funding z-score (contrarian signal)
        fr_mean = fr.rolling(168, min_periods=24).mean()
        fr_std = fr.rolling(168, min_periods=24).std()
        df["funding_zscore"] = (fr - fr_mean) / (fr_std + 1e-10)
        df["funding_mr_signal"] = -df["funding_zscore"]  # contrarian

        # Funding acceleration
        df["funding_accel"] = fr.diff().rolling(8, min_periods=1).mean()

        # Extreme funding flags (for regime detection)
        df["funding_extreme_long"] = (df["funding_zscore"] > 2.0).astype(float)
        df["funding_extreme_short"] = (df["funding_zscore"] < -2.0).astype(float)

        print(f"   Built 8 funding-derived features")
        print(f"   Funding z-score stats: mean={df['funding_zscore'].mean():.3f}, "
              f"std={df['funding_zscore'].std():.3f}")

    return df


def main():
    print("=" * 70)
    print("  FETCH FUNDING RATES FROM BYBIT")
    print("=" * 70)

    # Step 1: Fetch funding rate history
    funding_df = fetch_all_funding("BTCUSDT", start_date="2019-01-01")

    if funding_df.empty:
        print("[ERR] No funding data fetched. Check network.")
        return

    # Save raw funding data
    funding_path = ROOT / "data" / "raw" / "btcusdt_funding_rates.parquet"
    funding_path.parent.mkdir(parents=True, exist_ok=True)
    funding_df.to_parquet(funding_path)
    print(f"\n[SAVE] Raw funding data: {funding_path}")

    # Step 2: Merge with candle data
    candle_path = ROOT / "data" / "processed" / "btcusdt_1h_full_history_enhanced.parquet"
    if not candle_path.exists():
        print(f"[WARN] Candle data not found at {candle_path}")
        print("   Funding data saved separately. Merge will happen in training.")
        return

    df_merged = merge_funding_with_candles(candle_path, funding_df)

    # Save updated dataset
    out_path = ROOT / "data" / "processed" / "btcusdt_1h_full_history_enhanced.parquet"
    df_merged.to_parquet(out_path)
    print(f"\n[SAVE] Updated dataset with funding: {out_path}")
    print(f"   Total columns: {len(df_merged.columns)}")
    print(f"   Total rows: {len(df_merged):,}")

    # Quick stats
    if "funding_rate" in df_merged.columns:
        fr = df_merged["funding_rate"].dropna()
        print(f"\n[STATS] Funding rate summary:")
        print(f"   Count:   {len(fr):,}")
        print(f"   Mean:    {fr.mean():.6f}")
        print(f"   Std:     {fr.std():.6f}")
        print(f"   Min:     {fr.min():.6f}")
        print(f"   Max:     {fr.max():.6f}")
        print(f"   >0 (longs pay): {(fr > 0).mean():.1%}")
        print(f"   <0 (shorts pay): {(fr < 0).mean():.1%}")


if __name__ == "__main__":
    main()
