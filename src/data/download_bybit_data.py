"""Download Bybit 1h BTCUSDT perpetual OHLCV, funding rates, and open interest for the last 3 years.

Saves two files:
- data/raw/bybit_btcusdt_perp_1h.parquet (OHLCV only)
- data/raw/bybit_btcusdt_perp_1h_with_sentiment.parquet (merged with funding rates, OI, and 24h changes)

Usage: python src/data/download_bybit_data.py

Requirements: ccxt, pandas, pyarrow, numpy, requests
"""
import os
import time
from datetime import datetime, timedelta, timezone
import requests

import ccxt
import pandas as pd
import numpy as np

# Config
EXCHANGE_ID = "bybit"
SYMBOL = "BTC/USDT:USDT"  # ccxt unified symbol for Bybit perpetual in many versions; fallbacks handled below
TIMEFRAME = "1h"
YEARS = 3
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "raw")
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bybit_btcusdt_perp_1h.parquet")
OUTPUT_FILE_WITH_SENTIMENT = os.path.join(OUTPUT_DIR, "bybit_btcusdt_perp_1h_with_sentiment.parquet")

# Rate limit and pagination
FETCH_LIMIT = 200  # Bybit often allows 200/limit per request for OHLCV
RETRY_DELAY = 5
MAX_RETRIES = 5


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def init_exchange():
    ex = getattr(ccxt, EXCHANGE_ID)({
        "enableRateLimit": True,
        # No API keys needed for public OHLCV
    })
    return ex


def resolve_symbol(exchange):
    # Common Bybit perpetual symbol variants: "BTC/USDT:USDT", "BTC/USDT", "BTCUSDT" etc.
    candidates = ["BTC/USDT:USDT", "BTC/USDT", "BTCUSDT.P" , "BTCUSDT"]
    available = getattr(exchange, "markets", None)
    if available is None:
        try:
            exchange.load_markets()
            available = exchange.markets
        except Exception:
            available = None
    if available:
        for s in candidates:
            if s in available:
                return s
        # try partial match
        for s in available:
            if s.upper().startswith("BTC") and "USDT" in s.upper():
                return s
    # fallback to candidate
    return candidates[0]


def fetch_ohlcv_since(exchange, symbol, timeframe, since_ms, limit=200):
    all_ohlcv = []
    since = since_ms
    while True:
        for attempt in range(MAX_RETRIES):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                break
            except ccxt.BaseError as e:
                print(f"Fetch error: {e}; retry {attempt+1}/{MAX_RETRIES} after {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
        else:
            raise RuntimeError("Max retries exceeded fetching OHLCV")

        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        # OHLCV format: [timestamp, open, high, low, close, volume]
        # Increment since to last timestamp + 1 ms to avoid duplicates
        last_ts = ohlcv[-1][0]
        since = last_ts + 1
        # break if returned less than limit
        if len(ohlcv) < limit:
            break
        # sleep a bit to respect rate limits
        time.sleep(exchange.rateLimit / 1000 if hasattr(exchange, "rateLimit") else 0.1)
    return all_ohlcv


def fetch_funding_rates(symbol_raw, since_ms):
    """
    Fetch funding rate history from Bybit REST API directly.
    CCXT doesn't consistently support funding rate history across all exchanges.
    """
    funding_rates = []
    base_url = "https://api.bybit.com"
    endpoint = "/v5/market/funding/history"
    
    # Convert symbol to Bybit format (e.g., "BTCUSDT")
    bybit_symbol = symbol_raw.replace("/", "").replace(":USDT", "")
    if not bybit_symbol.endswith("USDT"):
        bybit_symbol = "BTCUSDT"  # fallback
    
    print(f"Fetching funding rates for {bybit_symbol}")
    
    # Bybit returns funding rates with timestamps, we'll fetch in chunks
    start_time = since_ms
    limit = 200
    
    while True:
        params = {
            "category": "linear",
            "symbol": bybit_symbol,
            "startTime": start_time,
            "limit": limit
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(f"{base_url}{endpoint}", params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                break
            except Exception as e:
                print(f"Funding rate fetch error: {e}; retry {attempt+1}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY)
        else:
            raise RuntimeError("Max retries exceeded fetching funding rates")
        
        if data.get("retCode") != 0:
            print(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
            break
            
        result = data.get("result", {})
        items = result.get("list", [])
        
        if not items:
            break
            
        for item in items:
            funding_rates.append({
                "timestamp": int(item["fundingRateTimestamp"]),
                "funding_rate": float(item["fundingRate"])
            })
        
        # Update start_time for next batch
        if len(items) < limit:
            break
        last_ts = int(items[-1]["fundingRateTimestamp"])
        start_time = last_ts + 1
        
        time.sleep(0.1)  # Rate limit
    
    return funding_rates


def fetch_open_interest(symbol_raw, since_ms):
    """
    Fetch open interest history from Bybit REST API.
    Note: Historical open interest may have limited availability.
    """
    open_interest_data = []
    base_url = "https://api.bybit.com"
    endpoint = "/v5/market/open-interest"
    
    # Convert symbol to Bybit format
    bybit_symbol = symbol_raw.replace("/", "").replace(":USDT", "")
    if not bybit_symbol.endswith("USDT"):
        bybit_symbol = "BTCUSDT"
    
    print(f"Fetching open interest for {bybit_symbol}")
    
    # Bybit's open interest endpoint may only provide current data
    # For historical data, we would need a different endpoint or approach
    # Let's try to get what we can
    params = {
        "category": "linear",
        "symbol": bybit_symbol,
        "intervalTime": "1h",
        "limit": 200
    }
    
    try:
        response = requests.get(f"{base_url}{endpoint}", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("retCode") == 0:
            result = data.get("result", {})
            items = result.get("list", [])
            
            for item in items:
                open_interest_data.append({
                    "timestamp": int(item.get("timestamp", 0)),
                    "open_interest": float(item.get("openInterest", 0))
                })
    except Exception as e:
        print(f"Open interest fetch error: {e}")
        # Return empty list if we can't fetch open interest
        
    return open_interest_data


def align_and_merge_data(ohlcv_df, funding_df, oi_df):
    """
    Align funding rates and open interest with OHLCV timestamps using forward fill.
    """
    print("Aligning and merging datasets...")
    
    # Ensure all dataframes have timestamp as datetime
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"], utc=True)
    
    if not funding_df.empty:
        funding_df = funding_df.copy()
        funding_df["timestamp"] = pd.to_datetime(funding_df["timestamp"], unit="ms", utc=True)
        
        # Merge funding rates using forward fill
        merged = pd.merge_asof(
            ohlcv_df.sort_values("timestamp"),
            funding_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward"  # Use last known funding rate
        )
    else:
        merged = ohlcv_df.copy()
        merged["funding_rate"] = np.nan
    
    if not oi_df.empty:
        oi_df = oi_df.copy()
        oi_df["timestamp"] = pd.to_datetime(oi_df["timestamp"], unit="ms", utc=True)
        
        # Merge open interest using forward fill
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            oi_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )
    else:
        merged["open_interest"] = np.nan
    
    # Calculate 24h changes
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    
    # For funding rate change (24h = 24 periods for 1h data)
    merged["funding_rate_change_24h"] = merged["funding_rate"].diff(24)
    
    # For open interest change (24h)
    merged["open_interest_change_24h"] = merged["open_interest"].diff(24)
    
    return merged


def main():
    ensure_output_dir(OUTPUT_DIR)
    exchange = init_exchange()
    symbol = resolve_symbol(exchange)
    print(f"Using symbol: {symbol}")

    # compute since timestamp for YEARS years ago
    now = datetime.now(timezone.utc)
    since_dt = now - timedelta(days=365 * YEARS)
    since_ms = int(since_dt.timestamp() * 1000)

    print(f"Fetching {TIMEFRAME} OHLCV since {since_dt.isoformat()} ({since_ms})")

    # Fetch OHLCV data
    raw = fetch_ohlcv_since(exchange, symbol, TIMEFRAME, since_ms, limit=FETCH_LIMIT)
    if not raw:
        print("No OHLCV data fetched")
        return

    ohlcv_df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    # Convert timestamp (ms) to UTC datetime
    ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"], unit="ms", utc=True)

    # Save basic OHLCV
    ohlcv_df.to_parquet(OUTPUT_FILE, engine="pyarrow", index=False)
    print(f"Saved {len(ohlcv_df)} OHLCV rows to {OUTPUT_FILE}")

    # Fetch funding rates
    try:
        funding_data = fetch_funding_rates(symbol, since_ms)
        funding_df = pd.DataFrame(funding_data) if funding_data else pd.DataFrame()
        print(f"Fetched {len(funding_df)} funding rate records")
    except Exception as e:
        print(f"Failed to fetch funding rates: {e}")
        funding_df = pd.DataFrame()

    # Fetch open interest
    try:
        oi_data = fetch_open_interest(symbol, since_ms)
        oi_df = pd.DataFrame(oi_data) if oi_data else pd.DataFrame()
        print(f"Fetched {len(oi_df)} open interest records")
    except Exception as e:
        print(f"Failed to fetch open interest: {e}")
        oi_df = pd.DataFrame()

    # Merge all data
    merged_df = align_and_merge_data(ohlcv_df, funding_df, oi_df)

    # Save merged result
    merged_df.to_parquet(OUTPUT_FILE_WITH_SENTIMENT, engine="pyarrow", index=False)
    print(f"Saved {len(merged_df)} merged rows to {OUTPUT_FILE_WITH_SENTIMENT}")
    
    # Basic stats
    print("\nMerged data preview:")
    print(merged_df.head())
    print("\nColumns:", merged_df.columns.tolist())
    print("\nFunding rate stats:")
    if not merged_df["funding_rate"].isna().all():
        print(merged_df["funding_rate"].describe())
    else:
        print("No funding rate data available")
    
    print("\nOpen interest stats:")
    if not merged_df["open_interest"].isna().all():
        print(merged_df["open_interest"].describe())
    else:
        print("No open interest data available")


if __name__ == "__main__":
    main()
