"""
Data download module for Bybit BTCUSDT perpetual futures.

This module provides functions to download OHLCV and funding rate data
from Bybit using CCXT and direct API calls.
"""
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd
import numpy as np
import requests


def download_bybit_btcusdt_1h(years: int = 3, 
                              include_funding: bool = True,
                              output_dir: Optional[Path] = None) -> tuple[Path, Optional[Path]]:
    """
    Download 1h OHLCV data for BTCUSDT perpetuals for the last N years.
    
    Args:
        years: Number of years of historical data to download
        include_funding: Whether to fetch and include funding rate data
        output_dir: Directory to save files (defaults to data/raw/)
    
    Returns:
        Tuple of (ohlcv_path, funding_path) where funding_path is None if not included
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize exchange
    print("🔗 Initializing Bybit connection...")
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "sandbox": False  # Use live API for historical data
    })
    
    # Resolve symbol
    symbol = _resolve_bybit_symbol(exchange)
    print(f"📊 Using symbol: {symbol}")
    
    # Calculate time range
    now = datetime.now(timezone.utc)
    since_dt = now - timedelta(days=365 * years)
    since_ms = int(since_dt.timestamp() * 1000)
    
    print(f"📅 Fetching data since: {since_dt.isoformat()} ({years} years)")
    
    # Download OHLCV data
    print("📈 Downloading OHLCV data...")
    ohlcv_data = _fetch_ohlcv_data(exchange, symbol, since_ms)
    
    # Create DataFrame with normalized columns
    ohlcv_df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"], unit="ms", utc=True)
    ohlcv_df.set_index("timestamp", inplace=True)
    
    print(f"✅ Downloaded {len(ohlcv_df):,} OHLCV records")
    
    # Save OHLCV only file
    ohlcv_path = output_dir / "bybit_btcusdt_1h.parquet"
    ohlcv_df.to_parquet(ohlcv_path, engine="pyarrow")
    print(f"💾 Saved OHLCV data to: {ohlcv_path}")
    
    funding_path = None
    
    if include_funding:
        try:
            print("💰 Downloading funding rate data...")
            funding_data = _fetch_funding_rate_data(since_ms)
            
            if funding_data:
                # Create funding DataFrame
                funding_df = pd.DataFrame(funding_data)
                funding_df["timestamp"] = pd.to_datetime(funding_df["timestamp"], unit="ms", utc=True)
                funding_df.set_index("timestamp", inplace=True)
                
                # Merge with OHLCV data using forward fill
                merged_df = ohlcv_df.copy()
                merged_df = pd.merge_asof(
                    merged_df.reset_index().sort_values("timestamp"),
                    funding_df.reset_index().sort_values("timestamp"),
                    on="timestamp",
                    direction="backward"
                )
                merged_df.set_index("timestamp", inplace=True)
                
                # Save merged file
                funding_path = output_dir / "bybit_btcusdt_1h_with_funding.parquet"
                merged_df.to_parquet(funding_path, engine="pyarrow")
                print(f"💾 Saved merged data to: {funding_path}")
                print(f"✅ Downloaded {len(funding_data):,} funding rate records")
            else:
                print("⚠️  No funding rate data available")
                
        except Exception as e:
            print(f"⚠️  Error downloading funding data: {e}")
    
    print("🎉 Download completed successfully!")
    return ohlcv_path, funding_path


def _resolve_bybit_symbol(exchange: ccxt.Exchange) -> str:
    """Resolve the correct Bybit symbol for BTCUSDT perpetuals."""
    candidates = ["BTC/USDT:USDT", "BTC/USDT", "BTCUSDT"]
    
    try:
        exchange.load_markets()
        markets = exchange.markets
        
        # Try exact matches first
        for symbol in candidates:
            if symbol in markets:
                return symbol
        
        # Try partial matches
        for market_symbol in markets:
            if "BTC" in market_symbol.upper() and "USDT" in market_symbol.upper():
                return market_symbol
                
    except Exception as e:
        print(f"Warning: Could not load markets: {e}")
    
    # Fallback
    return candidates[0]


def _fetch_ohlcv_data(exchange: ccxt.Exchange, symbol: str, since_ms: int, 
                      timeframe: str = "1h", limit: int = 1000) -> list:
    """Fetch OHLCV data with pagination and rate limiting."""
    all_ohlcv = []
    since = since_ms
    max_retries = 5
    retry_delay = 2
    
    while True:
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol, 
                    timeframe=timeframe, 
                    since=since, 
                    limit=limit
                )
                break
            except Exception as e:
                print(f"OHLCV fetch error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise
        
        if not ohlcv:
            break
            
        all_ohlcv.extend(ohlcv)
        
        # Update since to last timestamp + 1ms
        last_timestamp = ohlcv[-1][0]
        since = last_timestamp + 1
        
        # Break if we got less than requested (end of data)
        if len(ohlcv) < limit:
            break
        
        # Rate limiting
        time.sleep(exchange.rateLimit / 1000 if hasattr(exchange, "rateLimit") else 0.1)
        
        # Progress indicator
        if len(all_ohlcv) % 10000 == 0:
            print(f"   Downloaded {len(all_ohlcv):,} records...")
    
    return all_ohlcv


def _fetch_funding_rate_data(since_ms: int) -> list:
    """Fetch funding rate data directly from Bybit API."""
    base_url = "https://api.bybit.com"
    endpoint = "/v5/market/funding/history"
    
    funding_rates = []
    start_time = since_ms
    limit = 200
    max_retries = 5
    retry_delay = 2
    
    while True:
        params = {
            "category": "linear",
            "symbol": "BTCUSDT",
            "startTime": start_time,
            "limit": limit
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{base_url}{endpoint}", params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                break
            except Exception as e:
                print(f"Funding rate fetch error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise
        
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
        
        # Rate limiting
        time.sleep(0.1)
        
        # Progress indicator
        if len(funding_rates) % 1000 == 0:
            print(f"   Downloaded {len(funding_rates):,} funding records...")
    
    return funding_rates


if __name__ == "__main__":
    # Example usage
    print("Testing Bybit data download...")
    
    try:
        ohlcv_path, funding_path = download_bybit_btcusdt_1h(
            years=1,  # Download 1 year for testing
            include_funding=True
        )
        
        # Load and display sample data
        print(f"\nSample OHLCV data from {ohlcv_path}:")
        ohlcv_df = pd.read_parquet(ohlcv_path)
        print(ohlcv_df.head())
        print(f"Shape: {ohlcv_df.shape}")
        
        if funding_path:
            print(f"\nSample merged data from {funding_path}:")
            merged_df = pd.read_parquet(funding_path)
            print(merged_df.head())
            print(f"Shape: {merged_df.shape}")
            print(f"Columns: {merged_df.columns.tolist()}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()