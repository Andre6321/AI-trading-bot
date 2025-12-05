"""
Multi-source historical data downloader for BTC/USDT.

Downloads from multiple sources and combines:
1. Binance (via CCXT) - most reliable for long history
2. Yahoo Finance - BTC-USD data
3. Bybit (via CCXT) - for validation

Usage: python src/data/download_multi_source.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
from tqdm import tqdm

# Paths
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_from_binance(years=3, timeframe='1h'):
    """
    Download from Binance (usually has the most history).
    
    Args:
        years: Number of years of history
        timeframe: Timeframe (1h, 4h, 1d)
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"\n📥 Downloading from Binance ({years} years, {timeframe})...")
    
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        symbol = 'BTC/USDT'
        
        # Calculate start date
        now = datetime.now(timezone.utc)
        since_dt = now - timedelta(days=365 * years)
        since_ms = int(since_dt.timestamp() * 1000)
        
        print(f"   Fetching from {since_dt.date()} to {now.date()}...")
        
        all_candles = []
        current_since = since_ms
        
        with tqdm(desc="Downloading", unit="candles") as pbar:
            while current_since < int(now.timestamp() * 1000):
                try:
                    candles = exchange.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        since=current_since,
                        limit=1000  # Binance allows 1000
                    )
                    
                    if not candles:
                        break
                    
                    all_candles.extend(candles)
                    pbar.update(len(candles))
                    
                    # Move to next batch
                    current_since = candles[-1][0] + 1
                    
                    # Stop if we got less than requested (reached end)
                    if len(candles) < 1000:
                        break
                        
                except Exception as e:
                    print(f"\n   Warning: {e}")
                    break
        
        if not all_candles:
            print("   ❌ No data retrieved from Binance")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        print(f"   ✅ Downloaded {len(df)} candles from Binance")
        print(f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Binance download failed: {e}")
        return None


def download_from_yahoo(years=3, interval='1h'):
    """
    Download from Yahoo Finance (BTC-USD).
    
    Args:
        years: Number of years of history
        interval: Interval (1h, 1d)
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"\n📥 Downloading from Yahoo Finance ({years} years, {interval})...")
    
    try:
        # Yahoo Finance uses BTC-USD
        ticker = yf.Ticker("BTC-USD")
        
        # Calculate start/end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        print(f"   Fetching from {start_date.date()} to {end_date.date()}...")
        
        # Download data
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False
        )
        
        if df.empty:
            print("   ❌ No data retrieved from Yahoo Finance")
            return None
        
        # Rename columns to match our format
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'timestamp' if 'Datetime' in df.columns else 'Date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Ensure timestamp column exists
        if 'Date' in df.columns and 'timestamp' not in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})
        
        # Convert to UTC timezone
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
        
        # Select only needed columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.dropna(subset=['timestamp'])
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        print(f"   ✅ Downloaded {len(df)} candles from Yahoo Finance")
        print(f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Yahoo Finance download failed: {e}")
        return None


def merge_sources(binance_df, yahoo_df):
    """
    Merge data from multiple sources, preferring Binance.
    
    Args:
        binance_df: Binance data
        yahoo_df: Yahoo Finance data
        
    Returns:
        Merged DataFrame
    """
    print(f"\n🔄 Merging data sources...")
    
    dfs = []
    
    if binance_df is not None:
        binance_df['source'] = 'binance'
        dfs.append(binance_df)
        print(f"   Binance: {len(binance_df)} candles")
    
    if yahoo_df is not None:
        yahoo_df['source'] = 'yahoo'
        dfs.append(yahoo_df)
        print(f"   Yahoo: {len(yahoo_df)} candles")
    
    if not dfs:
        raise ValueError("No data available from any source!")
    
    # Combine all sources
    merged = pd.concat(dfs, ignore_index=True)
    
    # Sort by timestamp
    merged = merged.sort_values('timestamp')
    
    # Remove duplicates, keeping first (preferring Binance if both exist at same time)
    merged = merged.drop_duplicates(subset=['timestamp'], keep='first')
    
    # Drop source column
    merged = merged.drop(columns=['source'])
    
    print(f"   ✅ Merged to {len(merged)} unique candles")
    print(f"   Final range: {merged['timestamp'].min().date()} to {merged['timestamp'].max().date()}")
    
    return merged


def main():
    """Main download pipeline."""
    print("=" * 60)
    print("🚀 MULTI-SOURCE DATA DOWNLOADER")
    print("=" * 60)
    
    YEARS = 5
    TIMEFRAME = '1h'
    
    # Download from multiple sources
    binance_df = download_from_binance(years=YEARS, timeframe=TIMEFRAME)
    yahoo_df = download_from_yahoo(years=YEARS, interval=TIMEFRAME)
    
    # Merge data
    try:
        final_df = merge_sources(binance_df, yahoo_df)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    # Save to file
    output_file = OUTPUT_DIR / "btcusdt_1h_full_history.parquet"
    final_df.to_parquet(output_file, index=False)
    
    print(f"\n💾 Saved to: {output_file}")
    print(f"   File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    
    # Summary statistics
    print(f"\n📊 Data Summary:")
    print(f"   Total candles: {len(final_df)}")
    print(f"   Date range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
    print(f"   Duration: {(final_df['timestamp'].max() - final_df['timestamp'].min()).days} days")
    print(f"   Price range: ${final_df['low'].min():.2f} - ${final_df['high'].max():.2f}")
    print(f"\n   Sample data:")
    print(final_df.head())
    print(f"\n   Latest data:")
    print(final_df.tail())
    
    print(f"\n✅ Download complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
