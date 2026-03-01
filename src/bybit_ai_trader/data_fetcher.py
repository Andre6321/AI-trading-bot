"""
Real-time data fetcher for live trading with feature calculation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

from .client import BybitClient
from .research.features_enhanced import build_enhanced_features


logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches latest market data and calculates features for ML model.
    
    Maintains a rolling buffer of candlesticks to enable feature calculation
    that requires historical data (moving averages, RSI, etc.).
    """
    
    def __init__(
        self,
        client: BybitClient,
        symbol: str,
        interval: str = "240",  # 4-hour candles
        buffer_size: int = 200  # Keep last 200 candles for feature calculation
    ):
        """
        Initialize data fetcher.
        
        Args:
            client: BybitClient instance
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe in minutes (240 = 4 hours)
            buffer_size: Number of candles to maintain in buffer
        """
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.buffer_size = buffer_size
        
        # Data buffer
        self.candle_buffer: pd.DataFrame = pd.DataFrame()
        self.last_update = None
        
        logger.info(f"DataFetcher initialized: {symbol} @ {interval}min interval")
    
    def fetch_latest_data(self) -> pd.DataFrame:
        """
        Fetch latest candlestick data from exchange.
        
        Returns:
            DataFrame with OHLCV data
        """
        candles = self.client.get_latest_candles(
            symbol=self.symbol,
            interval=self.interval,
            limit=self.buffer_size
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Fetched {len(df)} candles for {self.symbol}")
        return df
    
    def update_buffer(self) -> bool:
        """
        Update the candle buffer with latest data.
        
        Returns:
            True if buffer was updated successfully
        """
        try:
            new_data = self.fetch_latest_data()
            
            if new_data.empty:
                logger.warning("No data received from exchange")
                return False
            
            # Update buffer
            if self.candle_buffer.empty:
                self.candle_buffer = new_data
            else:
                # Merge new data with existing buffer
                self.candle_buffer = pd.concat([self.candle_buffer, new_data])
                
                # Remove duplicates (keep last)
                self.candle_buffer = self.candle_buffer[~self.candle_buffer.index.duplicated(keep='last')]
                
                # Sort by timestamp
                self.candle_buffer.sort_index(inplace=True)
                
                # Keep only last buffer_size candles
                if len(self.candle_buffer) > self.buffer_size:
                    self.candle_buffer = self.candle_buffer.iloc[-self.buffer_size:]
            
            self.last_update = datetime.now()
            logger.debug(f"Buffer updated: {len(self.candle_buffer)} candles")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update buffer: {e}")
            return False
    
    def calculate_features(self) -> pd.DataFrame:
        """
        Calculate all features from the candle buffer.
        
        Returns:
            DataFrame with calculated features
        """
        if self.candle_buffer.empty:
            logger.warning("Buffer is empty, cannot calculate features")
            return pd.DataFrame()
        
        try:
            # Calculate all technical indicators with enhanced features
            df = build_enhanced_features(self.candle_buffer.copy())
            
            logger.debug(f"Calculated {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate features: {e}")
            return pd.DataFrame()
    
    def get_latest_features(self) -> Dict[str, float]:
        """
        Get latest feature values for ML model prediction.
        
        Returns:
            Dictionary of feature names and values
        """
        # Update buffer with latest data
        if not self.update_buffer():
            logger.error("Failed to update data buffer")
            return {}
        
        # Calculate features
        features_df = self.calculate_features()
        
        if features_df.empty:
            logger.error("No features calculated")
            return {}
        
        # Get latest row (most recent complete candle)
        # Note: Last candle might be incomplete, so we take second-to-last
        if len(features_df) < 2:
            logger.warning("Insufficient data for feature calculation")
            return {}
        
        latest_features = features_df.iloc[-2].to_dict()  # -2 to get last complete candle
        
        # Remove OHLCV columns and other non-feature columns
        columns_to_exclude = ['open', 'high', 'low', 'close', 'volume', 
                             'close_position', 'volume_ma_24h', 'volume_ratio']
        
        for col in columns_to_exclude:
            latest_features.pop(col, None)
        
        # Remove any NaN values
        latest_features = {k: v for k, v in latest_features.items() if not pd.isna(v)}
        
        logger.info(f"Latest features calculated: {len(latest_features)} features")
        logger.debug(f"Feature names: {list(latest_features.keys())}")
        return latest_features
    
    def get_current_price(self) -> float:
        """
        Get current market price.
        
        Returns:
            Current price
        """
        return self.client.get_latest_price(self.symbol)
    
    def is_new_candle_closed(self, last_candle_timestamp: pd.Timestamp) -> bool:
        """
        Check if a new candle has closed since last check.
        
        Args:
            last_candle_timestamp: Timestamp of last processed candle
            
        Returns:
            True if new candle is available
        """
        if self.candle_buffer.empty:
            return False
        
        current_latest = self.candle_buffer.index[-1]
        return current_latest > last_candle_timestamp
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get data fetcher status information.
        
        Returns:
            Status dictionary
        """
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "buffer_size": len(self.candle_buffer),
            "last_update": self.last_update,
            "last_candle": self.candle_buffer.index[-1] if not self.candle_buffer.empty else None
        }
