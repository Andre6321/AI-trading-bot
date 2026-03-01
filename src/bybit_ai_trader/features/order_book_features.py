"""
Order Book Feature Engineering for Trading Bot.

Fetches order book data from Bybit and calculates microstructure features:
- Bid/ask spread (liquidity indicator)
- Order book imbalance (buy/sell pressure)
- Large order detection (whale activity)
- Book depth (support/resistance levels)
- Price level clustering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pybit.unified_trading import HTTP
import time

logger = logging.getLogger(__name__)


class OrderBookAnalyzer:
    """
    Analyzes order book data to extract trading signals.
    
    Order book features capture real-time market microstructure and
    can significantly improve prediction accuracy (+5-10% expected).
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        depth_levels: int = 50,
        large_order_threshold: float = 10.0  # USD (e.g., 10 BTC = large order)
    ):
        """
        Initialize order book analyzer.
        
        Args:
            symbol: Trading pair symbol
            depth_levels: Number of price levels to fetch (max 50 for Bybit)
            large_order_threshold: Minimum size to consider "large order" in base currency
        """
        self.symbol = symbol
        self.depth_levels = depth_levels
        self.large_order_threshold = large_order_threshold
        
        # Initialize Bybit client (no auth needed for public data)
        self.client = HTTP(testnet=False)
        
        logger.info(f"OrderBookAnalyzer initialized for {symbol}")
    
    def fetch_order_book(self) -> Optional[Dict]:
        """
        Fetch current order book from Bybit.
        
        Returns:
            Dictionary with bids and asks, or None if error
        """
        try:
            response = self.client.get_orderbook(
                category="spot",
                symbol=self.symbol,
                limit=self.depth_levels
            )
            
            if response['retCode'] == 0:
                data = response['result']
                
                # Parse bids and asks
                bids = [(float(price), float(qty)) for price, qty in data['b']]
                asks = [(float(price), float(qty)) for price, qty in data['a']]
                
                return {
                    'bids': bids,
                    'asks': asks,
                    'timestamp': data['ts']
                }
            else:
                logger.error(f"Bybit API error: {response['retMsg']}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch order book: {e}")
            return None
    
    def calculate_spread_features(self, order_book: Dict) -> Dict[str, float]:
        """
        Calculate bid-ask spread features.
        
        Tight spread = high liquidity, easy to enter/exit
        Wide spread = low liquidity, expensive to trade
        
        Args:
            order_book: Order book data with bids and asks
            
        Returns:
            Dictionary of spread features
        """
        bids = order_book['bids']
        asks = order_book['asks']
        
        if not bids or not asks:
            return {}
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        
        # Absolute and relative spread
        spread_abs = best_ask - best_bid
        spread_pct = (spread_abs / best_bid) * 100
        
        # Mid price
        mid_price = (best_bid + best_ask) / 2
        
        # Spread relative to recent volatility (if available)
        # For now, just use percentage
        
        return {
            'ob_spread_abs': spread_abs,
            'ob_spread_pct': spread_pct,
            'ob_mid_price': mid_price,
            'ob_best_bid': best_bid,
            'ob_best_ask': best_ask
        }
    
    def calculate_imbalance_features(self, order_book: Dict, levels: int = 10) -> Dict[str, float]:
        """
        Calculate order book imbalance (buy vs sell pressure).
        
        Imbalance > 1: More buying pressure (bullish)
        Imbalance < 1: More selling pressure (bearish)
        
        Args:
            order_book: Order book data
            levels: Number of levels to analyze (default 10)
            
        Returns:
            Dictionary of imbalance features
        """
        bids = order_book['bids'][:levels]
        asks = order_book['asks'][:levels]
        
        if not bids or not asks:
            return {}
        
        # Sum volumes
        bid_volume = sum(qty for price, qty in bids)
        ask_volume = sum(qty for price, qty in asks)
        
        # Calculate imbalance ratio
        total_volume = bid_volume + ask_volume
        imbalance_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0
        imbalance_pct = (bid_volume - ask_volume) / total_volume * 100 if total_volume > 0 else 0
        
        # Volume-weighted average prices
        bid_vwap = sum(price * qty for price, qty in bids) / bid_volume if bid_volume > 0 else 0
        ask_vwap = sum(price * qty for price, qty in asks) / ask_volume if ask_volume > 0 else 0
        
        return {
            f'ob_imbalance_ratio_{levels}': imbalance_ratio,
            f'ob_imbalance_pct_{levels}': imbalance_pct,
            f'ob_bid_volume_{levels}': bid_volume,
            f'ob_ask_volume_{levels}': ask_volume,
            f'ob_total_volume_{levels}': total_volume,
            f'ob_bid_vwap_{levels}': bid_vwap,
            f'ob_ask_vwap_{levels}': ask_vwap
        }
    
    def detect_large_orders(self, order_book: Dict) -> Dict[str, float]:
        """
        Detect large orders (whale activity) in the order book.
        
        Large orders often indicate institutional activity and can act
        as support/resistance levels.
        
        Args:
            order_book: Order book data
            
        Returns:
            Dictionary of large order features
        """
        bids = order_book['bids']
        asks = order_book['asks']
        
        # Find large orders
        large_bids = [(price, qty) for price, qty in bids if qty >= self.large_order_threshold]
        large_asks = [(price, qty) for price, qty in asks if qty >= self.large_order_threshold]
        
        # Count and total volume
        large_bid_count = len(large_bids)
        large_ask_count = len(large_asks)
        large_bid_volume = sum(qty for price, qty in large_bids)
        large_ask_volume = sum(qty for price, qty in large_asks)
        
        # Distance to nearest large order
        mid_price = (bids[0][0] + asks[0][0]) / 2 if bids and asks else 0
        
        nearest_large_bid_dist = 0
        if large_bids:
            nearest_large_bid = max(large_bids, key=lambda x: x[0])[0]
            nearest_large_bid_dist = (mid_price - nearest_large_bid) / mid_price * 100
        
        nearest_large_ask_dist = 0
        if large_asks:
            nearest_large_ask = min(large_asks, key=lambda x: x[0])[0]
            nearest_large_ask_dist = (nearest_large_ask - mid_price) / mid_price * 100
        
        return {
            'ob_large_bid_count': large_bid_count,
            'ob_large_ask_count': large_ask_count,
            'ob_large_bid_volume': large_bid_volume,
            'ob_large_ask_volume': large_ask_volume,
            'ob_large_bid_dist_pct': nearest_large_bid_dist,
            'ob_large_ask_dist_pct': nearest_large_ask_dist,
            'ob_large_imbalance': large_bid_count - large_ask_count
        }
    
    def calculate_depth_features(self, order_book: Dict, depth_pct: float = 1.0) -> Dict[str, float]:
        """
        Calculate order book depth within X% of mid price.
        
        Depth measures how much volume is available near current price.
        High depth = strong support/resistance, low slippage
        
        Args:
            order_book: Order book data
            depth_pct: Percentage range to analyze (default 1%)
            
        Returns:
            Dictionary of depth features
        """
        bids = order_book['bids']
        asks = order_book['asks']
        
        if not bids or not asks:
            return {}
        
        mid_price = (bids[0][0] + asks[0][0]) / 2
        
        # Price range for depth calculation
        lower_bound = mid_price * (1 - depth_pct / 100)
        upper_bound = mid_price * (1 + depth_pct / 100)
        
        # Sum volume within range
        bid_depth = sum(qty for price, qty in bids if price >= lower_bound)
        ask_depth = sum(qty for price, qty in asks if price <= upper_bound)
        
        # Count levels within range
        bid_levels = len([1 for price, qty in bids if price >= lower_bound])
        ask_levels = len([1 for price, qty in asks if price <= upper_bound])
        
        return {
            f'ob_bid_depth_{int(depth_pct*100)}bp': bid_depth,
            f'ob_ask_depth_{int(depth_pct*100)}bp': ask_depth,
            f'ob_depth_imbalance_{int(depth_pct*100)}bp': bid_depth - ask_depth,
            f'ob_bid_levels_{int(depth_pct*100)}bp': bid_levels,
            f'ob_ask_levels_{int(depth_pct*100)}bp': ask_levels
        }
    
    def calculate_price_clustering(self, order_book: Dict) -> Dict[str, float]:
        """
        Detect price level clustering (psychological levels).
        
        Large orders often cluster at round numbers (e.g., $100k, $99k)
        which act as strong support/resistance.
        
        Args:
            order_book: Order book data
            
        Returns:
            Dictionary of clustering features
        """
        bids = order_book['bids']
        asks = order_book['asks']
        
        # Find "round" price levels (ending in 00, 000, etc)
        def is_round_level(price: float) -> bool:
            price_int = int(price)
            return price_int % 100 == 0 or price_int % 1000 == 0
        
        # Volume at round levels
        round_bid_volume = sum(qty for price, qty in bids if is_round_level(price))
        round_ask_volume = sum(qty for price, qty in asks if is_round_level(price))
        
        total_bid_volume = sum(qty for price, qty in bids)
        total_ask_volume = sum(qty for price, qty in asks)
        
        # Percentage at round levels
        round_bid_pct = (round_bid_volume / total_bid_volume * 100) if total_bid_volume > 0 else 0
        round_ask_pct = (round_ask_volume / total_ask_volume * 100) if total_ask_volume > 0 else 0
        
        return {
            'ob_round_bid_volume': round_bid_volume,
            'ob_round_ask_volume': round_ask_volume,
            'ob_round_bid_pct': round_bid_pct,
            'ob_round_ask_pct': round_ask_pct
        }
    
    def get_all_features(self) -> Dict[str, float]:
        """
        Fetch order book and calculate all features.
        
        Returns:
            Dictionary of all order book features
        """
        # Fetch order book
        order_book = self.fetch_order_book()
        
        if order_book is None:
            logger.warning("Could not fetch order book, returning empty features")
            return {}
        
        # Calculate all feature sets
        features = {}
        
        # 1. Spread features
        features.update(self.calculate_spread_features(order_book))
        
        # 2. Imbalance at multiple depths
        for levels in [5, 10, 20]:
            features.update(self.calculate_imbalance_features(order_book, levels=levels))
        
        # 3. Large order detection
        features.update(self.detect_large_orders(order_book))
        
        # 4. Depth features at multiple ranges
        for depth_pct in [0.5, 1.0, 2.0]:
            features.update(self.calculate_depth_features(order_book, depth_pct=depth_pct))
        
        # 5. Price clustering
        features.update(self.calculate_price_clustering(order_book))
        
        logger.debug(f"Calculated {len(features)} order book features")
        
        return features
    
    def monitor_order_book(self, duration_seconds: int = 60, interval_seconds: int = 5) -> pd.DataFrame:
        """
        Monitor order book over time and collect features.
        
        Useful for analyzing order book dynamics and building historical features.
        
        Args:
            duration_seconds: How long to monitor
            interval_seconds: Sampling interval
            
        Returns:
            DataFrame with time-series of order book features
        """
        print(f"\n📊 Monitoring order book for {duration_seconds}s (sampling every {interval_seconds}s)...")
        
        data = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            features = self.get_all_features()
            
            if features:
                features['timestamp'] = pd.Timestamp.now()
                data.append(features)
                print(f"  Sample {len(data)}: Spread {features.get('ob_spread_pct', 0):.3f}%, "
                      f"Imbalance {features.get('ob_imbalance_ratio_10', 1):.2f}")
            
            time.sleep(interval_seconds)
        
        df = pd.DataFrame(data)
        print(f"\n✅ Collected {len(df)} samples with {len(df.columns)-1} features")
        
        return df


def demonstrate_order_book_features():
    """Demonstrate order book feature extraction."""
    print("="*70)
    print("📖 ORDER BOOK FEATURE DEMONSTRATION")
    print("="*70)
    
    # Initialize analyzer
    analyzer = OrderBookAnalyzer(
        symbol="BTCUSDT",
        depth_levels=50,
        large_order_threshold=5.0  # 5 BTC
    )
    
    # Fetch and display features
    print("\n🔍 Fetching current order book...")
    features = analyzer.get_all_features()
    
    if not features:
        print("❌ Could not fetch order book data")
        return
    
    print(f"\n✅ Extracted {len(features)} order book features\n")
    
    # Display by category
    print("💰 SPREAD FEATURES:")
    for key, value in features.items():
        if 'spread' in key or 'mid_price' in key or 'best_' in key:
            print(f"   {key}: {value:.4f}")
    
    print("\n⚖️ IMBALANCE FEATURES (10 levels):")
    for key, value in features.items():
        if 'imbalance' in key and '_10' in key:
            print(f"   {key}: {value:.4f}")
    
    print("\n🐋 LARGE ORDER DETECTION:")
    for key, value in features.items():
        if 'large' in key:
            print(f"   {key}: {value:.4f}")
    
    print("\n📊 DEPTH FEATURES (1% range):")
    for key, value in features.items():
        if 'depth' in key and '100bp' in key:
            print(f"   {key}: {value:.4f}")
    
    print("\n🎯 CLUSTERING FEATURES:")
    for key, value in features.items():
        if 'round' in key:
            print(f"   {key}: {value:.4f}")
    
    # Interpretation
    print("\n" + "="*70)
    print("💡 SIGNAL INTERPRETATION")
    print("="*70)
    
    imbalance = features.get('ob_imbalance_ratio_10', 1.0)
    spread_pct = features.get('ob_spread_pct', 0)
    large_imbalance = features.get('ob_large_imbalance', 0)
    
    print(f"\n📈 Market Pressure:")
    if imbalance > 1.1:
        print(f"   🟢 BULLISH - Buy pressure dominates (ratio: {imbalance:.2f})")
    elif imbalance < 0.9:
        print(f"   🔴 BEARISH - Sell pressure dominates (ratio: {imbalance:.2f})")
    else:
        print(f"   🟡 NEUTRAL - Balanced order book (ratio: {imbalance:.2f})")
    
    print(f"\n💧 Liquidity:")
    if spread_pct < 0.01:
        print(f"   🟢 EXCELLENT - Very tight spread ({spread_pct:.4f}%)")
    elif spread_pct < 0.05:
        print(f"   🟡 GOOD - Normal spread ({spread_pct:.4f}%)")
    else:
        print(f"   🔴 POOR - Wide spread ({spread_pct:.4f}%)")
    
    print(f"\n🐋 Whale Activity:")
    if abs(large_imbalance) >= 3:
        direction = "BUY" if large_imbalance > 0 else "SELL"
        print(f"   ⚠️ SIGNIFICANT - Large orders on {direction} side ({abs(large_imbalance)} detected)")
    elif abs(large_imbalance) >= 1:
        print(f"   🟡 MODERATE - Some large orders ({abs(large_imbalance)} detected)")
    else:
        print(f"   🟢 LOW - No significant whale activity")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    demonstrate_order_book_features()
