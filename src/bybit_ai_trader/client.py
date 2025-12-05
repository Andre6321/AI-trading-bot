"""
High-level Bybit trading client with simplified methods and error handling.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal, ROUND_DOWN

from .api import BybitAPI


logger = logging.getLogger(__name__)


class BybitClient:
    """
    High-level wrapper around BybitAPI for simplified trading operations.
    
    Provides:
    - Simplified trading methods
    - Automatic retry logic
    - Error handling and logging
    - Position and balance management
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Bybit trading client.
        
        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
            testnet: Use testnet (True) or mainnet (False)
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api = BybitAPI(api_key, api_secret, testnet)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"BybitClient initialized ({'testnet' if testnet else 'mainnet'})")
    
    def _retry_on_failure(self, func, *args, **kwargs):
        """
        Retry a function call on failure.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        logger.error(f"All {self.max_retries} attempts failed")
        raise last_exception
    
    # ========== Market Data ==========
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get latest price for a symbol.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            
        Returns:
            Latest price as float
        """
        ticker = self._retry_on_failure(self.api.get_ticker, symbol)
        return float(ticker["lastPrice"])
    
    def get_latest_candles(
        self,
        symbol: str,
        interval: str,
        limit: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Get latest candlestick data.
        
        Args:
            symbol: Trading pair
            interval: Timeframe (1, 5, 15, 60, 240, D, W, M)
            limit: Number of candles
            
        Returns:
            List of candle dictionaries with OHLCV data
        """
        klines = self._retry_on_failure(
            self.api.get_klines,
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        # Convert to list of dictionaries
        candles = []
        for kline in reversed(klines):  # Bybit returns newest first
            candles.append({
                "timestamp": int(kline[0]),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5])
            })
        
        return candles
    
    # ========== Account & Balance ==========
    
    def get_balance(self, coin: str = "USDT") -> Dict[str, float]:
        """
        Get wallet balance for a specific coin.
        
        Args:
            coin: Coin symbol (USDT, BTC, etc.)
            
        Returns:
            Dictionary with wallet balance info
        """
        balance_data = self._retry_on_failure(self.api.get_wallet_balance)
        
        # Parse balance information
        for coin_data in balance_data["list"][0]["coin"]:
            if coin_data["coin"] == coin:
                return {
                    "coin": coin,
                    "equity": float(coin_data.get("equity", 0)),
                    "available": float(coin_data.get("availableToWithdraw", 0)),
                    "wallet_balance": float(coin_data.get("walletBalance", 0)),
                    "used": float(coin_data.get("locked", 0))
                }
        
        return {
            "coin": coin,
            "equity": 0.0,
            "available": 0.0,
            "wallet_balance": 0.0,
            "used": 0.0
        }
    
    # ========== Position Management ==========
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current position for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Position dictionary or None if no position
        """
        positions = self._retry_on_failure(self.api.get_positions, symbol=symbol)
        
        if not positions:
            return None
        
        position = positions[0]
        size = float(position.get("size", 0))
        
        if size == 0:
            return None
        
        return {
            "symbol": position["symbol"],
            "side": position["side"],
            "size": size,
            "entry_price": float(position.get("avgPrice", 0)),
            "mark_price": float(position.get("markPrice", 0)),
            "leverage": float(position.get("leverage", 1)),
            "unrealized_pnl": float(position.get("unrealisedPnl", 0)),
            "stop_loss": float(position.get("stopLoss", 0)) if position.get("stopLoss") else None,
            "take_profit": float(position.get("takeProfit", 0)) if position.get("takeProfit") else None
        }
    
    def has_position(self, symbol: str) -> bool:
        """
        Check if there's an open position for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            True if position exists, False otherwise
        """
        return self.get_position(symbol) is not None
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.
        
        Args:
            symbol: Trading pair
            leverage: Leverage value (1-100)
            
        Returns:
            True if successful
        """
        try:
            self._retry_on_failure(
                self.api.set_leverage,
                symbol=symbol,
                buy_leverage=str(leverage),
                sell_leverage=str(leverage)
            )
            logger.info(f"Set leverage to {leverage}x for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return False
    
    # ========== Order Execution ==========
    
    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[str]:
        """
        Place a market order.
        
        Args:
            symbol: Trading pair
            side: "Buy" or "Sell"
            quantity: Order quantity
            reduce_only: Reduce only flag
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            result = self._retry_on_failure(
                self.api.place_order,
                symbol=symbol,
                side=side,
                order_type="Market",
                qty=str(quantity),
                reduce_only=reduce_only,
                stop_loss=str(stop_loss) if stop_loss else None,
                take_profit=str(take_profit) if take_profit else None
            )
            
            order_id = result["orderId"]
            logger.info(f"Market order placed: {side} {quantity} {symbol} (ID: {order_id})")
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return None
    
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reduce_only: bool = False,
        time_in_force: str = "GTC"
    ) -> Optional[str]:
        """
        Place a limit order.
        
        Args:
            symbol: Trading pair
            side: "Buy" or "Sell"
            quantity: Order quantity
            price: Limit price
            reduce_only: Reduce only flag
            time_in_force: GTC, IOC, FOK
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            result = self._retry_on_failure(
                self.api.place_order,
                symbol=symbol,
                side=side,
                order_type="Limit",
                qty=str(quantity),
                price=str(price),
                reduce_only=reduce_only,
                time_in_force=time_in_force
            )
            
            order_id = result["orderId"]
            logger.info(f"Limit order placed: {side} {quantity} {symbol} @ {price} (ID: {order_id})")
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            return None
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            symbol: Trading pair
            order_id: Order ID to cancel
            
        Returns:
            True if successful
        """
        try:
            self._retry_on_failure(
                self.api.cancel_order,
                symbol=symbol,
                order_id=order_id
            )
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    def cancel_all_orders(self, symbol: str) -> bool:
        """
        Cancel all open orders for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            True if successful
        """
        try:
            self._retry_on_failure(self.api.cancel_all_orders, symbol=symbol)
            logger.info(f"All orders cancelled for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False
    
    def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get all open orders for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            List of open orders
        """
        try:
            orders = self._retry_on_failure(
                self.api.get_open_orders,
                symbol=symbol
            )
            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    # ========== Position Exit/Modification ==========
    
    def close_position(self, symbol: str) -> bool:
        """
        Close an open position using market order.
        
        Args:
            symbol: Trading pair
            
        Returns:
            True if successful
        """
        position = self.get_position(symbol)
        if not position:
            logger.warning(f"No position to close for {symbol}")
            return False
        
        # Determine close side (opposite of position side)
        close_side = "Sell" if position["side"] == "Buy" else "Buy"
        
        # Place reduce-only market order
        order_id = self.place_market_order(
            symbol=symbol,
            side=close_side,
            quantity=position["size"],
            reduce_only=True
        )
        
        return order_id is not None
    
    def set_stop_loss_take_profit(
        self,
        symbol: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Set stop loss and/or take profit for existing position.
        
        Args:
            symbol: Trading pair
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            True if successful
        """
        try:
            self._retry_on_failure(
                self.api.set_trading_stop,
                symbol=symbol,
                position_idx=0,  # 0 for one-way mode
                stop_loss=str(stop_loss) if stop_loss else None,
                take_profit=str(take_profit) if take_profit else None
            )
            logger.info(f"Set SL/TP for {symbol}: SL={stop_loss}, TP={take_profit}")
            return True
        except Exception as e:
            logger.error(f"Failed to set SL/TP: {e}")
            return False
    
    # ========== Helper Methods ==========
    
    def calculate_position_size(
        self,
        symbol: str,
        usd_amount: float,
        leverage: int = 1
    ) -> float:
        """
        Calculate position size in base currency for given USD amount.
        
        Args:
            symbol: Trading pair
            usd_amount: USD value of position
            leverage: Leverage to use
            
        Returns:
            Position size in base currency
        """
        price = self.get_latest_price(symbol)
        position_value = usd_amount * leverage
        size = position_value / price
        
        # Round down to avoid insufficient balance errors
        # TODO: Get precision from exchange info
        size = float(Decimal(str(size)).quantize(Decimal("0.001"), rounding=ROUND_DOWN))
        
        return size