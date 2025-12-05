# Contents of /bybit_ai_trader/bybit_ai_trader/src/bybit_ai_trader/api.py

"""
Bybit V5 API client with HMAC authentication for live trading.
"""

import hmac
import hashlib
import time
import json
import requests
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode


class BybitAPI:
    """
    Low-level Bybit V5 API client with HMAC SHA256 authentication.
    
    Handles authenticated requests to Bybit exchange for:
    - Order placement and management
    - Position queries
    - Balance and wallet information
    - Market data
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        recv_window: int = 5000
    ):
        """
        Initialize Bybit API client.
        
        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
            testnet: Use testnet (True) or mainnet (False)
            recv_window: Request validity window in milliseconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window = recv_window
        
        # Set base URL
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
    
    def _generate_signature(self, params: str) -> str:
        """
        Generate HMAC SHA256 signature for authentication.
        
        Args:
            params: String to sign (timestamp + api_key + recv_window + params)
            
        Returns:
            Hex signature string
        """
        return hmac.new(
            self.api_secret.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)
    
    def _send_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        auth_required: bool = True
    ) -> Dict[str, Any]:
        """
        Send HTTP request to Bybit API.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint (e.g., "/v5/order/create")
            params: Request parameters
            auth_required: Whether authentication is required
            
        Returns:
            API response as dictionary
            
        Raises:
            Exception: If API returns error
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if params is None:
            params = {}
        
        if auth_required:
            timestamp = self._get_timestamp()
            
            if method == "GET":
                # For GET requests, params go in query string
                param_str = urlencode(sorted(params.items()))
                sign_str = f"{timestamp}{self.api_key}{self.recv_window}{param_str}"
                signature = self._generate_signature(sign_str)
                
                headers.update({
                    "X-BAPI-API-KEY": self.api_key,
                    "X-BAPI-SIGN": signature,
                    "X-BAPI-TIMESTAMP": str(timestamp),
                    "X-BAPI-RECV-WINDOW": str(self.recv_window)
                })
                
                response = requests.get(url, params=params, headers=headers)
            
            else:  # POST, DELETE
                # For POST/DELETE, params go in body
                param_str = json.dumps(params, separators=(',', ':'))
                sign_str = f"{timestamp}{self.api_key}{self.recv_window}{param_str}"
                signature = self._generate_signature(sign_str)
                
                headers.update({
                    "X-BAPI-API-KEY": self.api_key,
                    "X-BAPI-SIGN": signature,
                    "X-BAPI-TIMESTAMP": str(timestamp),
                    "X-BAPI-RECV-WINDOW": str(self.recv_window)
                })
                
                if method == "POST":
                    response = requests.post(url, json=params, headers=headers)
                elif method == "DELETE":
                    response = requests.delete(url, json=params, headers=headers)
                else:
                    raise ValueError(f"Unsupported method: {method}")
        else:
            # Public endpoint (no authentication)
            if method == "GET":
                response = requests.get(url, params=params, headers=headers)
            else:
                raise ValueError(f"Unsupported public method: {method}")
        
        # Parse response
        try:
            data = response.json()
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response: {response.text}")
        
        # Check for API errors
        if data.get("retCode") != 0:
            raise Exception(f"API Error {data.get('retCode')}: {data.get('retMsg')}")
        
        return data
    
    # ========== Market Data (Public) ==========
    
    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 200
    ) -> List[List]:
        """
        Get candlestick/kline data.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of candles to return (max 1000)
            
        Returns:
            List of klines [timestamp, open, high, low, close, volume, turnover]
        """
        params = {
            "category": "linear",  # USDT perpetual
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time
        
        response = self._send_request("GET", "/v5/market/kline", params, auth_required=False)
        return response["result"]["list"]
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get latest ticker information.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            
        Returns:
            Ticker data with last price, bid, ask, volume, etc.
        """
        params = {
            "category": "linear",
            "symbol": symbol
        }
        
        response = self._send_request("GET", "/v5/market/tickers", params, auth_required=False)
        return response["result"]["list"][0]
    
    # ========== Account & Wallet ==========
    
    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict[str, Any]:
        """
        Get wallet balance.
        
        Args:
            account_type: Account type (UNIFIED, CONTRACT)
            
        Returns:
            Balance information
        """
        params = {"accountType": account_type}
        response = self._send_request("GET", "/v5/account/wallet-balance", params)
        return response["result"]
    
    # ========== Positions ==========
    
    def get_positions(
        self,
        symbol: Optional[str] = None,
        settle_coin: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get position information.
        
        Args:
            symbol: Trading pair (optional, returns all if not specified)
            settle_coin: Settle coin (USDT, USDC)
            
        Returns:
            List of positions
        """
        params = {"category": "linear"}
        
        if symbol:
            params["symbol"] = symbol
        if settle_coin:
            params["settleCoin"] = settle_coin
        
        response = self._send_request("GET", "/v5/position/list", params)
        return response["result"]["list"]
    
    def set_leverage(
        self,
        symbol: str,
        buy_leverage: str,
        sell_leverage: str
    ) -> Dict[str, Any]:
        """
        Set leverage for a symbol.
        
        Args:
            symbol: Trading pair
            buy_leverage: Leverage for long positions
            sell_leverage: Leverage for short positions
            
        Returns:
            API response
        """
        params = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(buy_leverage),
            "sellLeverage": str(sell_leverage)
        }
        
        response = self._send_request("POST", "/v5/position/set-leverage", params)
        return response["result"]
    
    # ========== Orders ==========
    
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: str,
        price: Optional[str] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        close_on_trigger: bool = False,
        order_link_id: Optional[str] = None,
        stop_loss: Optional[str] = None,
        take_profit: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Place an order.
        
        Args:
            symbol: Trading pair
            side: Buy or Sell
            order_type: Market, Limit
            qty: Order quantity
            price: Order price (required for Limit orders)
            time_in_force: GTC, IOC, FOK
            reduce_only: Reduce only order
            close_on_trigger: Close on trigger
            order_link_id: Custom order ID
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order information with orderId
        """
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": time_in_force
        }
        
        if price:
            params["price"] = str(price)
        if reduce_only:
            params["reduceOnly"] = True
        if close_on_trigger:
            params["closeOnTrigger"] = True
        if order_link_id:
            params["orderLinkId"] = order_link_id
        if stop_loss:
            params["stopLoss"] = str(stop_loss)
        if take_profit:
            params["takeProfit"] = str(take_profit)
        
        response = self._send_request("POST", "/v5/order/create", params)
        return response["result"]
    
    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            symbol: Trading pair
            order_id: Order ID (use orderId OR orderLinkId)
            order_link_id: Custom order ID
            
        Returns:
            Cancellation result
        """
        params = {
            "category": "linear",
            "symbol": symbol
        }
        
        if order_id:
            params["orderId"] = order_id
        elif order_link_id:
            params["orderLinkId"] = order_link_id
        else:
            raise ValueError("Must provide either order_id or order_link_id")
        
        response = self._send_request("POST", "/v5/order/cancel", params)
        return response["result"]
    
    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """
        Cancel all open orders for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Cancellation result
        """
        params = {
            "category": "linear",
            "symbol": symbol
        }
        
        response = self._send_request("POST", "/v5/order/cancel-all", params)
        return response["result"]
    
    def get_open_orders(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get open orders.
        
        Args:
            symbol: Trading pair (optional)
            order_id: Order ID (optional)
            order_link_id: Custom order ID (optional)
            
        Returns:
            List of open orders
        """
        params = {"category": "linear"}
        
        if symbol:
            params["symbol"] = symbol
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id
        
        response = self._send_request("GET", "/v5/order/realtime", params)
        return response["result"]["list"]
    
    # ========== Trading Position Management ==========
    
    def set_trading_stop(
        self,
        symbol: str,
        position_idx: int,
        stop_loss: Optional[str] = None,
        take_profit: Optional[str] = None,
        trailing_stop: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Set trading stop (SL/TP) for open position.
        
        Args:
            symbol: Trading pair
            position_idx: Position index (0: one-way mode, 1: buy hedge, 2: sell hedge)
            stop_loss: Stop loss price
            take_profit: Take profit price
            trailing_stop: Trailing stop distance
            
        Returns:
            API response
        """
        params = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": position_idx
        }
        
        if stop_loss:
            params["stopLoss"] = str(stop_loss)
        if take_profit:
            params["takeProfit"] = str(take_profit)
        if trailing_stop:
            params["trailingStop"] = str(trailing_stop)
        
        response = self._send_request("POST", "/v5/position/trading-stop", params)
        return response["result"]