"""
Risk management system for position sizing and risk controls.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk management system for trading bot.
    
    Handles:
    - Position sizing based on account balance
    - Stop-loss and take-profit calculation
    - Maximum drawdown protection
    - Daily loss limits
    - Portfolio exposure limits
    """
    
    def __init__(
        self,
        max_position_size_usd: float = 100.0,
        max_leverage: int = 1,
        stop_loss_pct: float = 0.02,  # 2% stop loss
        take_profit_pct: float = 0.04,  # 4% take profit (2:1 R:R)
        max_daily_loss_usd: float = 50.0,
        max_daily_trades: int = 10,
        max_portfolio_exposure_pct: float = 0.5,  # 50% of account
        risk_per_trade_pct: float = 0.01  # 1% of account per trade
    ):
        """
        Initialize risk manager.
        
        Args:
            max_position_size_usd: Maximum position size in USD
            max_leverage: Maximum allowed leverage
            stop_loss_pct: Stop loss as % of entry price
            take_profit_pct: Take profit as % of entry price
            max_daily_loss_usd: Maximum loss allowed per day
            max_daily_trades: Maximum trades per day
            max_portfolio_exposure_pct: Maximum % of account in positions
            risk_per_trade_pct: Risk per trade as % of account
        """
        # Position sizing parameters
        self.max_position_size_usd = max_position_size_usd
        self.max_leverage = max_leverage
        
        # Risk parameters
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        
        # Daily limits
        self.max_daily_loss_usd = max_daily_loss_usd
        self.max_daily_trades = max_daily_trades
        
        # Portfolio limits
        self.max_portfolio_exposure_pct = max_portfolio_exposure_pct
        
        # Tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        
        logger.info("RiskManager initialized with risk controls")
    
    def _reset_daily_limits_if_needed(self):
        """Reset daily counters if it's a new day."""
        today = datetime.now().date()
        if today > self.last_reset_date:
            logger.info(f"New day - resetting limits. Previous day: PnL=${self.daily_pnl:.2f}, Trades={self.daily_trades}")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = today
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        side: str,
        signal_confidence: float = 1.0
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            account_balance: Total account balance in USD
            entry_price: Expected entry price
            side: "Buy" or "Sell"
            signal_confidence: Signal confidence (0-1), scales position size
            
        Returns:
            Position size in base currency
        """
        # Calculate risk amount (how much we're willing to lose)
        risk_amount = account_balance * self.risk_per_trade_pct
        
        # Calculate position size based on stop loss
        # Position size = Risk amount / (Entry price * Stop loss %)
        position_value_usd = risk_amount / self.stop_loss_pct
        
        # Scale by signal confidence
        position_value_usd *= signal_confidence
        
        # Apply maximum position size limit
        position_value_usd = min(position_value_usd, self.max_position_size_usd)
        
        # Apply portfolio exposure limit
        max_by_exposure = account_balance * self.max_portfolio_exposure_pct
        position_value_usd = min(position_value_usd, max_by_exposure)
        
        # Convert to base currency quantity
        position_size = position_value_usd / entry_price
        
        # Round down to avoid insufficient balance
        position_size = float(Decimal(str(position_size)).quantize(Decimal("0.001"), rounding=ROUND_DOWN))
        
        logger.info(
            f"Position size calculated: ${position_value_usd:.2f} = {position_size:.4f} @ ${entry_price:.2f} "
            f"(Risk: ${risk_amount:.2f}, Confidence: {signal_confidence:.2f})"
        )
        
        return position_size
    
    def calculate_stop_loss_take_profit(
        self,
        entry_price: float,
        side: str
    ) -> Tuple[float, float]:
        """
        Calculate stop-loss and take-profit prices.
        
        Args:
            entry_price: Entry price
            side: "Buy" or "Sell"
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if side == "Buy":
            # Long position
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:
            # Short position
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
        
        logger.debug(f"SL/TP calculated for {side}: SL=${stop_loss:.2f}, TP=${take_profit:.2f}")
        return stop_loss, take_profit
    
    def can_open_new_position(
        self,
        account_balance: float,
        current_positions_value: float = 0.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a new position can be opened based on risk limits.
        
        Args:
            account_balance: Total account balance
            current_positions_value: USD value of current open positions
            
        Returns:
            Tuple of (can_trade, reason_if_not)
        """
        self._reset_daily_limits_if_needed()
        
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss_usd:
            return False, f"Daily loss limit reached: ${self.daily_pnl:.2f}"
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit reached: {self.daily_trades} trades"
        
        # Check portfolio exposure
        exposure_pct = current_positions_value / account_balance if account_balance > 0 else 0
        if exposure_pct >= self.max_portfolio_exposure_pct:
            return False, f"Portfolio exposure limit reached: {exposure_pct:.1%}"
        
        # Check account balance
        if account_balance < self.max_position_size_usd * 0.1:  # Need at least 10% of max position
            return False, f"Insufficient account balance: ${account_balance:.2f}"
        
        return True, None
    
    def record_trade(self, pnl: float):
        """
        Record a completed trade for daily tracking.
        
        Args:
            pnl: Realized PnL from the trade
        """
        self._reset_daily_limits_if_needed()
        
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        logger.info(
            f"Trade recorded: PnL=${pnl:.2f} | "
            f"Daily: PnL=${self.daily_pnl:.2f}, Trades={self.daily_trades}/{self.max_daily_trades}"
        )
    
    def get_risk_status(self) -> Dict[str, Any]:
        """
        Get current risk status and limits.
        
        Returns:
            Status dictionary
        """
        self._reset_daily_limits_if_needed()
        
        return {
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "max_daily_trades": self.max_daily_trades,
            "max_daily_loss": self.max_daily_loss_usd,
            "daily_loss_remaining": self.max_daily_loss_usd + self.daily_pnl,  # Positive is good
            "trades_remaining": self.max_daily_trades - self.daily_trades,
            "last_reset": self.last_reset_date,
            "risk_per_trade_pct": self.risk_per_trade_pct * 100,
            "stop_loss_pct": self.stop_loss_pct * 100,
            "take_profit_pct": self.take_profit_pct * 100
        }
    
    def adjust_for_volatility(self, volatility_multiplier: float):
        """
        Adjust risk parameters based on market volatility.
        
        Args:
            volatility_multiplier: Multiplier for risk parameters (e.g., 1.5 for high volatility)
        """
        # Widen stop loss in high volatility
        self.stop_loss_pct *= volatility_multiplier
        self.take_profit_pct *= volatility_multiplier
        
        logger.info(f"Risk parameters adjusted for volatility ({volatility_multiplier:.2f}x)")
    
    def validate_order_parameters(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate order parameters before execution.
        
        Args:
            symbol: Trading pair
            side: Buy or Sell
            quantity: Order quantity
            price: Order price
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check quantity
        if quantity <= 0:
            return False, f"Invalid quantity: {quantity}"
        
        # Check price
        if price <= 0:
            return False, f"Invalid price: {price}"
        
        # Check order value
        order_value = quantity * price
        if order_value > self.max_position_size_usd * self.max_leverage:
            return False, f"Order value too large: ${order_value:.2f}"
        
        if order_value < 5:  # Minimum order size (Bybit limit)
            return False, f"Order value too small: ${order_value:.2f} (min $5)"
        
        return True, None
