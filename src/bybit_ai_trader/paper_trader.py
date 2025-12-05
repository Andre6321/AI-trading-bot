"""
Paper trading (simulated trading) for testing strategies without risk.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid


logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """Represents a simulated open position."""
    position_id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0


@dataclass
class PaperTrade:
    """Represents a completed simulated trade."""
    trade_id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    realized_pnl: float
    fees: float
    reason: str  # "take_profit", "stop_loss", "signal", "manual"


class PaperTrader:
    """
    Simulates trading without using real money.
    
    Maintains a virtual portfolio and executes trades with realistic
    slippage and fees.
    """
    
    def __init__(
        self,
        initial_balance: float = 1000.0,
        fee_rate: float = 0.0006,  # 0.06% taker fee
        slippage_pct: float = 0.0005  # 0.05% slippage
    ):
        """
        Initialize paper trader.
        
        Args:
            initial_balance: Starting balance in USD
            fee_rate: Trading fee rate
            slippage_pct: Slippage percentage
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct
        
        # Trading state
        self.positions: Dict[str, PaperPosition] = {}
        self.trade_history: List[PaperTrade] = []
        self.order_history: List[Dict[str, Any]] = []
        
        logger.info(
            f"PaperTrader initialized: Balance=${initial_balance:.2f}, "
            f"Fee={fee_rate*100:.3f}%, Slippage={slippage_pct*100:.3f}%"
        )
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """
        Apply slippage to order price.
        
        Args:
            price: Original price
            side: "Buy" or "Sell"
            
        Returns:
            Price with slippage
        """
        if side == "Buy":
            # Buy slippage increases price
            return price * (1 + self.slippage_pct)
        else:
            # Sell slippage decreases price
            return price * (1 - self.slippage_pct)
    
    def _calculate_fees(self, order_value: float) -> float:
        """Calculate trading fees."""
        return order_value * self.fee_rate
    
    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        reduce_only: bool = False,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Simulate market order execution.
        
        Args:
            symbol: Trading pair
            side: "Buy" or "Sell"
            quantity: Order quantity
            current_price: Current market price
            reduce_only: Whether order reduces existing position
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order result dictionary
        """
        # Apply slippage
        exec_price = self._apply_slippage(current_price, side)
        
        # Calculate order value
        order_value = quantity * exec_price
        
        # Calculate fees
        fees = self._calculate_fees(order_value)
        
        # Check if this closes existing position
        if reduce_only and symbol in self.positions:
            return self._close_position(symbol, exec_price, fees, "signal")
        
        # Check balance for opening new position
        total_cost = order_value + fees
        if total_cost > self.balance:
            logger.warning(f"Insufficient balance: Need ${total_cost:.2f}, Have ${self.balance:.2f}")
            return {
                "success": False,
                "error": "Insufficient balance",
                "required": total_cost,
                "available": self.balance
            }
        
        # Open new position
        position_id = str(uuid.uuid4())[:8]
        position = PaperPosition(
            position_id=position_id,
            symbol=symbol,
            side=side,
            size=quantity,
            entry_price=exec_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[symbol] = position
        self.balance -= total_cost
        
        # Record order
        order = {
            "order_id": position_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": exec_price,
            "fees": fees,
            "timestamp": datetime.now(),
            "type": "open"
        }
        self.order_history.append(order)
        
        logger.info(
            f"PAPER TRADE OPENED: {side} {quantity} {symbol} @ ${exec_price:.2f} "
            f"(Fees: ${fees:.2f}, Balance: ${self.balance:.2f})"
        )
        
        return {
            "success": True,
            "order_id": position_id,
            "exec_price": exec_price,
            "fees": fees,
            "balance": self.balance
        }
    
    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        fees: float,
        reason: str
    ) -> Dict[str, Any]:
        """
        Close an existing position.
        
        Args:
            symbol: Trading pair
            exit_price: Exit price
            fees: Trading fees
            reason: Reason for closing
            
        Returns:
            Close result dictionary
        """
        if symbol not in self.positions:
            return {"success": False, "error": "No position to close"}
        
        position = self.positions[symbol]
        
        # Calculate PnL
        if position.side == "Buy":
            # Long position
            pnl = (exit_price - position.entry_price) * position.size
        else:
            # Short position
            pnl = (position.entry_price - exit_price) * position.size
        
        # Subtract fees
        realized_pnl = pnl - fees
        
        # Update balance
        position_value = position.size * exit_price
        self.balance += position_value - fees
        
        # Record trade
        trade = PaperTrade(
            trade_id=position.position_id,
            symbol=symbol,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            realized_pnl=realized_pnl,
            fees=fees,
            reason=reason
        )
        self.trade_history.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(
            f"PAPER TRADE CLOSED: {position.side} {position.size} {symbol} "
            f"@ ${exit_price:.2f} | PnL: ${realized_pnl:+.2f} | "
            f"Reason: {reason} | Balance: ${self.balance:.2f}"
        )
        
        return {
            "success": True,
            "trade_id": trade.trade_id,
            "pnl": realized_pnl,
            "exit_price": exit_price,
            "balance": self.balance
        }
    
    def update_positions(self, symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Update position with current price and check SL/TP.
        
        Args:
            symbol: Trading pair
            current_price: Current market price
            
        Returns:
            Close result if position was closed, None otherwise
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Update unrealized PnL
        if position.side == "Buy":
            position.unrealized_pnl = (current_price - position.entry_price) * position.size
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.size
        
        # Check stop loss
        if position.stop_loss:
            if (position.side == "Buy" and current_price <= position.stop_loss) or \
               (position.side == "Sell" and current_price >= position.stop_loss):
                logger.info(f"Stop loss triggered for {symbol} @ ${current_price:.2f}")
                fees = self._calculate_fees(position.size * current_price)
                return self._close_position(symbol, current_price, fees, "stop_loss")
        
        # Check take profit
        if position.take_profit:
            if (position.side == "Buy" and current_price >= position.take_profit) or \
               (position.side == "Sell" and current_price <= position.take_profit):
                logger.info(f"Take profit triggered for {symbol} @ ${current_price:.2f}")
                fees = self._calculate_fees(position.size * current_price)
                return self._close_position(symbol, current_price, fees, "take_profit")
        
        return None
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position information.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Position dictionary or None
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        return {
            "symbol": position.symbol,
            "side": position.side,
            "size": position.size,
            "entry_price": position.entry_price,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "unrealized_pnl": position.unrealized_pnl,
            "entry_time": position.entry_time
        }
    
    def has_position(self, symbol: str) -> bool:
        """Check if position exists."""
        return symbol in self.positions
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        total_value = self.balance
        
        # Add unrealized PnL from open positions
        for position in self.positions.values():
            total_value += position.unrealized_pnl
        
        return {
            "balance": self.balance,
            "equity": total_value,
            "unrealized_pnl": total_value - self.balance,
            "initial_balance": self.initial_balance,
            "total_pnl": total_value - self.initial_balance
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Calculate performance statistics.
        
        Returns:
            Performance metrics dictionary
        """
        if not self.trade_history:
            return {"error": "No trades yet"}
        
        # Calculate metrics
        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t.realized_pnl > 0]
        losing_trades = [t for t in self.trade_history if t.realized_pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t.realized_pnl for t in self.trade_history)
        total_fees = sum(t.fees for t in self.trade_history)
        
        avg_win = sum(t.realized_pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.realized_pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        profit_factor = abs(sum(t.realized_pnl for t in winning_trades) / sum(t.realized_pnl for t in losing_trades)) if losing_trades else float('inf')
        
        balance_info = self.get_balance()
        
        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate * 100,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "net_pnl": total_pnl - total_fees,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "return_pct": (balance_info["equity"] / self.initial_balance - 1) * 100,
            "current_balance": balance_info["balance"],
            "current_equity": balance_info["equity"]
        }
