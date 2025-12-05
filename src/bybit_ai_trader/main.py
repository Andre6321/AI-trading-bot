"""
Main trading bot orchestrator.

Coordinates all components to execute the ML trading strategy.
"""

import logging
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Import bot components
from .client import BybitClient
from .paper_trader import PaperTrader
from .data_fetcher import DataFetcher
from .risk_manager import RiskManager
from .strategies.ml_strategy import MLStrategy
from .utils.logger import setup_logger
from .utils.notifications import NotificationManager


# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    global running
    logger = logging.getLogger(__name__)
    logger.info("Shutdown signal received, stopping bot...")
    running = False


class TradingBot:
    """
    Main trading bot that orchestrates all components.
    """
    
    def __init__(self, config: dict):
        """
        Initialize trading bot.
        
        Args:
            config: Configuration dictionary from config_new.py
        """
        self.config = config
        
        # Setup logging
        self.logger = setup_logger(
            log_level=config.get("log_level", "INFO"),
            log_dir=config.get("log_dir", "logs")
        )
        
        # Initialize notification system
        self.notifier = NotificationManager(
            telegram_token=config.get("telegram_token"),
            telegram_chat_id=config.get("telegram_chat_id"),
            email_enabled=config.get("email_enabled", False)
        )
        
        # Trading parameters
        self.symbol = config["symbol"]
        self.interval = config["interval"]
        self.paper_trading = config["paper_trading"]
        self.check_interval_seconds = config.get("check_interval_seconds", 300)  # 5 minutes
        
        # Initialize components
        self._initialize_components()
        
        # State
        self.last_candle_timestamp = None
        self.last_signal = None
        
        self.logger.info("=" * 60)
        self.logger.info("Trading Bot Initialized")
        self.logger.info(f"Mode: {'PAPER TRADING' if self.paper_trading else 'LIVE TRADING'}")
        self.logger.info(f"Symbol: {self.symbol}")
        self.logger.info(f"Interval: {self.interval} minutes")
        self.logger.info("=" * 60)
    
    def _initialize_components(self):
        """Initialize all bot components."""
        self.logger.info("Initializing bot components...")
        
        # Initialize trading client or paper trader
        if self.paper_trading:
            self.trader = PaperTrader(
                initial_balance=self.config.get("paper_balance", 1000.0),
                fee_rate=self.config.get("fee_rate", 0.0006),
                slippage_pct=self.config.get("slippage_pct", 0.0005)
            )
            # For data fetching, still need real client
            self.client = BybitClient(
                api_key=self.config["api_key"],
                api_secret=self.config["api_secret"],
                testnet=self.config["use_testnet"]
            )
        else:
            self.client = BybitClient(
                api_key=self.config["api_key"],
                api_secret=self.config["api_secret"],
                testnet=self.config["use_testnet"]
            )
            self.trader = self.client
            
            # Set leverage for live trading
            self.client.set_leverage(self.symbol, self.config.get("leverage", 1))
        
        # Initialize data fetcher
        self.data_fetcher = DataFetcher(
            client=self.client,
            symbol=self.symbol,
            interval=self.interval
        )
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            max_position_size_usd=self.config.get("max_position_size_usd", 100.0),
            max_leverage=self.config.get("leverage", 1),
            stop_loss_pct=self.config.get("stop_loss_pct", 0.02),
            take_profit_pct=self.config.get("take_profit_pct", 0.04),
            max_daily_loss_usd=self.config.get("max_daily_loss_usd", 50.0),
            max_daily_trades=self.config.get("max_daily_trades", 10),
            risk_per_trade_pct=self.config.get("risk_per_trade_pct", 0.01)
        )
        
        # Initialize ML strategy
        model_path = self.config.get("model_path", "models/xgb_btcusdt_h4.pkl")
        metadata_path = self.config.get("metadata_path", "models/xgb_btcusdt_h4_features.json")
        
        self.strategy = MLStrategy(
            model_path=model_path,
            metadata_path=metadata_path,
            confidence_threshold=self.config.get("confidence_threshold", 0.6),
            use_trend_filter=self.config.get("use_trend_filter", True)
        )
        
        self.logger.info("All components initialized successfully")
    
    def _get_current_position(self) -> Optional[dict]:
        """Get current position information."""
        if self.paper_trading:
            return self.trader.get_position(self.symbol)
        else:
            return self.client.get_position(self.symbol)
    
    def _has_position(self) -> bool:
        """Check if we have an open position."""
        if self.paper_trading:
            return self.trader.has_position(self.symbol)
        else:
            return self.client.has_position(self.symbol)
    
    def _get_account_balance(self) -> float:
        """Get account balance."""
        if self.paper_trading:
            balance_info = self.trader.get_balance()
            return balance_info["balance"]
        else:
            balance_info = self.client.get_balance(coin="USDT")
            return balance_info["available"]
    
    def _execute_trade(self, signal: dict, current_price: float):
        """
        Execute trade based on signal.
        
        Args:
            signal: Trading signal from strategy
            current_price: Current market price
        """
        action = signal["action"]
        
        if action == "HOLD":
            return
        
        if action == "CLOSE":
            self._close_position(current_price, signal)
        
        elif action == "ENTER":
            self._open_position(signal, current_price)
    
    def _open_position(self, signal: dict, current_price: float):
        """Open a new position."""
        side = signal["side"]
        confidence = signal["confidence"]
        
        # Check if we can trade
        balance = self._get_account_balance()
        can_trade, reason = self.risk_manager.can_open_new_position(balance)
        
        if not can_trade:
            self.logger.warning(f"Cannot open position: {reason}")
            return
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            account_balance=balance,
            entry_price=current_price,
            side=side,
            signal_confidence=confidence
        )
        
        # Calculate SL/TP
        stop_loss, take_profit = self.risk_manager.calculate_stop_loss_take_profit(
            entry_price=current_price,
            side=side
        )
        
        # Validate order
        is_valid, error = self.risk_manager.validate_order_parameters(
            symbol=self.symbol,
            side=side,
            quantity=position_size,
            price=current_price
        )
        
        if not is_valid:
            self.logger.error(f"Invalid order parameters: {error}")
            return
        
        # Execute order
        self.logger.info(
            f"OPENING POSITION: {side} {position_size} {self.symbol} @ ${current_price:.2f}"
        )
        
        if self.paper_trading:
            result = self.trader.place_market_order(
                symbol=self.symbol,
                side=side,
                quantity=position_size,
                current_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        else:
            order_id = self.client.place_market_order(
                symbol=self.symbol,
                side=side,
                quantity=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            result = {"success": bool(order_id)}
        
        if result.get("success"):
            self.logger.info(f"Position opened successfully")
            self.notifier.notify_trade("OPEN", self.symbol, side, position_size, current_price)
        else:
            self.logger.error(f"Failed to open position: {result.get('error')}")
            self.notifier.notify_error(f"Failed to open position: {result.get('error')}")
    
    def _close_position(self, current_price: float, signal: dict):
        """Close existing position."""
        position = self._get_current_position()
        
        if not position:
            self.logger.warning("No position to close")
            return
        
        self.logger.info(f"CLOSING POSITION: {position['side']} {position['size']} {self.symbol}")
        
        if self.paper_trading:
            fees = self.trader._calculate_fees(position['size'] * current_price)
            result = self.trader._close_position(
                symbol=self.symbol,
                exit_price=current_price,
                fees=fees,
                reason="signal"
            )
        else:
            success = self.client.close_position(self.symbol)
            result = {"success": success}
        
        if result.get("success"):
            pnl = result.get("pnl", 0)
            self.logger.info(f"Position closed successfully: PnL = ${pnl:+.2f}")
            self.notifier.notify_trade("CLOSE", self.symbol, position['side'], position['size'], current_price)
            
            # Record trade in risk manager
            self.risk_manager.record_trade(pnl)
        else:
            self.logger.error(f"Failed to close position")
            self.notifier.notify_error("Failed to close position")
    
    def _update_positions(self, current_price: float):
        """Update positions with current price (check SL/TP for paper trading)."""
        if self.paper_trading and self._has_position():
            result = self.trader.update_positions(self.symbol, current_price)
            if result and result.get("success"):
                pnl = result.get("pnl", 0)
                self.logger.info(f"Position auto-closed: PnL = ${pnl:+.2f}")
                self.risk_manager.record_trade(pnl)
    
    def _print_status(self):
        """Print current bot status."""
        self.logger.info("=" * 60)
        self.logger.info("BOT STATUS")
        self.logger.info("=" * 60)
        
        # Account info
        if self.paper_trading:
            balance_info = self.trader.get_balance()
            self.logger.info(f"Balance: ${balance_info['balance']:.2f}")
            self.logger.info(f"Equity: ${balance_info['equity']:.2f}")
            self.logger.info(f"Total PnL: ${balance_info['total_pnl']:+.2f}")
        else:
            balance_info = self.client.get_balance(coin="USDT")
            self.logger.info(f"Balance: ${balance_info['wallet_balance']:.2f}")
            self.logger.info(f"Available: ${balance_info['available']:.2f}")
        
        # Position info
        position = self._get_current_position()
        if position:
            self.logger.info(f"Position: {position['side']} {position['size']} @ ${position['entry_price']:.2f}")
            self.logger.info(f"Unrealized PnL: ${position['unrealized_pnl']:+.2f}")
        else:
            self.logger.info("Position: None")
        
        # Risk status
        risk_status = self.risk_manager.get_risk_status()
        self.logger.info(f"Daily Trades: {risk_status['daily_trades']}/{risk_status['max_daily_trades']}")
        self.logger.info(f"Daily PnL: ${risk_status['daily_pnl']:+.2f}")
        
        self.logger.info("=" * 60)
    
    def run(self):
        """Main bot loop."""
        global running
        
        self.logger.info("Starting bot main loop...")
        self.notifier.notify_bot_status("STARTED", f"Trading {self.symbol}")
        
        try:
            while running:
                try:
                    # Get current price
                    current_price = self.data_fetcher.get_current_price()
                    
                    # Update positions (for paper trading SL/TP)
                    self._update_positions(current_price)
                    
                    # Get latest features
                    features = self.data_fetcher.get_latest_features()
                    
                    if not features:
                        self.logger.warning("Failed to get features, waiting for next cycle")
                        time.sleep(self.check_interval_seconds)
                        continue
                    
                    # Get current position
                    position = self._get_current_position()
                    current_position_side = position["side"] if position else None
                    
                    # Generate trading signal
                    signal = self.strategy.generate_signal(features, current_position_side)
                    
                    self.logger.info(
                        f"Signal: {signal['action']} | "
                        f"Prediction: {signal['prediction']} @ {signal['confidence']:.3f} | "
                        f"Price: ${current_price:.2f}"
                    )
                    
                    # Execute trade if needed
                    self._execute_trade(signal, current_price)
                    
                    # Print status periodically
                    self._print_status()
                    
                    # Wait for next check
                    self.logger.info(f"Waiting {self.check_interval_seconds}s until next check...")
                    time.sleep(self.check_interval_seconds)
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}", exc_info=True)
                    self.notifier.notify_error(f"Bot error: {str(e)}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        
        finally:
            self._shutdown()
    
    def _shutdown(self):
        """Graceful shutdown."""
        self.logger.info("Shutting down bot...")
        
        # Print final stats
        if self.paper_trading:
            try:
                stats = self.trader.get_performance_stats()
                self.logger.info("=" * 60)
                self.logger.info("FINAL PERFORMANCE STATS")
                self.logger.info("=" * 60)
                for key, value in stats.items():
                    self.logger.info(f"{key}: {value}")
                self.logger.info("=" * 60)
                
                self.notifier.notify_daily_summary(stats)
            except:
                pass
        
        self.notifier.notify_bot_status("STOPPED", "Bot shut down gracefully")
        self.logger.info("Bot stopped")


def main():
    """Main entry point."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    from .config_new import Config
    config = Config()
    
    # Convert Config class attributes to dictionary for TradingBot
    config_dict = {
        "api_key": config.API_KEY,
        "api_secret": config.API_SECRET,
        "use_testnet": config.USE_TESTNET,
        "symbol": config.SYMBOL,
        "interval": "240",  # 4-hour candles
        "leverage": config.LEVERAGE,
        "max_position_size_usd": config.MAX_POSITION_SIZE,
        "stop_loss_pct": config.STOP_LOSS_PCT / 100,  # Convert to decimal
        "take_profit_pct": config.TAKE_PROFIT_PCT / 100,
        "max_daily_loss_usd": config.MAX_POSITION_SIZE * (config.MAX_DAILY_LOSS_PCT / 100),
        "max_daily_trades": 10,
        "risk_per_trade_pct": 0.01,
        "model_path": config.MODEL_PATH,
        "metadata_path": config.MODEL_FEATURES_PATH,
        "confidence_threshold": config.ML_PREDICTION_THRESHOLD,
        "use_trend_filter": config.USE_TREND_FILTER,
        "paper_trading": config.PAPER_TRADING,
        "paper_balance": 1000.0,
        "fee_rate": 0.0006,
        "slippage_pct": 0.0005,
        "check_interval_seconds": config.CHECK_INTERVAL_SECONDS,
        "log_level": config.LOG_LEVEL,
        "log_dir": "logs",
        "telegram_token": config.TELEGRAM_BOT_TOKEN if config.TELEGRAM_ENABLED else None,
        "telegram_chat_id": config.TELEGRAM_CHAT_ID if config.TELEGRAM_ENABLED else None,
        "email_enabled": config.EMAIL_ENABLED
    }
    
    # Create and run bot
    bot = TradingBot(config_dict)
    bot.run()


if __name__ == "__main__":
    main()
