"""
Paper Trading Bot - Regime-Aware ML Strategy

This script runs the trading bot in paper trading mode using:
- RegimeAwareMLStrategy with HMM regime detection
- Regime-specific models (BEAR, SIDEWAYS, BULL)
- Fixed bugs from code review
- Real-time Bybit data
- Simulated trading with realistic fees/slippage

Usage:
    python scripts/start_paper_trading.py
"""

import sys
import signal
import time
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bybit_ai_trader.client import BybitClient
from bybit_ai_trader.paper_trader import PaperTrader
from bybit_ai_trader.data_fetcher import DataFetcher
from bybit_ai_trader.risk_manager import RiskManager
from bybit_ai_trader.strategies.regime_aware_strategy import RegimeAwareMLStrategy
from bybit_ai_trader.utils.logger import setup_logger

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C for graceful shutdown."""
    global running
    print("\n\n⚠️  Shutdown signal received...")
    running = False

signal.signal(signal.SIGINT, signal_handler)


def display_banner():
    """Display startup banner."""
    print("=" * 80)
    print(" " * 20 + "BYBIT AI TRADING BOT - PAPER TRADING MODE")
    print("=" * 80)
    print()
    print("  📊 Strategy: Regime-Aware ML (HMM + Ensemble Models)")
    print("  💰 Mode: PAPER TRADING (Simulated)")
    print("  🎯 Symbol: BTCUSDT")
    print("  ⏱️  Interval: Check every 5 minutes")
    print()
    print("  Models:")
    print("    🐻 BEAR Regime: 0.7323 ROC AUC")
    print("    ↔️  SIDEWAYS Regime: 0.7494 ROC AUC")
    print("    🐂 BULL Regime: 0.7626 ROC AUC")
    print()
    print("=" * 80)
    print()


def main():
    """Main paper trading loop."""
    
    # Display banner
    display_banner()
    
    # Setup logging
    logger = setup_logger(log_level="INFO", log_dir="logs")
    logger.info("Starting Paper Trading Bot...")
    
    # Configuration
    config = {
        "symbol": "BTCUSDT",
        "interval": 60,  # 1-hour candles
        "paper_balance": 10000.0,  # Start with $10k
        "fee_rate": 0.0006,  # 0.06% Bybit taker fee
        "slippage_pct": 0.0005,  # 0.05% slippage
        "check_interval_seconds": 300,  # Check every 5 minutes
        "confidence_threshold": 0.6,  # 60% minimum confidence
        "max_position_size_usd": 500.0,  # Max $500 per trade
        "stop_loss_pct": 0.02,  # 2% stop loss
        "take_profit_pct": 0.04,  # 4% take profit
        "risk_per_trade_pct": 0.02,  # Risk 2% per trade
    }
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        
        # Paper trader
        paper_trader = PaperTrader(
            initial_balance=config["paper_balance"],
            fee_rate=config["fee_rate"],
            slippage_pct=config["slippage_pct"]
        )
        
        # Bybit client (for real market data)
        client = BybitClient(
            api_key="",  # Not needed for public data
            api_secret="",
            testnet=False  # Use mainnet for real prices
        )
        
        # Data fetcher
        data_fetcher = DataFetcher(
            client=client,
            symbol=config["symbol"],
            interval=config["interval"],
            buffer_size=200  # Need enough for regime detection (168+ hours)
        )
        
        # Risk manager
        risk_manager = RiskManager(
            max_position_size_usd=config["max_position_size_usd"],
            max_leverage=1,
            stop_loss_pct=config["stop_loss_pct"],
            take_profit_pct=config["take_profit_pct"],
            risk_per_trade_pct=config["risk_per_trade_pct"]
        )
        
        # Regime-aware ML strategy
        strategy = RegimeAwareMLStrategy(
            regime_models_dir="models",
            hmm_model_path="models/market_regime_hmm.pkl",
            confidence_threshold=config["confidence_threshold"],
            use_trend_filter=True,
            regime_switch_delay=3  # 3-hour delay before switching regimes
        )
        
        logger.info("✅ All components initialized")
        
        # Display initial state
        balance = paper_trader.get_balance()
        print(f"\n💰 Initial Balance: ${balance['balance']:,.2f}")
        print(f"🎯 Max Position Size: ${config['max_position_size_usd']:,.2f}")
        print(f"🛡️  Stop Loss: {config['stop_loss_pct']*100}%")
        print(f"🎁 Take Profit: {config['take_profit_pct']*100}%")
        print()
        print("🚀 Bot started! Press Ctrl+C to stop.")
        print("=" * 80)
        print()
        
        iteration = 0
        
        # Main trading loop
        while running:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*60}")
            
            try:
                # Update market data
                logger.info("Fetching latest market data...")
                data_fetcher.update()
                
                # Get recent data for regime detection
                recent_data = data_fetcher.get_recent_data(200)  # Last 200 candles
                if recent_data is None or len(recent_data) < 50:
                    logger.warning("Insufficient data, waiting...")
                    time.sleep(config["check_interval_seconds"])
                    continue
                
                current_price = recent_data['close'].iloc[-1]
                logger.info(f"📈 Current Price: ${current_price:,.2f}")
                
                # Get latest features
                features = data_fetcher.get_latest_features()
                if not features:
                    logger.warning("No features available, waiting...")
                    time.sleep(config["check_interval_seconds"])
                    continue
                
                # Check existing position
                current_position = paper_trader.get_position(config["symbol"])
                
                # Update position with current price (check SL/TP)
                if current_position:
                    close_result = paper_trader.update_positions(config["symbol"], current_price)
                    if close_result:
                        logger.info(f"Position closed by SL/TP: {close_result}")
                        current_position = None
                
                # Generate trading signal
                logger.info("Generating trading signal...")
                signal = strategy.generate_signal(
                    features=features,
                    recent_data=recent_data,
                    current_position=current_position['side'] if current_position else None
                )
                
                # Log signal
                logger.info(f"📊 Signal: {signal['action']}")
                logger.info(f"   Regime: {signal['regime'].upper()} (conf: {signal['regime_confidence']:.3f})")
                logger.info(f"   Prediction: {signal['prediction']} (conf: {signal['confidence']:.3f})")
                logger.info(f"   Reasoning: {', '.join(signal['reasoning'])}")
                
                # Execute trade based on signal
                if signal['action'] == 'CLOSE':
                    if current_position:
                        fees = paper_trader._calculate_fees(current_position['size'] * current_price)
                        result = paper_trader._close_position(
                            symbol=config["symbol"],
                            exit_price=current_price,
                            fees=fees,
                            reason="signal"
                        )
                        logger.info(f"✅ Position closed: PnL ${result['pnl']:+.2f}")
                
                elif signal['action'] == 'ENTER':
                    # Calculate position size
                    balance_info = paper_trader.get_balance()
                    position_size_usd = min(
                        config["max_position_size_usd"],
                        balance_info['balance'] * config['risk_per_trade_pct']
                    )
                    position_size = position_size_usd / current_price
                    
                    # Calculate SL/TP
                    atr_pct = features.get('atr_pct')
                    volatility_regime = features.get('volatility_regime', 'normal')
                    
                    stop_loss, take_profit = risk_manager.calculate_stop_loss_take_profit(
                        entry_price=current_price,
                        side=signal['side'],
                        atr_pct=atr_pct,
                        volatility_regime=volatility_regime,
                        market_regime=signal['regime']  # Pass regime for regime-specific SL/TP
                    )
                    
                    # Place order
                    result = paper_trader.place_market_order(
                        symbol=config["symbol"],
                        side=signal['side'],
                        quantity=position_size,
                        current_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    if result['success']:
                        logger.info(f"✅ Position opened: {signal['side']} {position_size:.4f} BTC")
                        logger.info(f"   Entry: ${result['exec_price']:,.2f}")
                        logger.info(f"   SL: ${stop_loss:,.2f} ({((stop_loss/current_price-1)*100):+.2f}%)")
                        logger.info(f"   TP: ${take_profit:,.2f} ({((take_profit/current_price-1)*100):+.2f}%)")
                    else:
                        logger.warning(f"❌ Failed to open position: {result.get('error')}")
                
                # Display performance stats
                balance = paper_trader.get_balance()
                logger.info(f"\n💰 Account Status:")
                logger.info(f"   Balance: ${balance['balance']:,.2f}")
                logger.info(f"   Equity: ${balance['equity']:,.2f}")
                logger.info(f"   Total PnL: ${balance['total_pnl']:+,.2f} ({(balance['total_pnl']/config['paper_balance']*100):+.2f}%)")
                
                if len(paper_trader.trade_history) > 0:
                    stats = paper_trader.get_performance_stats()
                    logger.info(f"\n📊 Performance Stats:")
                    logger.info(f"   Total Trades: {stats['total_trades']}")
                    logger.info(f"   Win Rate: {stats['win_rate']:.1f}%")
                    logger.info(f"   Profit Factor: {stats['profit_factor']:.2f}")
                    logger.info(f"   Net PnL: ${stats['net_pnl']:+,.2f}")
                    logger.info(f"   Return: {stats['return_pct']:+.2f}%")
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                import traceback
                traceback.print_exc()
            
            # Wait before next iteration
            if running:
                logger.info(f"\n⏳ Waiting {config['check_interval_seconds']} seconds...")
                time.sleep(config['check_interval_seconds'])
        
        # Shutdown
        logger.info("\n" + "="*60)
        logger.info("SHUTTING DOWN")
        logger.info("="*60)
        
        # Final stats
        balance = paper_trader.get_balance()
        print(f"\n📊 FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Initial Balance: ${config['paper_balance']:,.2f}")
        print(f"Final Equity: ${balance['equity']:,.2f}")
        print(f"Total PnL: ${balance['total_pnl']:+,.2f} ({(balance['total_pnl']/config['paper_balance']*100):+.2f}%)")
        
        if len(paper_trader.trade_history) > 0:
            stats = paper_trader.get_performance_stats()
            print(f"\nTotal Trades: {stats['total_trades']}")
            print(f"Win Rate: {stats['win_rate']:.1f}%")
            print(f"Profit Factor: {stats['profit_factor']:.2f}")
            print(f"Avg Win: ${stats['avg_win']:+.2f}")
            print(f"Avg Loss: ${stats['avg_loss']:+.2f}")
        
        print(f"{'='*60}\n")
        logger.info("Bot stopped gracefully")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
