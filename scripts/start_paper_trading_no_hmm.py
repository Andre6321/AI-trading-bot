"""
Paper Trading Bot - Simplified Version Without HMM Dependency
Uses pre-trained regime models with simplified regime detection
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import signal
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from src.bybit_ai_trader.paper_trader import PaperTrader
from src.bybit_ai_trader.client import BybitClient
from src.bybit_ai_trader.data_fetcher import DataFetcher
from src.bybit_ai_trader.risk_manager import RiskManager


class SimplifiedRegimeDetector:
    """Simplified regime detection using volatility and returns without HMM"""
    
    def __init__(self):
        self.lookback = 24  # 24 hours for regime analysis
        
    def detect_regime(self, df: pd.DataFrame) -> str:
        """
        Detect market regime using volatility and returns
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            'bear', 'sideways', or 'bull'
        """
        if len(df) < self.lookback:
            return 'sideways'
            
        # Calculate metrics over lookback period
        recent = df.tail(self.lookback).copy()
        
        # Price change
        price_change = (recent['close'].iloc[-1] / recent['close'].iloc[0] - 1)
        
        # Volatility (std of returns)
        returns = recent['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Trend strength (linear regression slope)
        x = np.arange(len(recent))
        y = recent['close'].values
        slope = np.polyfit(x, y, 1)[0]
        slope_normalized = slope / recent['close'].mean()
        
        # Regime classification
        # BEAR: Negative trend, higher volatility
        if price_change < -0.02 and slope_normalized < -0.0001:
            return 'bear'
        # BULL: Positive trend, moderate volatility
        elif price_change > 0.02 and slope_normalized > 0.0001:
            return 'bull'
        # SIDEWAYS: Low trend, any volatility
        else:
            return 'sideways'


class SimplifiedMLStrategy:
    """ML strategy with simplified regime detection"""
    
    def __init__(self, data_fetcher: DataFetcher, models_dir: Path, confidence_threshold: float = 0.6):
        self.data_fetcher = data_fetcher
        self.models_dir = models_dir
        self.confidence_threshold = confidence_threshold
        self.regime_detector = SimplifiedRegimeDetector()
        self.models = {}
        self.scalers = {}
        
        # Load regime-specific models
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained regime models"""
        regimes = ['bear', 'sideways', 'bull']
        
        for regime in regimes:
            # Try XGBoost models first (no catboost/lightgbm dependencies)
            xgb_path = self.models_dir / f'xgb_{regime}_regime.pkl'
            ensemble_path = self.models_dir / f'ensemble_{regime}_regime.pkl'
            
            model_path = xgb_path if xgb_path.exists() else ensemble_path
            
            if model_path.exists():
                try:
                    # Try joblib first (more robust)
                    model_data = joblib.load(model_path)
                    
                    # Check if it's a dict or direct model
                    if isinstance(model_data, dict):
                        # XGBoost models stored as {'model': classifier, 'scaler': scaler}
                        if 'xgb' in model_path.name or 'model' in model_data:
                            self.models[regime] = model_data.get('model', model_data.get('xgb'))
                        else:
                            # Ensemble models
                            self.models[regime] = model_data['ensemble']
                        self.scalers[regime] = model_data.get('scaler')
                    else:
                        # Direct model object
                        self.models[regime] = model_data
                        self.scalers[regime] = None
                        
                    print(f"✅ Loaded {regime} regime model ({model_path.name})")
                except Exception as e:
                    print(f"⚠️ Error loading {regime} model: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️ Warning: {regime} regime model not found")
                
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for prediction using enhanced feature builder"""
        try:
            # Use the same feature builder as training
            df_features = build_enhanced_features(df)
            return df_features
        except Exception as e:
            print(f"\u26a0\ufe0f Feature calculation error: {e}")
            return df
        
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Generate trading signal
        
        Returns:
            Dict with 'signal', 'confidence', 'regime', 'features'
        """
        # Detect regime
        regime = self.regime_detector.detect_regime(df)
        
        # Calculate features
        df_features = self.calculate_features(df)
        
        # Get model for current regime
        if regime not in self.models:
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'regime': regime,
                'error': f'No model for {regime} regime'
            }
            
        model = self.models[regime]
        scaler = self.scalers.get(regime)
        
        # Prepare features
        feature_cols = [col for col in df_features.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        latest_features = df_features[feature_cols].iloc[-1:].copy()
        
        # Handle NaN values
        latest_features = latest_features.ffill().bfill().fillna(0)
        
        # Scale features if scaler exists
        if scaler is not None:
            try:
                features_scaled = scaler.transform(latest_features)
            except Exception as e:
                print(f"⚠️ Scaling error: {e}")
                features_scaled = latest_features.values
        else:
            features_scaled = latest_features.values
            
        # Predict
        try:
            pred_proba = model.predict_proba(features_scaled)[0]
            pred_class = model.predict(features_scaled)[0]
            
            # Get confidence (probability of predicted class)
            confidence = pred_proba[pred_class]
            
            # Determine signal
            if pred_class == 1 and confidence >= self.confidence_threshold:
                trade_signal = 'long'
            else:
                trade_signal = 'hold'
                
            return {
                'signal': trade_signal,
                'confidence': float(confidence),
                'regime': regime,
                'pred_proba': pred_proba.tolist(),
                'pred_class': int(pred_class)
            }
            
        except Exception as e:
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'regime': regime,
                'error': str(e)
            }


def main():
    """Main paper trading loop"""
    print("=" * 60)
    print("🤖 STARTING PAPER TRADING BOT (Simplified Regime Detection)")
    print("=" * 60)
    
    # Configuration
    SYMBOL = 'BTCUSDT'
    TIMEFRAME = '1h'
    INITIAL_BALANCE = 10000.0
    MAX_POSITION_SIZE = 500.0
    CONFIDENCE_THRESHOLD = 0.6
    CHECK_INTERVAL = 300  # 5 minutes
    
    # Paths
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / 'models'
    
    # Initialize components
    print("\n📦 Initializing components...")
    
    try:
        paper_trader = PaperTrader(initial_balance=INITIAL_BALANCE)
        
        # Get API credentials from environment
        api_key = os.getenv('BYBIT_API_KEY', 'paper_trading_key')
        api_secret = os.getenv('BYBIT_API_SECRET', 'paper_trading_secret')
        use_testnet = os.getenv('USE_TESTNET', 'true').lower() == 'true'
        
        client = BybitClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=use_testnet
        )
        
        # Convert timeframe to minutes for DataFetcher
        timeframe_map = {'1m': '1', '5m': '5', '15m': '15', '1h': '60', '4h': '240', '1d': 'D'}
        interval = timeframe_map.get(TIMEFRAME, '60')
        
        data_fetcher = DataFetcher(
            client=client,
            symbol=SYMBOL,
            interval=interval,
            buffer_size=200
        )
        
        risk_manager = RiskManager()
        strategy = SimplifiedMLStrategy(
            models_dir=models_dir,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        print("✅ All components initialized")
        print(f"   Mode: {'Testnet' if use_testnet else 'Mainnet'}")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return
        
    # Set up graceful shutdown
    shutdown = {'flag': False}
    
    def signal_handler(sig, frame):
        print("\n\n🛑 Shutdown signal received...")
        shutdown['flag'] = True
        
    signal.signal(signal.SIGINT, signal_handler)
    
    # Main trading loop
    print(f"\n🚀 Starting paper trading...")
    print(f"   Symbol: {SYMBOL}")
    print(f"   Timeframe: {TIMEFRAME}")
    print(f"   Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"   Check Interval: {CHECK_INTERVAL}s")
    print("\n" + "=" * 60)
    
    iteration = 0
    
    while not shutdown['flag']:
        iteration += 1
        print(f"\n{'=' * 60}")
        print(f"📊 Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")
        
        try:
            # Fetch market data
            print("\n📥 Fetching market data...")
            
            # Update buffer and get latest data
            if not data_fetcher.update_buffer():
                print("⚠️ Failed to update data buffer, retrying...")
                time.sleep(CHECK_INTERVAL)
                continue
                
            df = data_fetcher.candle_buffer
            
            if df is None or len(df) == 0:
                print("⚠️ No data in buffer, retrying...")
                time.sleep(CHECK_INTERVAL)
                continue
                
            current_price = float(df['close'].iloc[-1])
            print(f"   Current Price: ${current_price:,.2f}")
            print(f"   Data Points: {len(df)}")
            
            # Generate signal
            print("\n🔮 Generating signal...")
            prediction = strategy.predict(df)
            
            print(f"   Regime: {prediction.get('regime', 'unknown').upper()}")
            print(f"   Signal: {prediction.get('signal', 'hold').upper()}")
            print(f"   Confidence: {prediction.get('confidence', 0):.2%}")
            
            if 'error' in prediction:
                print(f"   ⚠️ Error: {prediction['error']}")
            
            # Execute trades
            position = paper_trader.get_position(SYMBOL)
            trade_signal = prediction.get('signal', 'hold')
            confidence = prediction.get('confidence', 0)
            
            if trade_signal == 'long' and position is None:
                # Calculate position size
                position_size = min(MAX_POSITION_SIZE, paper_trader.balance * 0.05)
                quantity = position_size / current_price
                
                # Calculate SL/TP
                sl_price = current_price * 0.98  # 2% SL
                tp_price = current_price * 1.04  # 4% TP
                
                # Place order
                order = paper_trader.place_market_order(
                    symbol=SYMBOL,
                    side='buy',
                    size=quantity,
                    current_price=current_price,
                    stop_loss=sl_price,
                    take_profit=tp_price
                )
                
                if order:
                    
                    print(f"\n✅ LONG POSITION OPENED")
                    print(f"   Entry: ${current_price:,.2f}")
                    print(f"   Quantity: {quantity:.4f}")
                    print(f"   Position Size: ${position_size:.2f}")
                    print(f"   Stop Loss: ${sl_price:,.2f} (-2%)")
                    print(f"   Take Profit: ${tp_price:,.2f} (+4%)")
                    
            elif position is not None:
                # Update position with current price (checks SL/TP automatically)
                closed_trade = paper_trader.update_positions(SYMBOL, current_price)
                
                if closed_trade:
                    pnl = closed_trade['realized_pnl']
                    reason = closed_trade['reason']
                    
                    if reason == 'stop_loss':
                        print(f"\n❌ STOP LOSS HIT")
                        print(f"   Exit: ${current_price:,.2f}")
                        print(f"   PnL: ${pnl:,.2f}")
                    elif reason == 'take_profit':
                        print(f"\n✅ TAKE PROFIT HIT")
                        print(f"   Exit: ${current_price:,.2f}")
                        print(f"   PnL: ${pnl:,.2f}")
            
            # Display portfolio status
            balance_info = paper_trader.get_balance()
            print(f"\n💼 Portfolio Status:")
            print(f"   Balance: ${balance_info['balance']:,.2f}")
            print(f"   Equity: ${balance_info['equity']:,.2f}")
            
            if position:
                unrealized_pnl = balance_info['unrealized_pnl']
                print(f"   Position: LONG {position['size']:.4f} @ ${position['entry_price']:,.2f}")
                print(f"   Unrealized PnL: ${unrealized_pnl:,.2f}")
                
            # Display stats
            if len(paper_trader.trade_history) > 0:
                stats = paper_trader.get_performance_stats()
                print(f"\n📈 Performance:")
                print(f"   Total Trades: {stats['total_trades']}")
                print(f"   Wins: {stats['winning_trades']} ({stats['win_rate']:.1f}%)")
                print(f"   Losses: {stats['losing_trades']}")
                print(f"   Profit Factor: {stats['profit_factor']:.2f}")
                print(f"   Return: {stats['return_pct']:.2f}%")
            
        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            
        # Wait before next check
        if not shutdown['flag']:
            print(f"\n⏱️ Waiting {CHECK_INTERVAL}s until next check...")
            time.sleep(CHECK_INTERVAL)
    
    # Final statistics
    print("\n" + "=" * 60)
    print("📊 FINAL STATISTICS")
    print("=" * 60)
    
    balance_info = paper_trader.get_balance()
    print(f"\nBalance: ${balance_info['balance']:,.2f}")
    print(f"Equity: ${balance_info['equity']:,.2f}")
    
    if len(paper_trader.trade_history) > 0:
        stats = paper_trader.get_performance_stats()
        print(f"Total PnL: ${stats['total_pnl']:,.2f}")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Total Return: {stats['return_pct']:.2f}%")
    else:
        print("No trades executed")
    
    print("\n✅ Paper trading session ended")


if __name__ == '__main__':
    main()
