"""
Configuration management for Bybit AI Trader.

Handles environment variables, API credentials, and trading parameters
with support for testnet/mainnet environments.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """Configuration class for trading bot."""
    
    # ==================== API Configuration ====================
    
    # TODO: Set your Bybit API credentials in .env file or environment variables
    # Get your API keys from: https://testnet.bybit.com/app/user/api-management (testnet)
    # or https://www.bybit.com/app/user/api-management (mainnet)
    API_KEY = os.getenv('BYBIT_API_KEY', '')
    API_SECRET = os.getenv('BYBIT_API_SECRET', '')
    
    # Use testnet for testing (set to False for real trading)
    USE_TESTNET = os.getenv('USE_TESTNET', 'True').lower() == 'true'
    
    # ==================== Trading Configuration ====================
    
    # Trading pair
    SYMBOL = os.getenv('TRADING_SYMBOL', 'BTCUSDT')
    
    # Leverage (1-100, be careful with high leverage!)
    LEVERAGE = int(os.getenv('LEVERAGE', '2'))
    
    # Position sizing
    POSITION_SIZE_USD = float(os.getenv('POSITION_SIZE_USD', '100'))  # Fixed position size in USD
    POSITION_SIZE_PCT = float(os.getenv('POSITION_SIZE_PCT', '10'))   # % of account balance
    USE_FIXED_SIZE = os.getenv('USE_FIXED_SIZE', 'False').lower() == 'true'
    
    # Risk management
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '1000'))  # Max position size in USD
    STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '2.0'))  # Stop loss percentage
    TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '4.0'))  # Take profit percentage
    MAX_DAILY_LOSS_PCT = float(os.getenv('MAX_DAILY_LOSS_PCT', '5.0'))  # Max daily loss %
    
    # ==================== ML Model Configuration ====================
    
    # Path to trained model
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/xgb_btcusdt_h4.pkl')
    MODEL_FEATURES_PATH = os.getenv('MODEL_FEATURES_PATH', 'models/xgb_btcusdt_h4_features.json')
    
    # ML strategy parameters
    ML_PREDICTION_THRESHOLD = float(os.getenv('ML_PREDICTION_THRESHOLD', '0.55'))  # Min probability for signal
    USE_TREND_FILTER = os.getenv('USE_TREND_FILTER', 'True').lower() == 'true'  # Require MA trend confirmation
    
    # Feature calculation
    FEATURE_LOOKBACK_HOURS = int(os.getenv('FEATURE_LOOKBACK_HOURS', '200'))  # Candles needed for features
    
    # ==================== Bot Operation ====================
    
    # Trading mode
    PAPER_TRADING = os.getenv('PAPER_TRADING', 'True').lower() == 'true'  # Paper trading mode (no real orders)
    
    # Update intervals
    CHECK_INTERVAL_SECONDS = int(os.getenv('CHECK_INTERVAL_SECONDS', '300'))  # 5 minutes
    DATA_REFRESH_MINUTES = int(os.getenv('DATA_REFRESH_MINUTES', '60'))  # Refresh data every hour
    
    # Maximum open positions
    MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '1'))
    
    # ==================== Logging Configuration ====================
    
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/trading_bot.log')
    LOG_TO_CONSOLE = os.getenv('LOG_TO_CONSOLE', 'True').lower() == 'true'
    
    # ==================== Database Configuration ====================
    
    # SQLite database for trade history
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/trading_history.db')
    
    # ==================== Notification Configuration ====================
    
    # Telegram notifications (optional)
    TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'False').lower() == 'true'
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Email notifications (optional)
    EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'False').lower() == 'true'
    EMAIL_FROM = os.getenv('EMAIL_FROM', '')
    EMAIL_TO = os.getenv('EMAIL_TO', '')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
    EMAIL_SMTP_SERVER = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
    EMAIL_SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', '587'))
    
    # ==================== Validation ====================
    
    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        """
        Validate configuration.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check API credentials
        if not cls.API_KEY or not cls.API_SECRET:
            errors.append("API_KEY and API_SECRET must be set in environment variables or .env file")
        
        # Check leverage
        if cls.LEVERAGE < 1 or cls.LEVERAGE > 100:
            errors.append(f"LEVERAGE must be between 1 and 100, got {cls.LEVERAGE}")
        
        # Check position size
        if cls.POSITION_SIZE_USD <= 0:
            errors.append(f"POSITION_SIZE_USD must be positive, got {cls.POSITION_SIZE_USD}")
        
        if cls.POSITION_SIZE_PCT <= 0 or cls.POSITION_SIZE_PCT > 100:
            errors.append(f"POSITION_SIZE_PCT must be between 0 and 100, got {cls.POSITION_SIZE_PCT}")
        
        # Check risk parameters
        if cls.STOP_LOSS_PCT <= 0:
            errors.append(f"STOP_LOSS_PCT must be positive, got {cls.STOP_LOSS_PCT}")
        
        if cls.TAKE_PROFIT_PCT <= 0:
            errors.append(f"TAKE_PROFIT_PCT must be positive, got {cls.TAKE_PROFIT_PCT}")
        
        # Check ML parameters
        if cls.ML_PREDICTION_THRESHOLD < 0 or cls.ML_PREDICTION_THRESHOLD > 1:
            errors.append(f"ML_PREDICTION_THRESHOLD must be between 0 and 1, got {cls.ML_PREDICTION_THRESHOLD}")
        
        # Check model files exist
        if not Path(cls.MODEL_PATH).exists():
            errors.append(f"Model file not found: {cls.MODEL_PATH}")
        
        if not Path(cls.MODEL_FEATURES_PATH).exists():
            errors.append(f"Model features file not found: {cls.MODEL_FEATURES_PATH}")
        
        # Warn about paper trading
        if not cls.PAPER_TRADING and not cls.USE_TESTNET:
            errors.append("WARNING: Paper trading is disabled and using MAINNET - real money at risk!")
        
        return len(errors) == 0, errors
    
    @classmethod
    def print_config(cls):
        """Print current configuration (hiding sensitive data)."""
        print("=" * 60)
        print("BYBIT AI TRADER CONFIGURATION")
        print("=" * 60)
        print(f"Environment: {'TESTNET' if cls.USE_TESTNET else 'MAINNET'}")
        print(f"Paper Trading: {cls.PAPER_TRADING}")
        print(f"API Key: {cls.API_KEY[:8]}...{cls.API_KEY[-4:] if len(cls.API_KEY) > 12 else '***'}")
        print()
        print(f"Trading Symbol: {cls.SYMBOL}")
        print(f"Leverage: {cls.LEVERAGE}x")
        print(f"Position Size: ${cls.POSITION_SIZE_USD} (fixed)" if cls.USE_FIXED_SIZE else f"Position Size: {cls.POSITION_SIZE_PCT}% of balance")
        print(f"Stop Loss: {cls.STOP_LOSS_PCT}%")
        print(f"Take Profit: {cls.TAKE_PROFIT_PCT}%")
        print()
        print(f"ML Model: {cls.MODEL_PATH}")
        print(f"Prediction Threshold: {cls.ML_PREDICTION_THRESHOLD}")
        print(f"Trend Filter: {cls.USE_TREND_FILTER}")
        print()
        print(f"Check Interval: {cls.CHECK_INTERVAL_SECONDS}s")
        print(f"Max Open Positions: {cls.MAX_OPEN_POSITIONS}")
        print("=" * 60)


# Create config instance
config = Config()


if __name__ == "__main__":
    # Validate configuration
    is_valid, errors = Config.validate()
    
    if is_valid:
        print("✅ Configuration is valid")
        Config.print_config()
    else:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
