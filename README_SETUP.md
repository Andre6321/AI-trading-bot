# Bybit AI Trading Bot - Complete Setup Guide

## 🎉 Implementation Complete!

All components have been implemented. The bot is ready for testing in paper trading mode.

## 📋 What's Been Implemented

### ✅ Core Trading Infrastructure
- **Bybit V5 API Client** (`api.py`) - HMAC authentication, order management, position queries
- **High-Level Trading Client** (`client.py`) - Simplified trading methods with error handling
- **Paper Trading System** (`paper_trader.py`) - Risk-free testing with realistic simulation

### ✅ Strategy & Analysis
- **ML Strategy** (`strategies/ml_strategy.py`) - XGBoost model integration for live trading
- **Data Fetcher** (`data_fetcher.py`) - Real-time data pipeline with feature calculation
- **Feature Engineering** - 27 technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Trained Model** - XGBoost classifier (56% accuracy on test set)

### ✅ Risk Management
- **Risk Manager** (`risk_manager.py`) - Position sizing, SL/TP calculation, exposure limits
- **Daily Limits** - Maximum loss per day, maximum trades per day
- **Portfolio Protection** - Maximum exposure percentage, leverage limits

### ✅ Operations & Monitoring
- **Logging System** (`utils/logger.py`) - Structured logging with file rotation
- **Notifications** (`utils/notifications.py`) - Telegram and email alerts
- **Configuration** (`config_new.py`) - Environment-based config management
- **Main Bot** (`main.py`) - Complete orchestration of all components

## 🚀 Quick Start

### Step 1: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 2: Configure Environment

```powershell
# Copy the example environment file
Copy-Item .env.example .env

# Edit .env with your settings
notepad .env
```

**Important Configuration Items:**
```bash
# Bybit API credentials (get from https://testnet.bybit.com/app/user/api-management)
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here

# Trading mode (CRITICAL: Start with paper trading!)
PAPER_TRADING=true
USE_TESTNET=true

# Risk parameters (adjust based on your risk tolerance)
MAX_POSITION_SIZE_USD=100.0
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.04
```

### Step 3: Verify Model Files

Check that you have the trained model:
```powershell
# Should exist from previous training
Get-ChildItem models/

# Expected files:
# - xgb_btcusdt_h4.pkl
# - xgb_btcusdt_h4_features.json
```

If missing, train the model:
```powershell
python scripts/train_xgboost_simple.py
```

### Step 4: Run the Bot (Paper Trading Mode)

```powershell
python scripts/run_bot.py
```

## 📊 Bot Operation

### What the Bot Does

1. **Initialization**
   - Loads trained XGBoost model
   - Connects to Bybit (testnet or mainnet)
   - Initializes paper trader or live trader
   - Sets up risk management and monitoring

2. **Main Loop** (Every 5 minutes by default)
   - Fetches latest market data
   - Calculates 27 technical features
   - Generates ML prediction (UP/DOWN)
   - Checks risk limits
   - Executes trade if signal is strong enough
   - Monitors open positions for SL/TP
   - Logs all actions

3. **Position Management**
   - Automatically sets stop-loss and take-profit
   - Monitors for trend changes
   - Closes positions when signal reverses
   - Tracks daily PnL and trade count

4. **Risk Controls**
   - Position sizing based on account balance
   - Maximum daily loss limit
   - Maximum number of trades per day
   - Portfolio exposure limits

## 🔧 Configuration Options

### Trading Parameters (in .env)

```bash
# Symbol and timeframe
SYMBOL=BTCUSDT
INTERVAL=240  # 4-hour candles

# Position sizing
MAX_POSITION_SIZE_USD=100.0
LEVERAGE=1
RISK_PER_TRADE_PCT=0.01  # 1% of account per trade

# Risk management
STOP_LOSS_PCT=0.02       # 2% stop loss
TAKE_PROFIT_PCT=0.04     # 4% take profit (2:1 R:R)
MAX_DAILY_LOSS_USD=50.0
MAX_DAILY_TRADES=10

# Strategy
CONFIDENCE_THRESHOLD=0.6  # Minimum ML confidence to trade
USE_TREND_FILTER=true     # Only trade with the trend

# Logging
LOG_LEVEL=INFO
CHECK_INTERVAL_SECONDS=300  # Check every 5 minutes
```

### Paper Trading Parameters

```bash
PAPER_TRADING=true
PAPER_BALANCE=1000.0
FEE_RATE=0.0006           # 0.06% taker fee
SLIPPAGE_PCT=0.0005       # 0.05% slippage
```

## 📈 Monitoring & Logs

### Log Files (in `logs/` directory)

- `trading_bot_YYYYMMDD.log` - Detailed bot logs
- `trades_YYYYMMDD.log` - Trade journal (entries and exits)

### Real-Time Monitoring

The bot prints status updates every cycle:
```
============================================================
BOT STATUS
============================================================
Balance: $1000.00
Equity: $1015.50
Total PnL: +$15.50
Position: Buy 0.001 @ $45000.00
Unrealized PnL: +$15.50
Daily Trades: 3/10
Daily PnL: +$15.50
============================================================
```

### Telegram Notifications (Optional)

If configured, you'll receive:
- 🤖 Trade open/close notifications
- ⚠️ Error alerts
- ✅ Bot start/stop notifications
- 📊 Daily performance summaries

## ⚠️ Safety Checklist

Before moving to live trading, ensure you've:

- [ ] **Tested in paper trading mode for at least 1-2 weeks**
- [ ] **Reviewed all trades in paper mode and understand the strategy**
- [ ] **Tested on testnet with small amounts ($10-20)**
- [ ] **Verified stop-loss triggers correctly**
- [ ] **Confirmed daily loss limits work**
- [ ] **Set up monitoring and alerts**
- [ ] **Have an emergency stop procedure**
- [ ] **Start with VERY SMALL position sizes**
- [ ] **Use leverage=1 initially**
- [ ] **Monitor closely for first week**

## 🐛 Troubleshooting

### Bot won't start

```powershell
# Check Python environment
python --version  # Should be 3.8+

# Check dependencies
pip list | Select-String "xgboost|pandas|numpy|requests"

# Check configuration
Get-Content .env | Select-String "API_KEY"
```

### API Authentication Errors

1. Verify API keys are correct
2. Check API permissions (need "Orders" and "Positions")
3. Ensure USE_TESTNET matches your API keys (testnet keys won't work on mainnet)

### No Trading Signals

- Check that features are being calculated correctly
- Verify model file exists and loads
- Lower CONFIDENCE_THRESHOLD if too restrictive
- Check USE_TREND_FILTER - may be filtering out signals

### Model Loading Errors

```powershell
# Re-train the model
python scripts/train_xgboost_simple.py

# Verify model file
Get-ChildItem models/ -Filter "*.pkl"
```

## 📁 File Structure

```
bybit_ai_trader/
├── .env                                    # Your configuration (DO NOT COMMIT!)
├── .env.example                            # Configuration template
├── README_SETUP.md                         # This file
├── IMPLEMENTATION.md                       # Implementation details
│
├── src/bybit_ai_trader/
│   ├── api.py                             # ✅ Bybit V5 API client
│   ├── client.py                          # ✅ High-level trading client
│   ├── config_new.py                      # ✅ Configuration management
│   ├── data_fetcher.py                    # ✅ Real-time data pipeline
│   ├── main.py                            # ✅ Main bot orchestrator
│   ├── paper_trader.py                    # ✅ Paper trading simulator
│   ├── risk_manager.py                    # ✅ Risk management system
│   │
│   ├── strategies/
│   │   └── ml_strategy.py                 # ✅ ML trading strategy
│   │
│   ├── research/
│   │   ├── features.py                    # ✅ Feature engineering
│   │   └── ml_dataset.py                  # ✅ ML dataset builder
│   │
│   └── utils/
│       ├── logger.py                      # ✅ Logging system
│       └── notifications.py               # ✅ Notification system
│
├── scripts/
│   ├── run_bot.py                         # ✅ Bot launcher
│   ├── train_xgboost_simple.py            # ✅ Model training
│   └── build_features.py                  # ✅ Feature generation
│
├── models/
│   ├── xgb_btcusdt_h4.pkl                # ✅ Trained model
│   └── xgb_btcusdt_h4_features.json      # ✅ Model metadata
│
└── logs/                                   # Created on first run
    ├── trading_bot_20240101.log
    └── trades_20240101.log
```

## 🎯 Next Steps

### For Testing (Recommended First Steps)

1. **Paper Trading Testing**
   ```powershell
   # Run in paper trading mode for 1-2 weeks
   python scripts/run_bot.py
   ```

2. **Review Performance**
   ```powershell
   # Check logs
   Get-Content logs/trades_*.log -Tail 50
   ```

3. **Optimize Parameters**
   - Adjust CONFIDENCE_THRESHOLD
   - Tune STOP_LOSS_PCT and TAKE_PROFIT_PCT
   - Optimize position sizing

### For Production (After Extensive Testing)

1. **Switch to Testnet**
   ```bash
   # In .env
   PAPER_TRADING=false
   USE_TESTNET=true
   MAX_POSITION_SIZE_USD=10.0  # Start small!
   ```

2. **Monitor Closely**
   - Watch for unexpected behavior
   - Verify order execution
   - Check fees and slippage

3. **Move to Mainnet** (Only after testnet success)
   ```bash
   # In .env
   USE_TESTNET=false
   # Get mainnet API keys from https://www.bybit.com/app/user/api-management
   BYBIT_API_KEY=mainnet_key_here
   BYBIT_API_SECRET=mainnet_secret_here
   ```

## 📚 Additional Resources

- **Bybit API Docs**: https://bybit-exchange.github.io/docs/v5/intro
- **Testnet**: https://testnet.bybit.com/
- **API Management**: https://testnet.bybit.com/app/user/api-management

## ⚠️ Disclaimer

**IMPORTANT**: This trading bot is provided for educational purposes. Cryptocurrency trading carries substantial risk. You could lose all your invested capital. Always:

- Start with paper trading
- Test thoroughly on testnet
- Use small amounts initially
- Never invest more than you can afford to lose
- Monitor the bot continuously
- Have a stop-loss strategy
- Understand the risks involved

The developers are not responsible for any financial losses.

## 🙋 Support

Check logs first:
```powershell
# View recent logs
Get-Content logs/trading_bot_*.log -Tail 100

# Search for errors
Get-Content logs/trading_bot_*.log | Select-String "ERROR"
```

Common issues are documented in the Troubleshooting section above.

---

**Good luck and trade safely! 🚀📈**
