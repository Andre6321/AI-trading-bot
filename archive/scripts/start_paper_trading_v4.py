"""
Paper Trading Bot V4 -- Bidirectional (LONG + SHORT)

Uses the V4 bidirectional ensemble model with:
  - LONG model: P(long TP hit) based on TP=2.5x ATR, SL=1.5x ATR
  - SHORT model: P(short TP hit) based on TP=2.5x ATR, SL=1.5x ATR
  - Regime filter: skip low-volatility (bottom 25% ATR)
  - Funding rate features (if available)
  - Isotonic calibration for reliable probabilities
  - ATR-based dynamic SL/TP with timeout at 12 bars

Uses real-time Bybit market data + simulated order execution.

Usage:
    python scripts/start_paper_trading_v4.py
"""

import sys
import signal
import time
import pickle
import logging
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bybit_ai_trader.client import BybitClient
from bybit_ai_trader.paper_trader import PaperTrader
from features.alpha_features import build_alpha_features

# Graceful shutdown
running = True
def _sig_handler(sig, frame):
    global running
    print("\n\n⚠️  Shutdown signal received. Finishing current iteration...")
    running = False
signal.signal(signal.SIGINT, _sig_handler)


# ======================================================================
# CONFIG
# ======================================================================
CFG = {
    "symbol": "BTCUSDT",
    "interval": "60",          # 1h candles
    "buffer_size": 300,        # candles to keep for feature calc
    "check_interval_sec": 300, # check every 5 min (trade on new candle only)

    # Model
    "model_path": "models/ensemble_v4_bidirectional.pkl",
    "fallback_model_path": "models/ensemble_v3_optimized.pkl",

    # Thresholds
    "long_threshold": 0.20,    # P(long TP) > this → go long
    "short_threshold": 0.20,   # P(short TP) > this → go short

    # Risk
    "initial_balance": 10000.0,
    "risk_per_trade_pct": 0.02,
    "max_position_usd": 1000.0,
    "fee_rate": 0.00075,
    "slippage_pct": 0.0002,
    "max_hold_bars": 12,

    # Barrier multipliers (must match training)
    "long_tp_mult": 2.5,
    "long_sl_mult": 1.5,
    "short_tp_mult": 2.5,
    "short_sl_mult": 1.5,
}


# ======================================================================
# Model loader
# ======================================================================

def load_model(model_path: str):
    """Load V4 (bidirectional) or V3 (long-only) model."""
    path = Path(model_path)
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    version = "v4" if "long_xgb" in artifact else "v3"
    return artifact, version


def predict(artifact, version, features_row: pd.DataFrame):
    """
    Predict P(long TP) and P(short TP).
    Returns (long_prob, short_prob).
    """
    feat_cols = artifact.get("feature_cols", [])
    # Ensure all required features are present
    missing = [c for c in feat_cols if c not in features_row.columns]
    if missing:
        logging.warning(f"Missing {len(missing)} features: {missing[:5]}...")
        for c in missing:
            features_row[c] = 0.0

    X = features_row[feat_cols]

    if version == "v4":
        p_long_xgb = artifact["long_xgb"].predict_proba(X)[:, 1]
        p_long_lgb = artifact["long_lgb"].predict_proba(X)[:, 1]
        p_long = (p_long_xgb + p_long_lgb) / 2.0
        if "long_calibrator" in artifact:
            p_long = artifact["long_calibrator"].predict(p_long)

        p_short_xgb = artifact["short_xgb"].predict_proba(X)[:, 1]
        p_short_lgb = artifact["short_lgb"].predict_proba(X)[:, 1]
        p_short = (p_short_xgb + p_short_lgb) / 2.0
        if "short_calibrator" in artifact:
            p_short = artifact["short_calibrator"].predict(p_short)

        return float(p_long[0]), float(p_short[0])

    else:  # v3 (long-only)
        p_xgb = artifact["xgb_model"].predict_proba(X)[:, 1]
        p_lgb = artifact["lgb_model"].predict_proba(X)[:, 1]
        p_cat = artifact["cat_model"].predict_proba(X)[:, 1]
        p = (p_xgb + p_lgb + p_cat) / 3.0
        if "calibrator" in artifact:
            p = artifact["calibrator"].predict(p)
        return float(p[0]), 0.0  # no short signal in V3


# ======================================================================
# Data fetcher (simple, using Bybit public API)
# ======================================================================

def fetch_candles(client: BybitClient, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetch latest candles and return as DataFrame."""
    candles = client.get_latest_candles(symbol, interval, limit)
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all features from raw OHLCV data."""
    # Base features
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Log returns
    df["log_ret_1h"] = np.log(close / close.shift(1))
    df["log_ret_4h"] = np.log(close / close.shift(4))
    df["log_ret_24h"] = np.log(close / close.shift(24))

    # Moving averages
    for w in [10, 20, 50, 100]:
        df[f"ma_{w}"] = close.rolling(w).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # ATR
    hl = high - low
    hc = np.abs(high - close.shift(1))
    lc = np.abs(low - close.shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / close * 100

    # Volume
    df["vol_24h"] = df["volume"].rolling(24).mean()

    # Time features
    if hasattr(df.index, "hour"):
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Price vs MA
    df["price_vs_ma_20"] = (close - df["ma_20"]) / (df["ma_20"] + 1e-10) * 100

    # Bollinger
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-10) * 100
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # CCI
    tp = (high + low + close) / 3
    tp_ma = tp.rolling(20).mean()
    tp_dev = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df["cci"] = (tp - tp_ma) / (0.015 * tp_dev + 1e-10)

    # Williams %R
    hh = high.rolling(14).max()
    ll = low.rolling(14).min()
    df["williams_r"] = -100 * (hh - close) / (hh - ll + 1e-10)

    # Volume ratio
    df["volume_ratio"] = df["volume"] / (df["vol_24h"] + 1e-10)

    # Alpha features
    df = build_alpha_features(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ======================================================================
# MAIN TRADING LOOP
# ======================================================================

def main():
    print("=" * 80)
    print(" " * 15 + "BYBIT AI TRADING BOT V4 - PAPER TRADING (LONG+SHORT)")
    print("=" * 80)

    ROOT = Path(__file__).parent.parent

    # Load model
    model_path = ROOT / CFG["model_path"]
    fallback_path = ROOT / CFG["fallback_model_path"]

    artifact, version = load_model(str(model_path))
    if artifact is None:
        print(f"  V4 model not found at {model_path}")
        artifact, version = load_model(str(fallback_path))
        if artifact is None:
            print(f"  V3 model also not found at {fallback_path}")
            print("  Run train_v4_bidirectional.py first!")
            return

    # Update config from model
    model_cfg = artifact.get("config", {})
    if model_cfg:
        CFG["long_tp_mult"] = model_cfg.get("long_tp_mult", CFG["long_tp_mult"])
        CFG["long_sl_mult"] = model_cfg.get("long_sl_mult", CFG["long_sl_mult"])
        CFG["short_tp_mult"] = model_cfg.get("short_tp_mult", CFG["short_tp_mult"])
        CFG["short_sl_mult"] = model_cfg.get("short_sl_mult", CFG["short_sl_mult"])
        CFG["max_hold_bars"] = model_cfg.get("max_holding", CFG["max_hold_bars"])

    feat_cols = artifact.get("feature_cols", [])
    regime_thr = artifact.get("regime_threshold_atr_pct", None)

    print(f"\n  📊 Model: {version.upper()}")
    print(f"  📈 Features: {len(feat_cols)}")
    print(f"  🎯 Long threshold: {CFG['long_threshold']}")
    print(f"  🎯 Short threshold: {CFG['short_threshold']}")
    print(f"  🛡️  Barriers: L-TP={CFG['long_tp_mult']}x L-SL={CFG['long_sl_mult']}x "
          f"S-TP={CFG['short_tp_mult']}x S-SL={CFG['short_sl_mult']}x")
    print(f"  ⏱️  Max hold: {CFG['max_hold_bars']} bars")
    if regime_thr:
        print(f"  🌡️  Regime filter: ATR% >= {regime_thr:.4f}%")
    print()

    # Init components
    client = BybitClient(api_key="", api_secret="", testnet=False)
    paper = PaperTrader(
        initial_balance=CFG["initial_balance"],
        fee_rate=CFG["fee_rate"],
        slippage_pct=CFG["slippage_pct"],
    )

    print(f"  💰 Starting balance: ${CFG['initial_balance']:,.2f}")
    print(f"  🚀 Bot started! Press Ctrl+C to stop.\n")
    print("=" * 80)

    last_candle_ts = None
    entry_bar_count = 0
    iteration = 0

    while running:
        iteration += 1
        now = datetime.now()

        try:
            # Fetch latest candles
            df = fetch_candles(client, CFG["symbol"], CFG["interval"], CFG["buffer_size"])
            if df.empty or len(df) < 100:
                print(f"  [{now:%H:%M}] Insufficient data ({len(df)} candles), waiting...")
                time.sleep(CFG["check_interval_sec"])
                continue

            current_candle_ts = df.index[-2]  # last COMPLETE candle
            current_price = float(df["close"].iloc[-1])

            # Only trade on new candle close
            if last_candle_ts is not None and current_candle_ts <= last_candle_ts:
                # Still check SL/TP on current price
                pos = paper.get_position(CFG["symbol"])
                if pos:
                    result = paper.update_positions(CFG["symbol"], current_price)
                    if result:
                        entry_bar_count = 0
                        print(f"  [{now:%H:%M}] 🔔 Position closed by {result.get('reason', 'SL/TP')}: "
                              f"PnL ${result.get('pnl', 0):+.2f}")

                    # Manual timeout check
                    entry_bar_count += 0  # don't increment between candles

                time.sleep(CFG["check_interval_sec"])
                continue

            last_candle_ts = current_candle_ts

            # Build features on complete candles (exclude current incomplete bar)
            df_complete = df.iloc[:-1].copy()
            df_feat = build_features(df_complete)
            df_feat = df_feat.dropna()

            if len(df_feat) < 10:
                print(f"  [{now:%H:%M}] Features insufficient ({len(df_feat)} rows)")
                time.sleep(CFG["check_interval_sec"])
                continue

            # Get latest row
            latest = df_feat.iloc[[-1]]
            atr_pct_val = float(latest["atr_pct"].iloc[0]) if "atr_pct" in latest.columns else 0
            atr_val = float(latest["atr_14"].iloc[0]) if "atr_14" in latest.columns else 0
            rsi_val = float(latest["rsi_14"].iloc[0]) if "rsi_14" in latest.columns else 50

            # Regime check
            regime_ok = True
            if regime_thr and atr_pct_val < regime_thr:
                regime_ok = False

            # Predict
            long_prob, short_prob = predict(artifact, version, latest)

            # Position management
            pos = paper.get_position(CFG["symbol"])

            # Check timeout for existing position
            if pos:
                entry_bar_count += 1
                result = paper.update_positions(CFG["symbol"], current_price)
                if result:
                    entry_bar_count = 0
                    print(f"  [{now:%H:%M}] 🔔 Position closed: PnL ${result.get('pnl', 0):+.2f}")
                    pos = None

                if pos and entry_bar_count >= CFG["max_hold_bars"]:
                    fees = paper._calculate_fees(pos["size"] * current_price)
                    result = paper._close_position(CFG["symbol"], current_price, fees, "timeout")
                    entry_bar_count = 0
                    print(f"  [{now:%H:%M}] ⏰ TIMEOUT close: PnL ${result.get('pnl', 0):+.2f}")
                    pos = None

            # Generate signal (only if no position)
            action = "HOLD"
            if pos is None and regime_ok:
                if long_prob > CFG["long_threshold"] and long_prob >= short_prob:
                    action = "LONG"
                elif short_prob > CFG["short_threshold"] and short_prob > long_prob:
                    action = "SHORT"

            # Execute
            if action in ("LONG", "SHORT"):
                side = "Buy" if action == "LONG" else "Sell"
                tp_mult = CFG["long_tp_mult"] if action == "LONG" else CFG["short_tp_mult"]
                sl_mult = CFG["long_sl_mult"] if action == "LONG" else CFG["short_sl_mult"]

                if action == "LONG":
                    sl_price = current_price - sl_mult * atr_val
                    tp_price = current_price + tp_mult * atr_val
                else:
                    sl_price = current_price + sl_mult * atr_val
                    tp_price = current_price - tp_mult * atr_val

                sl_dist = abs(current_price - sl_price) / current_price
                pos_size_usd = min(
                    paper.balance * CFG["risk_per_trade_pct"] / max(sl_dist, 0.001),
                    CFG["max_position_usd"],
                    paper.balance * 0.30,
                )
                pos_size = pos_size_usd / current_price

                if pos_size_usd >= 20:
                    result = paper.place_market_order(
                        symbol=CFG["symbol"],
                        side=side,
                        quantity=pos_size,
                        current_price=current_price,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                    )
                    if result["success"]:
                        entry_bar_count = 0
                        emoji = "🟢" if action == "LONG" else "🔴"
                        print(f"  [{now:%H:%M}] {emoji} {action} @ ${current_price:,.2f}  "
                              f"P(L)={long_prob:.3f} P(S)={short_prob:.3f}  "
                              f"SL=${sl_price:,.0f} TP=${tp_price:,.0f}  "
                              f"Size=${pos_size_usd:,.0f}")

            # Status line
            bal = paper.get_balance()
            regime_str = "✅" if regime_ok else "🔴LOW-VOL"
            pos_str = "FLAT"
            if pos:
                pos_str = f"{pos['side']} ${pos.get('unrealized_pnl', 0):+.0f}"

            if iteration % 12 == 1 or action != "HOLD":
                print(f"  [{now:%H:%M}] BTC=${current_price:,.0f} | "
                      f"P(L)={long_prob:.3f} P(S)={short_prob:.3f} | "
                      f"RSI={rsi_val:.0f} ATR%={atr_pct_val:.3f} {regime_str} | "
                      f"{pos_str} | Eq=${bal['equity']:,.0f} "
                      f"({(bal['total_pnl']/CFG['initial_balance']*100):+.1f}%)")

        except Exception as e:
            print(f"  [{now:%H:%M}] ❌ Error: {e}")
            traceback.print_exc()

        time.sleep(CFG["check_interval_sec"])

    # Shutdown
    print("\n" + "=" * 80)
    print("  FINAL RESULTS")
    print("=" * 80)

    bal = paper.get_balance()
    print(f"  Initial: ${CFG['initial_balance']:,.2f}")
    print(f"  Final:   ${bal['equity']:,.2f}")
    print(f"  PnL:     ${bal['total_pnl']:+,.2f} "
          f"({bal['total_pnl']/CFG['initial_balance']*100:+.1f}%)")

    if paper.trade_history:
        stats = paper.get_performance_stats()
        print(f"\n  Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Avg Win: ${stats['avg_win']:+.2f}")
        print(f"  Avg Loss: ${stats['avg_loss']:+.2f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
