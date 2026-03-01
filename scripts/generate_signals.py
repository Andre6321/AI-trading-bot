"""
Live Signal Generator for the 8h Regression Model.

Fetches the latest BTCUSDT 1H data from Bybit, computes all features,
runs the trained XGBoost + LightGBM ensemble, and outputs a clear
BUY / SELL / HOLD signal with confidence and trade parameters.

Usage:
    python scripts/generate_signals.py              # single check
    python scripts/generate_signals.py --loop 60    # check every 60 minutes
    python scripts/generate_signals.py --webhook URL  # POST signals to webhook

Prerequisites:
    python scripts/train_reg8h_final.py   # train the final model first
"""

import sys
import json
import time
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features.alpha_features import build_alpha_features

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"

# ---------------------------------------------------------------------------
# DATA FETCHER (Bybit V5 public API -- no auth needed for klines)
# ---------------------------------------------------------------------------
BYBIT_BASE = "https://api.bybit.com"


def fetch_bybit_klines(symbol="BTCUSDT", interval="60", limit=300):
    """
    Fetch 1H klines from Bybit V5 public API.
    No API key needed for market data.
    Paginates if more than 200 bars needed (API limit per request).
    """
    all_rows = []
    end_time = None  # start from most recent

    while len(all_rows) < limit:
        batch = min(200, limit - len(all_rows))
        url = f"{BYBIT_BASE}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": batch,
        }
        if end_time is not None:
            params["end"] = end_time

        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit API error: {data.get('retMsg')}")

        rows = data["result"]["list"]
        if not rows:
            break

        all_rows.extend(rows)

        # Next batch ends before the oldest timestamp of this batch
        oldest_ts = int(rows[-1][0])
        end_time = oldest_ts - 1

        if len(rows) < batch:
            break  # no more data available

    # Deduplicate by timestamp and reverse to chronological order
    seen = set()
    unique = []
    for r in all_rows:
        ts = r[0]
        if ts not in seen:
            seen.add(ts)
            unique.append(r)
    unique.sort(key=lambda x: int(x[0]))

    df = pd.DataFrame(unique, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)
    df.set_index('timestamp', inplace=True)
    return df


def fetch_bybit_funding_rate(symbol="BTCUSDT"):
    """Fetch current funding rate from Bybit."""
    url = f"{BYBIT_BASE}/v5/market/tickers"
    params = {"category": "linear", "symbol": symbol}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("retCode") == 0:
            ticker = data["result"]["list"][0]
            return float(ticker.get("fundingRate", 0))
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# SIGNAL GENERATOR
# ---------------------------------------------------------------------------

class SignalGenerator:
    """Generates trading signals from the 8h regression model."""

    def __init__(self):
        # Load metadata
        meta_path = MODEL_DIR / "reg_8h_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                "Model not trained yet! Run: python scripts/train_reg8h_final.py"
            )

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.feature_cols = self.meta['feature_cols']
        self.thresholds = self.meta['signal_thresholds']
        self.strategy = self.meta['best_strategy_config']

        # Load models
        lgb_path = self.meta['lgb_model_path']
        xgb_path = self.meta['xgb_model_path']

        self.lgb_model = lgb.Booster(model_file=lgb_path)
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(xgb_path)

        print(f"  Models loaded ({self.meta['n_features']} features)")

    def compute_features(self, df_raw):
        """Build all features from raw OHLCV data.

        Pipeline: raw OHLCV → build_enhanced_features (94 cols) → build_alpha_features (+32 = 126).
        Needs ~750 bars of 1H data for warm-up (MA periods, daily rolling windows).

        Note: For live data, funding_rate is mostly 0 in the historical bars
        (only current value available from API). We fill funding features with 0
        rather than dropping NaN rows, since the trained model can handle it.
        """
        # We need to avoid the internal dropna() in build_features / build_enhanced_features
        # because live data won't have full funding rate history.
        # Instead, we build features manually with forward-fill instead of dropna.

        df = df_raw.copy()

        # --- Stage 1: Base features (from features.py logic) ---
        df['log_ret_1h'] = np.log(df['close'] / df['close'].shift(1))
        df['log_ret_4h'] = np.log(df['close'] / df['close'].shift(4))
        df['log_ret_24h'] = np.log(df['close'] / df['close'].shift(24))

        for w in [10, 20, 50, 100]:
            df[f'ma_{w}'] = df['close'].rolling(w).mean()

        df['rsi_14'] = self._rsi(df['close'], 14)
        df['atr_14'] = self._atr(df['high'], df['low'], df['close'], 14)
        df['vol_24h'] = df['log_ret_1h'].rolling(24).std()

        hour = df.index.hour
        dow = df.index.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        if 'funding_rate' in df.columns:
            df['funding_rate_24h_change'] = df['funding_rate'].diff(24).fillna(0)
            fr_mean = df['funding_rate'].rolling(720, min_periods=1).mean()
            fr_std = df['funding_rate'].rolling(720, min_periods=1).std().replace(0, 1e-10)
            df['funding_rate_zscore_30d'] = (df['funding_rate'] - fr_mean) / fr_std

        for w in [20, 50]:
            df[f'price_vs_ma_{w}'] = df['close'] / df[f'ma_{w}'] - 1
        df['hl_spread_pct'] = (df['high'] - df['low']) / df['close'] * 100
        df['close_position'] = ((df['close'] - df['low']) / (df['high'] - df['low'])).fillna(0.5)
        df['volume_ma_24h'] = df['volume'].rolling(24).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_24h'] + 1e-10)
        df['momentum_3h'] = df['close'] / df['close'].shift(3) - 1
        df['momentum_12h'] = df['close'] / df['close'].shift(12) - 1

        # --- Stage 2: Enhanced features (from features_enhanced.py logic) ---
        import talib

        # Multi-timeframe
        df['close_4h'] = df['close'].rolling(4).mean()
        df['high_4h'] = df['high'].rolling(4).max()
        df['low_4h'] = df['low'].rolling(4).min()
        df['close_1d'] = df['close'].rolling(24).mean()
        df['high_1d'] = df['high'].rolling(24).max()
        df['low_1d'] = df['low'].rolling(24).min()
        df['volume_1d'] = df['volume'].rolling(24).sum()

        close_4h_smooth = df['close'].rolling(4).mean()
        close_1d_smooth = df['close'].rolling(24).mean()
        df['rsi_1h'] = talib.RSI(df['close'].values, timeperiod=14)
        df['rsi_4h'] = talib.RSI(close_4h_smooth.values, timeperiod=14)
        df['rsi_1d'] = talib.RSI(close_1d_smooth.values, timeperiod=14)
        df['ma_20_1h'] = df['close'].rolling(20).mean()
        df['ma_20_4h'] = df['close'].rolling(80).mean()
        df['ma_20_1d'] = df['close'].rolling(480).mean()

        above_1h = (df['close'] > df['ma_20_1h']).astype(int)
        above_4h = (df['close'] > df['ma_20_4h']).astype(int)
        above_1d = (df['close'] > df['ma_20_1d']).astype(int)
        df['trend_alignment'] = (above_1h + above_4h + above_1d) / 3

        # Volatility
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        atr_pct = df['atr'] / df['close'] * 100
        df['atr_percentile'] = atr_pct.rolling(100, min_periods=1).rank(pct=True)
        df['hist_volatility'] = df['close'].pct_change().rolling(24).std() * np.sqrt(24)
        df['volatility_regime'] = (df['atr_percentile'] > 0.7).astype(int)

        # Volume
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ma_50'] = df['volume'].rolling(50).mean()
        df['volume_spike'] = (df['volume'] > 2 * df['volume_ma_20']).astype(int)
        typical = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical * df['volume']).rolling(24).sum() / (df['volume'].rolling(24).sum() + 1e-10)
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)
        direction = np.sign(df['close'].diff())
        df['obv'] = (direction * df['volume']).cumsum()
        df['obv_ma'] = df['obv'].rolling(20).mean()
        df['obv_trend'] = (df['obv'] > df['obv_ma']).astype(int)
        df['vpt'] = (df['close'].pct_change() * df['volume']).cumsum()

        # Trend strength
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        macd, macd_sig, macd_hist = talib.MACD(df['close'].values, 12, 26, 9)
        df['macd'] = macd
        df['macd_signal'] = macd_sig
        df['macd_hist'] = macd_hist
        df['macd_cross'] = np.sign(df['macd'] - df['macd_signal']).diff().fillna(0)
        df['sar'] = talib.SAR(df['high'].values, df['low'].values)
        df['sar_trend'] = (df['close'] > df['sar']).astype(int)
        df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=20)
        df['williams_r'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        df['momentum_10'] = talib.MOM(df['close'].values, timeperiod=10)
        df['momentum_20'] = talib.MOM(df['close'].values, timeperiod=20)
        df['roc_10'] = talib.ROC(df['close'].values, timeperiod=10)
        df['roc_20'] = talib.ROC(df['close'].values, timeperiod=20)

        # Market structure
        df['distance_from_high'] = (df['close'] - df['high'].rolling(50).max()) / df['close']
        df['distance_from_low'] = (df['close'] - df['low'].rolling(50).min()) / df['close']
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['is_doji'] = (df['body_size'] < 0.1).astype(int)

        # Support/resistance
        price_q = pd.cut(df['close'], bins=20, labels=False)
        df['price_level'] = price_q
        df['level_touches'] = price_q.rolling(200, min_periods=1).apply(
            lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 0, raw=False)

        # Funding rate enhanced
        if 'funding_rate' in df.columns:
            fr = df['funding_rate']
            df['funding_cum_8h'] = fr.rolling(8, min_periods=1).sum()
            df['funding_cum_24h'] = fr.rolling(24, min_periods=1).sum()
            df['funding_cum_72h'] = fr.rolling(72, min_periods=1).sum()
            fr_ma_168 = fr.rolling(168, min_periods=1).mean()
            fr_std_168 = fr.rolling(168, min_periods=1).std().replace(0, 1e-10)
            df['funding_zscore'] = (fr - fr_ma_168) / fr_std_168
            df['funding_mr_signal'] = -df['funding_zscore']
            df['funding_accel'] = fr.diff().rolling(8, min_periods=1).mean()
            df['funding_extreme_long'] = (df['funding_zscore'] > 2).astype(int)
            df['funding_extreme_short'] = (df['funding_zscore'] < -2).astype(int)

        # --- Stage 3: Alpha features ---
        df = build_alpha_features(df)

        # Fill NaN (from warm-up) instead of dropping
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.bfill().fillna(0)

        return df

    @staticmethod
    def _rsi(close, window=14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-10)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _atr(high, low, close, window=14):
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def predict(self, df):
        """
        Run the ensemble prediction on the latest bar.
        Returns (predicted_8h_return, long_signal_strength, short_signal_strength).
        """
        # Get feature values for the last row
        latest = df.iloc[-1]
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            print(f"  WARNING: Missing features: {missing[:5]}...")

        available = [c for c in self.feature_cols if c in df.columns]
        X = df[available].iloc[[-1]]  # last row as DataFrame

        # Fill any remaining NaN with 0 (shouldn't happen with enough history)
        X = X.fillna(0)

        # Predict with both models
        lgb_pred = self.lgb_model.predict(X)[0]
        xgb_pred = self.xgb_model.predict(X)[0]
        pred_return = (lgb_pred + xgb_pred) / 2

        # Convert to signal strengths (sigmoid scaling, same as V6)
        long_strength = 1.0 / (1.0 + np.exp(-pred_return * 200))
        short_strength = 1.0 / (1.0 + np.exp(pred_return * 200))

        return pred_return, long_strength, short_strength

    def generate_signal(self, df, current_price=None, current_atr=None):
        """
        Generate a trading signal from the latest data.

        Returns a dict with:
          - action: 'BUY', 'SELL', or 'HOLD'
          - confidence: 'high', 'medium', 'low'
          - predicted_return_8h: expected % return over next 8 hours
          - entry_price, stop_loss, take_profit, trail_stop
          - reasoning: human-readable explanation
        """
        pred_ret, long_str, short_str = self.predict(df)

        # Determine thresholds
        thr = self.thresholds['top_5pct']   # best config uses top 5%
        thr_10 = self.thresholds['top_10pct']
        thr_20 = self.thresholds['top_20pct']

        if current_price is None:
            current_price = float(df['close'].iloc[-1])
        if current_atr is None:
            if 'atr_14' in df.columns:
                current_atr = float(df['atr_14'].iloc[-1])
            else:
                # Compute ATR manually
                h = df['high'].values[-14:]
                l = df['low'].values[-14:]
                c = df['close'].values[-14:]
                tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)),
                                                   abs(l - np.roll(c, 1))))[1:]
                current_atr = float(np.mean(tr))

        # Check ATR regime filter (skip ultra-low vol)
        atr_pct = current_atr / current_price * 100
        regime_ok = atr_pct >= self.meta.get('atr_pct_25_threshold', 0.3)

        # Signal logic
        action = 'HOLD'
        confidence = 'none'
        reasoning = []

        if not regime_ok:
            reasoning.append(f"Low volatility regime (ATR%={atr_pct:.2f}%), skipping")
        elif long_str >= thr['long']:
            action = 'BUY'
            if long_str >= thr['long']:
                confidence = 'high'
                reasoning.append(f"LONG signal in top 5% (strength={long_str:.4f} >= {thr['long']:.4f})")
            elif long_str >= thr_10['long']:
                confidence = 'medium'
                reasoning.append(f"LONG signal in top 10%")
        elif short_str >= thr['short']:
            action = 'SELL'
            if short_str >= thr['short']:
                confidence = 'high'
                reasoning.append(f"SHORT signal in top 5% (strength={short_str:.4f} >= {thr['short']:.4f})")
            elif short_str >= thr_10['short']:
                confidence = 'medium'
                reasoning.append(f"SHORT signal in top 10%")
        else:
            reasoning.append(f"No signal (long={long_str:.4f}, short={short_str:.4f})")
            reasoning.append(f"Need long >= {thr['long']:.4f} or short >= {thr['short']:.4f}")

        # Calculate trade parameters
        sl_mult = self.strategy['sl_mult_atr']  # 1.5
        tp_mult = self.strategy['tp_mult_atr']  # 3.0
        trail_mult = self.strategy['trail_mult_atr']  # 0.75

        if action == 'BUY':
            entry = current_price
            sl = entry - sl_mult * current_atr
            tp = entry + tp_mult * current_atr
            trail = entry - trail_mult * current_atr
            reasoning.append(f"Predicted +{pred_ret*100:.3f}% over 8h")
        elif action == 'SELL':
            entry = current_price
            sl = entry + sl_mult * current_atr
            tp = entry - tp_mult * current_atr
            trail = entry + trail_mult * current_atr
            reasoning.append(f"Predicted {pred_ret*100:.3f}% over 8h")
        else:
            entry = sl = tp = trail = None

        # Risk per trade ($25K account)
        risk_usd = None
        position_size_btc = None
        if entry and sl:
            account_size = 25000
            risk_pct = 0.01  # 1%
            sl_dist = abs(entry - sl)
            risk_usd = account_size * risk_pct  # $250
            position_size_btc = risk_usd / sl_dist if sl_dist > 0 else 0

        return {
            'timestamp': str(datetime.now(timezone.utc)),
            'action': action,
            'confidence': confidence,
            'predicted_return_8h': round(pred_ret * 100, 4),
            'long_signal': round(long_str, 4),
            'short_signal': round(short_str, 4),
            'current_price': round(current_price, 2),
            'entry_price': round(entry, 2) if entry else None,
            'stop_loss': round(sl, 2) if sl else None,
            'take_profit': round(tp, 2) if tp else None,
            'trailing_stop_initial': round(trail, 2) if trail else None,
            'atr_14': round(current_atr, 2),
            'atr_pct': round(atr_pct, 3),
            'regime_ok': regime_ok,
            'risk_usd': round(risk_usd, 2) if risk_usd else None,
            'position_size_btc': round(position_size_btc, 6) if position_size_btc else None,
            'max_hold_hours': self.strategy['max_hold_bars'],  # 12 hours
            'reasoning': ' | '.join(reasoning),
        }


# ---------------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------------

def print_signal(sig):
    """Pretty-print a trading signal."""
    action_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪'}
    conf_emoji = {'high': '🔥', 'medium': '⚡', 'low': '💤', 'none': ''}

    print(f"\n{'='*65}")
    print(f"  {action_emoji.get(sig['action'], '')} SIGNAL: {sig['action']}  "
          f"{conf_emoji.get(sig['confidence'], '')} [{sig['confidence']}]")
    print(f"{'='*65}")
    print(f"  Time:          {sig['timestamp']}")
    print(f"  BTC Price:     ${sig['current_price']:,.2f}")
    print(f"  Predicted 8h:  {sig['predicted_return_8h']:+.4f}%")
    print(f"  Long Signal:   {sig['long_signal']:.4f}")
    print(f"  Short Signal:  {sig['short_signal']:.4f}")
    print(f"  ATR(14):       ${sig['atr_14']:,.2f} ({sig['atr_pct']:.3f}%)")
    print(f"  Regime OK:     {'Yes' if sig['regime_ok'] else 'NO (low vol)'}")

    if sig['action'] != 'HOLD':
        print(f"  {'─'*50}")
        print(f"  Entry:         ${sig['entry_price']:,.2f}")
        print(f"  Stop Loss:     ${sig['stop_loss']:,.2f}")
        print(f"  Take Profit:   ${sig['take_profit']:,.2f}")
        print(f"  Trail Start:   ${sig['trailing_stop_initial']:,.2f}")
        print(f"  Max Hold:      {sig['max_hold_hours']} hours")
        print(f"  Risk ($25K):   ${sig['risk_usd']:,.2f}")
        print(f"  Position:      {sig['position_size_btc']:.6f} BTC")

    print(f"  {'─'*50}")
    print(f"  Reasoning:     {sig['reasoning']}")
    print(f"{'='*65}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BTC 8h Regression Signal Generator")
    parser.add_argument('--loop', type=int, default=0,
                        help='Re-check every N minutes (0 = run once)')
    parser.add_argument('--webhook', type=str, default=None,
                        help='POST signals to this URL (e.g. Discord webhook)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    args = parser.parse_args()

    print("=" * 65)
    print("  BTC 8H REGRESSION SIGNAL GENERATOR")
    print("=" * 65)

    gen = SignalGenerator()

    while True:
        try:
            print(f"\n  Fetching latest {args.symbol} 1H candles...")
            df_raw = fetch_bybit_klines(symbol=args.symbol, interval="60", limit=750)
            print(f"  Got {len(df_raw)} candles, latest: {df_raw.index[-1]}")

            # Fetch funding rate and add to dataframe
            funding = fetch_bybit_funding_rate(args.symbol)
            df_raw['funding_rate'] = 0.0
            # Bybit funding is every 8h, we just use current as approximation
            df_raw.loc[df_raw.index[-1], 'funding_rate'] = funding

            print(f"  Building features...")
            df_feat = gen.compute_features(df_raw)

            signal = gen.generate_signal(df_feat)
            print_signal(signal)

            # Post to webhook if configured
            if args.webhook and signal['action'] != 'HOLD':
                try:
                    payload = {
                        "content": (
                            f"**{signal['action']}** BTC @ ${signal['current_price']:,.2f}\n"
                            f"Predicted 8h: {signal['predicted_return_8h']:+.4f}%\n"
                            f"SL: ${signal['stop_loss']:,.2f} | TP: ${signal['take_profit']:,.2f}\n"
                            f"Risk: ${signal['risk_usd']:,.2f} | Pos: {signal['position_size_btc']:.6f} BTC"
                        )
                    }
                    requests.post(args.webhook, json=payload, timeout=10)
                    print(f"  Webhook sent!")
                except Exception as e:
                    print(f"  Webhook failed: {e}")

            # Save latest signal to file
            sig_path = ROOT / "outputs" / "latest_signal.json"
            with open(sig_path, 'w') as f:
                json.dump(signal, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        if args.loop <= 0:
            break
        print(f"\n  Sleeping {args.loop} minutes...")
        time.sleep(args.loop * 60)


if __name__ == "__main__":
    main()
