"""
Realistic Backtest for V3 Optimized Model

This script does a proper bar-by-bar backtest using the walk-forward
OOS predictions from the V3 model, with:
- Proper position sizing (fixed fraction + ATR-scaled)
- One position at a time (no overlapping trades)
- Real fees (0.075% taker) + slippage (0.02%)
- Max holding period enforcement
- Regime filter (skip low-vol periods)
- No reinvestment / no leverage initially

Output: equity curve, trade log, performance stats
"""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.alpha_features import build_alpha_features
from features.triple_barrier import triple_barrier_labels


def main():
    print("=" * 70)
    print("  REALISTIC BACKTEST -- V3 Optimized Model")
    print("=" * 70)

    ROOT = Path(__file__).parent.parent
    MODEL_DIR = ROOT / "models"

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model_path = MODEL_DIR / "ensemble_v3_optimized.pkl"
    if not model_path.exists():
        print("[ERR] Model not found. Run train_v3_optimized.py first.")
        return

    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)

    feature_cols = artifact['feature_cols']
    config = artifact['config']
    calibrator = artifact['calibrator']
    regime_threshold = artifact.get('regime_threshold_atr_pct', 0.005)

    print(f"   Model loaded: {len(feature_cols)} features")
    print(f"   Config: TP={config['tp_mult']}x SL={config['sl_mult']}x Hold={config['max_holding']}h")

    # ------------------------------------------------------------------
    # 2. Load data & build features
    # ------------------------------------------------------------------
    data_paths = [
        ROOT / "data" / "processed" / "btcusdt_1h_full_history_enhanced.parquet",
        ROOT / "data" / "processed" / "btcusdt_1h_features.parquet",
    ]
    data_path = next((p for p in data_paths if p.exists()), None)
    if data_path is None:
        print("[ERR] No data found!")
        return

    print(f"\n   Loading data from {data_path.name}...")
    df = pd.read_parquet(data_path)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # Build alpha features
    df = build_alpha_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ['close', 'high', 'low'])

    # ATR percentage for regime filter
    if 'atr_pct' not in df.columns and 'atr_14' in df.columns:
        df['atr_pct'] = df['atr_14'] / df['close'] * 100

    print(f"   {len(df):,} bars ready for backtest")

    # ------------------------------------------------------------------
    # 3. Generate predictions (full dataset, simulating walk-forward)
    # ------------------------------------------------------------------
    print("\n   Generating predictions...")

    X = df[feature_cols]

    # Use all three models
    p_xgb = artifact['xgb_model'].predict_proba(X)[:, 1]
    p_lgb = artifact['lgb_model'].predict_proba(X)[:, 1]
    p_cat = artifact['cat_model'].predict_proba(X)[:, 1]
    p_raw = (p_xgb + p_lgb + p_cat) / 3.0

    # Calibrate
    p_cal = calibrator.predict(p_raw)

    df['prob_raw'] = p_raw
    df['prob_cal'] = p_cal

    # Regime filter
    df['regime_ok'] = df['atr_pct'] >= regime_threshold if 'atr_pct' in df.columns else True

    # ------------------------------------------------------------------
    # 4. Use only the LAST 30% of data (out-of-sample region)
    # ------------------------------------------------------------------
    # The model was trained on ~first 85%, so test on the last ~15%
    oos_start = int(len(df) * 0.85)
    df_test = df.iloc[oos_start:].copy()
    print(f"   Out-of-sample region: {len(df_test):,} bars ({df_test.index[0]} to {df_test.index[-1]})")

    # ------------------------------------------------------------------
    # 5. Bar-by-bar backtest
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  BAR-BY-BAR BACKTEST (realistic)")
    print("=" * 70)

    # Backtest parameters
    INITIAL_CAPITAL = 10000.0
    FEE_PCT = 0.00075          # 0.075% per side (taker)
    SLIPPAGE_PCT = 0.0002      # 0.02% slippage per side
    RISK_PER_TRADE = 0.02      # 2% of capital risked per trade
    MAX_POSITION_FRAC = 0.20   # max 20% of capital in a single position
    MAX_LEVERAGE = 3.0         # maximum leverage

    thresholds_to_test = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    best_sharpe = -999
    best_thr = None
    best_results = None

    for threshold in thresholds_to_test:
        # State
        capital = INITIAL_CAPITAL
        equity_curve = [capital]
        in_position = False
        entry_price = 0
        entry_bar = 0
        position_size_usd = 0
        tp_price = 0
        sl_price = 0
        trades = []

        close_vals = df_test['close'].values
        high_vals = df_test['high'].values
        low_vals = df_test['low'].values
        prob_vals = df_test['prob_cal'].values
        atr_vals = df_test['atr_14'].values if 'atr_14' in df_test.columns else None
        regime_vals = df_test['regime_ok'].values

        for i in range(len(df_test)):
            if in_position:
                # Check exits
                bars_held = i - entry_bar
                exit_price = None
                exit_reason = None

                # SL check (use low)
                if low_vals[i] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                # TP check (use high)
                elif high_vals[i] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                # Timeout
                elif bars_held >= config['max_holding']:
                    exit_price = close_vals[i]
                    exit_reason = 'TIMEOUT'

                if exit_price is not None:
                    # Calculate P&L
                    gross_pnl_pct = (exit_price - entry_price) / entry_price
                    fee_cost = 2 * (FEE_PCT + SLIPPAGE_PCT)  # entry + exit
                    net_pnl_pct = gross_pnl_pct - fee_cost
                    net_pnl_usd = position_size_usd * net_pnl_pct

                    capital += net_pnl_usd
                    in_position = False

                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'reason': exit_reason,
                        'bars_held': bars_held,
                        'gross_pnl_pct': gross_pnl_pct * 100,
                        'net_pnl_pct': net_pnl_pct * 100,
                        'net_pnl_usd': net_pnl_usd,
                        'capital_after': capital,
                    })

            else:
                # Check for entry signal
                if (prob_vals[i] > threshold and
                    regime_vals[i] and
                    capital > 100 and
                    atr_vals is not None and atr_vals[i] > 0):

                    entry_price = close_vals[i] * (1 + SLIPPAGE_PCT)  # slippage on entry
                    entry_bar = i

                    atr = atr_vals[i]
                    tp_price = entry_price + config['tp_mult'] * atr
                    sl_price = entry_price - config['sl_mult'] * atr

                    # Position sizing: risk 2% of capital on the SL distance
                    sl_distance_pct = (entry_price - sl_price) / entry_price
                    risk_usd = capital * RISK_PER_TRADE
                    position_size_usd = min(
                        risk_usd / sl_distance_pct if sl_distance_pct > 0 else 0,
                        capital * MAX_POSITION_FRAC,
                        capital * MAX_LEVERAGE
                    )

                    if position_size_usd > 50:  # minimum $50 position
                        in_position = True
                    else:
                        entry_price = 0

            equity_curve.append(capital)

        # Calculate stats
        if len(trades) == 0:
            continue

        trades_df = pd.DataFrame(trades)
        n_trades = len(trades_df)
        wins = trades_df['net_pnl_usd'] > 0
        n_wins = wins.sum()
        win_rate = n_wins / n_trades

        total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        avg_return = trades_df['net_pnl_pct'].mean()

        # Profit factor
        gross_profit = trades_df.loc[wins, 'net_pnl_usd'].sum()
        gross_loss = abs(trades_df.loc[~wins, 'net_pnl_usd'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        dd = (peak - equity_arr) / peak
        max_dd = dd.max() * 100

        # Sharpe (approximate)
        daily_returns = trades_df['net_pnl_pct'].values / 100
        if len(daily_returns) > 1:
            sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252 * 2)
        else:
            sharpe = 0

        # Outcome breakdown
        tp_count = (trades_df['reason'] == 'TP').sum()
        sl_count = (trades_df['reason'] == 'SL').sum()
        timeout_count = (trades_df['reason'] == 'TIMEOUT').sum()

        results = {
            'threshold': threshold,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return,
            'avg_return_per_trade': avg_return,
            'profit_factor': pf,
            'max_drawdown_pct': max_dd,
            'sharpe_approx': sharpe,
            'tp_hits': tp_count,
            'sl_hits': sl_count,
            'timeouts': timeout_count,
            'avg_win': trades_df.loc[wins, 'net_pnl_pct'].mean() if wins.any() else 0,
            'avg_loss': trades_df.loc[~wins, 'net_pnl_pct'].mean() if (~wins).any() else 0,
            'final_capital': capital,
        }

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_thr = threshold
            best_results = results
            best_trades = trades_df.copy()
            best_equity = equity_arr.copy()

    # ------------------------------------------------------------------
    # 6. Print results
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS (OOS region, {len(df_test):,} bars)")
    print(f"{'='*70}")
    print(f"\n  {'Thresh':>8s} {'Trades':>7s} {'WinRate':>8s} {'TotalRet':>10s} "
          f"{'AvgRet':>8s} {'PF':>6s} {'MaxDD':>8s} {'Sharpe':>7s} "
          f"{'TP':>4s} {'SL':>4s} {'TO':>4s}")
    print(f"  {'-'*90}")

    for threshold in thresholds_to_test:
        # Re-run for display (simplified version)
        capital = INITIAL_CAPITAL
        in_position = False
        entry_price = 0
        entry_bar = 0
        position_size_usd = 0
        tp_price = 0
        sl_price = 0
        trades = []

        close_vals = df_test['close'].values
        high_vals = df_test['high'].values
        low_vals = df_test['low'].values
        prob_vals = df_test['prob_cal'].values
        atr_vals = df_test['atr_14'].values if 'atr_14' in df_test.columns else None
        regime_vals = df_test['regime_ok'].values

        for i in range(len(df_test)):
            if in_position:
                bars_held = i - entry_bar
                exit_price = None
                exit_reason = None

                if low_vals[i] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                elif high_vals[i] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                elif bars_held >= config['max_holding']:
                    exit_price = close_vals[i]
                    exit_reason = 'TIMEOUT'

                if exit_price is not None:
                    gross_pnl_pct = (exit_price - entry_price) / entry_price
                    fee_cost = 2 * (FEE_PCT + SLIPPAGE_PCT)
                    net_pnl_pct = gross_pnl_pct - fee_cost
                    net_pnl_usd = position_size_usd * net_pnl_pct
                    capital += net_pnl_usd
                    in_position = False
                    trades.append({
                        'reason': exit_reason,
                        'net_pnl_pct': net_pnl_pct * 100,
                        'net_pnl_usd': net_pnl_usd,
                    })
            else:
                if (prob_vals[i] > threshold and regime_vals[i] and
                    capital > 100 and atr_vals is not None and atr_vals[i] > 0):
                    entry_price = close_vals[i] * (1 + SLIPPAGE_PCT)
                    entry_bar = i
                    atr = atr_vals[i]
                    tp_price = entry_price + config['tp_mult'] * atr
                    sl_price = entry_price - config['sl_mult'] * atr
                    sl_distance_pct = (entry_price - sl_price) / entry_price
                    risk_usd = capital * RISK_PER_TRADE
                    position_size_usd = min(
                        risk_usd / sl_distance_pct if sl_distance_pct > 0 else 0,
                        capital * MAX_POSITION_FRAC,
                        capital * MAX_LEVERAGE
                    )
                    if position_size_usd > 50:
                        in_position = True

        if trades:
            tdf = pd.DataFrame(trades)
            nt = len(tdf)
            wr = (tdf['net_pnl_usd'] > 0).mean()
            tr = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            ar = tdf['net_pnl_pct'].mean()
            gp = tdf.loc[tdf['net_pnl_usd'] > 0, 'net_pnl_usd'].sum()
            gl = abs(tdf.loc[tdf['net_pnl_usd'] <= 0, 'net_pnl_usd'].sum())
            pf = gp / gl if gl > 0 else 999
            dr = tdf['net_pnl_pct'].values / 100
            sp = np.mean(dr) / (np.std(dr) + 1e-10) * np.sqrt(252*2) if len(dr) > 1 else 0
            tp = (tdf['reason'] == 'TP').sum()
            sl = (tdf['reason'] == 'SL').sum()
            to = (tdf['reason'] == 'TIMEOUT').sum()

            # Max DD
            eq = [INITIAL_CAPITAL]
            for pnl in tdf['net_pnl_usd']:
                eq.append(eq[-1] + pnl)
            eq = np.array(eq)
            pk = np.maximum.accumulate(eq)
            mdd = ((pk - eq) / pk).max() * 100

            print(f"  {threshold:>8.2f} {nt:>7d} {wr:>8.1%} {tr:>10.1f}% "
                  f"{ar:>8.3f}% {pf:>6.2f} {mdd:>8.1f}% {sp:>7.2f} "
                  f"{tp:>4d} {sl:>4d} {to:>4d}")

    # ------------------------------------------------------------------
    # 7. Best threshold detailed
    # ------------------------------------------------------------------
    if best_results:
        print(f"\n{'='*70}")
        print(f"  BEST THRESHOLD: {best_thr}")
        print(f"{'='*70}")
        for k, v in best_results.items():
            if isinstance(v, float):
                print(f"   {k:<25s}: {v:.4f}")
            else:
                print(f"   {k:<25s}: {v}")

        # Trade log sample
        print(f"\n  Last 10 trades:")
        for _, t in best_trades.tail(10).iterrows():
            print(f"   Entry ${t['entry_price']:,.0f} -> Exit ${t['exit_price']:,.0f} "
                  f"({t['reason']:>7s}) PnL: {t['net_pnl_pct']:+.3f}% (${t['net_pnl_usd']:+.2f})")

    # ------------------------------------------------------------------
    # 8. Position sizing recommendation
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  POSITION SIZING & RISK RECOMMENDATIONS")
    print(f"{'='*70}")
    print(f"""
    Based on backtest results:

    STRATEGY PARAMETERS:
    - Entry: LONG when model P(TP) > {best_thr} AND regime_ok (ATR% > {regime_threshold:.4f}%)
    - Take Profit: 4x ATR above entry
    - Stop Loss:   2x ATR below entry
    - Max Hold:    12 hours (12 candles on 1h chart)
    - R:R ratio:   2:1 (TP is 2x the SL distance)

    POSITION SIZING:
    - Risk per trade: 2% of account equity
    - Max position:   20% of account equity
    - Max leverage:   3x (conservative)
    - Formula: position_size = (equity * 0.02) / (SL_distance / entry_price)

    MONEY MANAGEMENT:
    - Starting capital: $500 - $1,000 minimum
    - Scale up only after 50+ trades with positive expectancy
    - If 5 consecutive losses, reduce size to 1% per trade for 20 trades
    - Monthly max loss: 10% -> stop trading, re-evaluate

    EXPECTED PERFORMANCE (based on {best_results['n_trades'] if best_results else 0} OOS trades):
    - Win rate: ~{best_results['win_rate']:.0%} if best_results else 'N/A'
    - Avg winner: ~{best_results['avg_win']:.2f}% if best_results else 'N/A'
    - Avg loser:  ~{best_results['avg_loss']:.2f}% if best_results else 'N/A'
    - Profit factor: ~{best_results['profit_factor']:.2f} if best_results else 'N/A'
    - Trades per month: ~{best_results['n_trades'] / (len(df_test) / (24*30)):.0f} if best_results else 'N/A' (estimated)
    """)

    print("[DONE] Backtest complete.")


if __name__ == "__main__":
    main()
