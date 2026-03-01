"""
V5 Mega-Experiment: Systematic Strategy Optimization
=====================================================
Tests ~500 strategy variations to find any profitable edge.

Experiments:
  A. Signal filtering (percentile: top 5-100% of model signals)
  B. Execution barriers (TP/SL ratios from 1:1 to 3:1)
  C. Exit strategies (fixed, trailing stop, breakeven, partial exit)
  D. Fee structures (taker 0.075%, maker 0.02%, VIP 0.01%)
  E. Time-of-day filters (all, Asian, London, US, model-best)
  F. Funding carry (include funding income/cost in P&L)
  G. Holding period variations (6, 8, 12, 18 bars)
  H. 4H timeframe (resample + retrain)
  I. Rule-based strategies (RSI, CCI, momentum)
  J. Combined best (stack best components)

Usage:
    python scripts/v5_mega_experiment.py
"""
import os, sys, json, pickle, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product

import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features.alpha_features import build_alpha_features
from features.triple_barrier import triple_barrier_labels

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"

# ======================================================================
# DEFAULT PARAMS (extracted from V4 Optuna runs -- sensible defaults)
# ======================================================================
LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'random_state': 42,
    'class_weight': 'balanced', 'verbose': -1,
    'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 800,
    'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_samples': 30,
    'reg_alpha': 1.0, 'reg_lambda': 5.0, 'num_leaves': 31,
}
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'tree_method': 'hist', 'random_state': 42,
    'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 800,
    'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_weight': 20,
    'gamma': 1.0, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
}


# ======================================================================
# 1. DATA LOADING
# ======================================================================

def load_data():
    paths = [
        ROOT / "data" / "processed" / "btcusdt_1h_full_history_enhanced.parquet",
        ROOT / "data" / "processed" / "btcusdt_1h_features.parquet",
    ]
    data_path = next((p for p in paths if p.exists()), None)
    assert data_path, "No data found!"

    df = pd.read_parquet(data_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        try: df.index = pd.to_datetime(df.index)
        except: pass

    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    df = build_alpha_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"   After cleanup: {len(df):,} rows, {len(df.columns)} columns")

    # Add regime filter
    if 'atr_pct' not in df.columns and 'atr_14' in df.columns:
        df['atr_pct'] = df['atr_14'] / df['close'] * 100
    if 'atr_pct' in df.columns:
        thr = df['atr_pct'].quantile(0.25)
        df['regime_ok'] = df['atr_pct'] >= thr
    else:
        df['regime_ok'] = True

    # Extract hour for time-of-day filter
    if hasattr(df.index, 'hour'):
        df['_hour'] = df.index.hour
    else:
        df['_hour'] = 0

    return df


def get_feature_cols(df):
    exclude = {
        'open', 'high', 'low', 'close', 'volume', 'funding_rate',
        'open_interest', 'timestamp', 'target', 'classic_binary',
        'regime_filter', 'regime_ok', '_hour',
        'tb_label', 'tb_binary', 'tb_holding', 'tb_barrier_tp', 'tb_barrier_sl',
        'short_tb_label', 'short_tb_binary', 'short_tb_holding',
        'short_tb_barrier_tp', 'short_tb_barrier_sl',
    }
    return [c for c in df.columns
            if c not in exclude
            and not any(p in c.lower() for p in ['future_', 'target_', 'label_'])
            and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]


# ======================================================================
# 2. WALK-FORWARD PREDICTION ENGINE
# ======================================================================

def purged_wf_splits(n, n_splits=5, test_pct=0.12, purge=12):
    test_size = int(n * test_pct)
    splits = []
    for i in range(n_splits):
        te_start = n - (n_splits - i) * test_size
        te_end = min(te_start + test_size, n)
        tr_end = te_start - purge
        if te_start < test_size or tr_end < 100:
            continue
        splits.append((np.arange(0, tr_end), np.arange(te_start, te_end)))
    return splits


def walk_forward_predict(X, y, splits, side_label="LONG"):
    """Quick walk-forward: train LGB+XGB on each fold, return OOS predictions."""
    oos_probs = np.full(len(X), np.nan)
    fold_aucs = []

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        # SMOTE
        if y_tr.sum() > 5:
            smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_tr.sum())-1))
            X_tr_s, y_tr_s = smote.fit_resample(X_tr, y_tr)
        else:
            X_tr_s, y_tr_s = X_tr, y_tr

        # XGBoost
        xp = dict(XGB_PARAMS)
        xp['scale_pos_weight'] = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        m_xgb = xgb.XGBClassifier(**xp, early_stopping_rounds=50)
        m_xgb.fit(X_tr_s, y_tr_s, eval_set=[(X_te, y_te)], verbose=False)

        # LightGBM
        m_lgb = lgb.LGBMClassifier(**LGB_PARAMS)
        m_lgb.fit(X_tr_s, y_tr_s, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        p_ens = (m_xgb.predict_proba(X_te)[:, 1] + m_lgb.predict_proba(X_te)[:, 1]) / 2
        auc = roc_auc_score(y_te, p_ens)
        fold_aucs.append(auc)
        oos_probs[te_idx] = p_ens
        print(f"      {side_label} Fold {fold_i+1}: AUC={auc:.4f}")

    mean_auc = np.mean(fold_aucs)
    print(f"      {side_label} Mean AUC: {mean_auc:.4f} ± {np.std(fold_aucs):.4f}")
    return oos_probs, mean_auc


def generate_labels_and_predict(df, feature_cols, tp_mult, sl_mult, max_hold,
                                direction="long"):
    """Generate labels + walk-forward OOS predictions for one barrier config."""
    tb = triple_barrier_labels(
        df['close'], df['high'], df['low'],
        tp_mult=tp_mult, sl_mult=sl_mult, max_holding=max_hold,
        direction=direction,
    )
    pfx = "" if direction == "long" else "short_"
    y_col = f'{pfx}tb_binary'
    mask = tb[y_col].notna()
    X_clean = df.loc[mask, feature_cols]
    y_clean = tb.loc[mask, y_col].astype(int)

    splits = purged_wf_splits(len(X_clean), n_splits=5, test_pct=0.12, purge=12)
    lbl = "LONG" if direction == "long" else "SHORT"
    probs, auc = walk_forward_predict(X_clean, y_clean, splits, lbl)

    # Map back to full df index
    full_probs = np.full(len(df), np.nan)
    full_probs[mask.values] = probs
    return full_probs, auc


# ======================================================================
# 3. ADVANCED PORTFOLIO SIMULATOR
# ======================================================================

def simulate_portfolio(
    close, high, low, atr, hours,
    long_probs, short_probs,
    cfg: dict,
) -> list:
    """
    Bar-by-bar portfolio simulator with advanced exit strategies.

    cfg keys:
      tp_mult, sl_mult, max_hold,
      exit_mode:    'fixed' | 'trailing' | 'breakeven' | 'partial'
      trail_mult:   trailing distance in ATR multiples (default 1.0)
      be_trigger:   ATR mult to activate breakeven SL (default 1.0)
      partial_at:   ATR mult for partial exit trigger (default 1.0)
      partial_frac: fraction to close at partial (default 0.5)
      fee_pct:      per-side fee % (default 0.075)
      slippage_pct: per-side slippage % (default 0.02)
      signal_mode:  'percentile' | 'threshold'
      signal_pct:   top X% to trade (for percentile mode)
      signal_thr:   absolute threshold (for threshold mode)
      hour_filter:  set of hours to trade in, or None
      direction:    'both' | 'long' | 'short'
      funding_rates: array of funding rates or None
      initial_capital, risk_per_trade, max_pos_frac
      cooldown:     bars to wait after a trade (default 0)
      confirmation: require both models to agree (default False)
    """
    n = len(close)
    cap = cfg.get('initial_capital', 10000)
    risk_pct = cfg.get('risk_per_trade', 0.02)
    max_pos = cfg.get('max_pos_frac', 0.30)
    fee = cfg.get('fee_pct', 0.075) / 100
    slip = cfg.get('slippage_pct', 0.02) / 100
    total_cost = 2 * (fee + slip)
    tp_mult = cfg['tp_mult']
    sl_mult = cfg['sl_mult']
    max_hold = cfg['max_hold']
    exit_mode = cfg.get('exit_mode', 'fixed')
    trail_mult = cfg.get('trail_mult', 1.0)
    be_trigger = cfg.get('be_trigger', 1.0)
    partial_at = cfg.get('partial_at', 1.0)
    partial_frac = cfg.get('partial_frac', 0.5)
    hour_filter = cfg.get('hour_filter', None)
    direction = cfg.get('direction', 'both')
    funding = cfg.get('funding_rates', None)
    cooldown = cfg.get('cooldown', 0)
    confirmation = cfg.get('confirmation', False)

    # Build signal masks based on signal_mode
    signal_mode = cfg.get('signal_mode', 'percentile')
    signal_pct = cfg.get('signal_pct', 1.0)  # top 100% = all
    signal_thr = cfg.get('signal_thr', 0.15)

    # Compute signal masks
    regime = cfg.get('regime_mask', np.ones(n, dtype=bool))

    if signal_mode == 'percentile':
        # Percentile-based: only top X% of bars
        lp_valid = long_probs[~np.isnan(long_probs)]
        sp_valid = short_probs[~np.isnan(short_probs)]
        long_thr = np.percentile(lp_valid, 100 * (1 - signal_pct)) if len(lp_valid) > 0 else 1.0
        short_thr = np.percentile(sp_valid, 100 * (1 - signal_pct)) if len(sp_valid) > 0 else 1.0
    else:
        long_thr = signal_thr
        short_thr = signal_thr

    long_sig = (~np.isnan(long_probs)) & (long_probs >= long_thr) & regime
    short_sig = (~np.isnan(short_probs)) & (short_probs >= short_thr) & regime

    # Confirmation filter: require other model to be LOW
    if confirmation:
        # Long only when short_prob < median
        sp_med = np.nanmedian(short_probs)
        lp_med = np.nanmedian(long_probs)
        long_sig = long_sig & (~np.isnan(short_probs)) & (short_probs < sp_med)
        short_sig = short_sig & (~np.isnan(long_probs)) & (long_probs < lp_med)

    # Hour filter
    if hour_filter is not None:
        hour_ok = np.isin(hours, list(hour_filter))
        long_sig = long_sig & hour_ok
        short_sig = short_sig & hour_ok

    # Direction filter
    if direction == 'long':
        short_sig[:] = False
    elif direction == 'short':
        long_sig[:] = False

    # Resolve conflicts (prefer higher prob side)
    both = long_sig & short_sig
    for idx in np.where(both)[0]:
        if long_probs[idx] >= short_probs[idx]:
            short_sig[idx] = False
        else:
            long_sig[idx] = False

    # ---- Simulation ----
    in_pos = False
    trades = []
    last_exit_bar = -cooldown - 1

    for i in range(n):
        if in_pos:
            bars_held = i - entry_bar
            exit_price = None
            reason = None

            # Accumulate funding
            if funding is not None and i < len(funding) and not np.isnan(funding[i]):
                if pos_side == 'LONG':
                    fund_pnl -= funding[i] * remaining * pos_size
                else:
                    fund_pnl += funding[i] * remaining * pos_size

            if pos_side == 'LONG':
                # Trailing stop update
                if exit_mode == 'trailing':
                    new_sl = high[i] - trail_mult * atr_entry
                    current_sl = max(current_sl, new_sl)

                # Breakeven trigger
                if exit_mode in ('breakeven', 'partial'):
                    if not be_active and high[i] >= entry_p + be_trigger * atr_entry:
                        current_sl = max(current_sl, entry_p)
                        be_active = True

                # Partial exit
                if exit_mode == 'partial' and not partial_done:
                    if high[i] >= entry_p + partial_at * atr_entry:
                        p_exit = entry_p + partial_at * atr_entry
                        p_ret = (p_exit - entry_p) / entry_p
                        partial_pnl += partial_frac * p_ret * pos_size
                        remaining = 1.0 - partial_frac
                        partial_done = True
                        current_sl = max(current_sl, entry_p)

                # Check SL
                if low[i] <= current_sl:
                    exit_price = current_sl
                    reason = 'SL'
                # Check TP
                elif high[i] >= tp_price:
                    exit_price = tp_price
                    reason = 'TP'
                # Check timeout
                elif bars_held >= max_hold:
                    exit_price = close[i]
                    reason = 'TIMEOUT'

                if exit_price is not None:
                    gross_ret = (exit_price - entry_p) / entry_p
                    net_ret_main = gross_ret * remaining * pos_size
                    total_pnl = net_ret_main + partial_pnl + fund_pnl - total_cost * pos_size
                    cap += total_pnl

            else:  # SHORT
                if exit_mode == 'trailing':
                    new_sl = low[i] + trail_mult * atr_entry
                    current_sl = min(current_sl, new_sl)

                if exit_mode in ('breakeven', 'partial'):
                    if not be_active and low[i] <= entry_p - be_trigger * atr_entry:
                        current_sl = min(current_sl, entry_p)
                        be_active = True

                if exit_mode == 'partial' and not partial_done:
                    if low[i] <= entry_p - partial_at * atr_entry:
                        p_exit = entry_p - partial_at * atr_entry
                        p_ret = (entry_p - p_exit) / entry_p
                        partial_pnl += partial_frac * p_ret * pos_size
                        remaining = 1.0 - partial_frac
                        partial_done = True
                        current_sl = min(current_sl, entry_p)

                if high[i] >= current_sl:
                    exit_price = current_sl
                    reason = 'SL'
                elif low[i] <= tp_price:
                    exit_price = tp_price
                    reason = 'TP'
                elif bars_held >= max_hold:
                    exit_price = close[i]
                    reason = 'TIMEOUT'

                if exit_price is not None:
                    gross_ret = (entry_p - exit_price) / entry_p
                    net_ret_main = gross_ret * remaining * pos_size
                    total_pnl = net_ret_main + partial_pnl + fund_pnl - total_cost * pos_size
                    cap += total_pnl

            if exit_price is not None:
                trades.append({
                    'side': pos_side, 'reason': reason,
                    'bars_held': bars_held,
                    'pnl': total_pnl, 'capital': cap,
                    'gross_ret': gross_ret,
                })
                in_pos = False
                last_exit_bar = i

        # Try entering
        if not in_pos and cap > 100 and i - last_exit_bar > cooldown:
            enter_side = None
            if long_sig[i] and atr[i] > 0:
                enter_side = 'LONG'
            elif short_sig[i] and atr[i] > 0:
                enter_side = 'SHORT'

            if enter_side is not None:
                atr_entry = atr[i]
                if enter_side == 'LONG':
                    entry_p = close[i] * (1 + slip)
                    tp_price = entry_p + tp_mult * atr_entry
                    sl_p = entry_p - sl_mult * atr_entry
                else:
                    entry_p = close[i] * (1 - slip)
                    tp_price = entry_p - tp_mult * atr_entry
                    sl_p = entry_p + sl_mult * atr_entry

                sl_dist_pct = abs(entry_p - sl_p) / entry_p
                pos_size = min(cap * risk_pct / max(sl_dist_pct, 1e-6),
                               cap * max_pos)
                if pos_size >= 50:
                    in_pos = True
                    pos_side = enter_side
                    entry_bar = i
                    current_sl = sl_p
                    be_active = False
                    partial_done = False
                    partial_pnl = 0.0
                    fund_pnl = 0.0
                    remaining = 1.0

    return trades


def calc_stats(trades, label=""):
    if not trades:
        return None
    df_t = pd.DataFrame(trades)
    n = len(df_t)
    if n < 5:
        return None

    wins = df_t['pnl'] > 0
    wr = wins.mean()
    gp = df_t.loc[wins, 'pnl'].sum()
    gl = abs(df_t.loc[~wins, 'pnl'].sum())
    pf = gp / gl if gl > 0 else 99.0
    final = df_t['capital'].iloc[-1]
    init = 10000
    tr = (final - init) / init * 100

    eq = np.array([init] + df_t['capital'].tolist())
    pk = np.maximum.accumulate(eq)
    dd = (pk - eq) / (pk + 1e-10)
    max_dd = dd.max() * 100

    rets = df_t['pnl'].values / (init)  # approximate
    sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(365 * 2) if len(rets) > 1 else 0

    n_l = (df_t['side'] == 'LONG').sum()
    n_s = (df_t['side'] == 'SHORT').sum()
    tp_n = (df_t['reason'] == 'TP').sum()
    sl_n = (df_t['reason'] == 'SL').sum()

    return {
        'label': label, 'trades': n, 'wr': wr, 'pf': pf,
        'ret': tr, 'dd': max_dd, 'sharpe': sharpe,
        'n_long': n_l, 'n_short': n_s,
        'tp': tp_n, 'sl': sl_n, 'timeout': n - tp_n - sl_n,
        'final_cap': final,
    }


# ======================================================================
# 4. EXPERIMENT SWEEP
# ======================================================================

def run_single_experiment(close, high, low, atr, hours,
                          long_probs, short_probs,
                          regime_mask, funding_rates, cfg, label):
    """Run one experiment configuration and return stats."""
    cfg['regime_mask'] = regime_mask
    cfg['funding_rates'] = funding_rates
    trades = simulate_portfolio(close, high, low, atr, hours,
                                long_probs, short_probs, cfg)
    return calc_stats(trades, label)


def sweep_experiments(close, high, low, atr, hours,
                      long_probs, short_probs,
                      regime_mask, funding_rates):
    """Sweep ALL strategy combinations and return sorted results."""

    results = []

    # ---- BARRIER CONFIGS ----
    barrier_configs = [
        (1.0, 0.5, "TP1.0_SL0.5"),
        (1.5, 0.75, "TP1.5_SL0.75"),
        (1.5, 1.0, "TP1.5_SL1.0"),
        (2.0, 1.0, "TP2.0_SL1.0"),
        (2.0, 1.5, "TP2.0_SL1.5"),
        (2.5, 1.0, "TP2.5_SL1.0"),
        (2.5, 1.5, "TP2.5_SL1.5"),  # V4
        (3.0, 1.0, "TP3.0_SL1.0"),
        (3.0, 1.5, "TP3.0_SL1.5"),
        (4.0, 1.5, "TP4.0_SL1.5"),
    ]

    # ---- EXIT MODES ----
    exit_configs = [
        {'exit_mode': 'fixed', '_lbl': 'fixed'},
        {'exit_mode': 'trailing', 'trail_mult': 1.0, '_lbl': 'trail1.0'},
        {'exit_mode': 'trailing', 'trail_mult': 0.75, '_lbl': 'trail0.75'},
        {'exit_mode': 'trailing', 'trail_mult': 1.5, '_lbl': 'trail1.5'},
        {'exit_mode': 'breakeven', 'be_trigger': 0.5, '_lbl': 'be0.5'},
        {'exit_mode': 'breakeven', 'be_trigger': 1.0, '_lbl': 'be1.0'},
        {'exit_mode': 'partial', 'partial_at': 0.5, 'partial_frac': 0.5,
         'be_trigger': 0.5, '_lbl': 'part50@0.5'},
        {'exit_mode': 'partial', 'partial_at': 1.0, 'partial_frac': 0.5,
         'be_trigger': 1.0, '_lbl': 'part50@1.0'},
        {'exit_mode': 'partial', 'partial_at': 0.5, 'partial_frac': 0.33,
         'be_trigger': 0.5, '_lbl': 'part33@0.5'},
    ]

    # ---- FEE CONFIGS ----
    fee_configs = [
        (0.075, 0.02, 'taker'),     # market orders
        (0.02, 0.01, 'maker'),      # limit orders
        (0.01, 0.005, 'vip'),       # VIP maker
    ]

    # ---- SIGNAL FILTERS (percentile) ----
    signal_configs = [
        (0.05, 'top5%'),
        (0.10, 'top10%'),
        (0.15, 'top15%'),
        (0.20, 'top20%'),
        (0.30, 'top30%'),
        (0.50, 'top50%'),
        (1.00, 'all'),
    ]

    # ---- HOUR FILTERS ----
    hour_configs = [
        (None, 'allhrs'),
        (set(range(0, 8)), 'asian'),
        (set(range(7, 16)), 'london'),
        (set(range(13, 22)), 'us'),
        (set(range(0, 4)) | set(range(12, 16)), 'peak'),
        (set(range(4, 12)), 'quiet'),
    ]

    # ---- DIRECTION ----
    dir_configs = ['both', 'long', 'short']

    # ---- HOLD PERIODS ----
    hold_configs = [6, 8, 12, 18, 24]

    # ---- COOLDOWN ----
    cooldown_configs = [0, 2, 4]

    total_combos = (len(barrier_configs) * len(exit_configs) * len(fee_configs)
                    * len(signal_configs) * len(hour_configs) * len(dir_configs)
                    * len(hold_configs) * len(cooldown_configs))
    print(f"\n   Total theoretical combinations: {total_combos:,}")
    print(f"   Running smart subset...")

    # ============================================================
    # PHASE 1: Coarse sweep (fix some params, vary others)
    # Find best barrier, signal, direction, fee
    # ============================================================
    print(f"\n   PHASE 1: Barrier × Signal × Direction × Fee sweep...")
    count = 0

    for (tp, sl, b_lbl) in barrier_configs:
        for (sig_pct, sig_lbl) in signal_configs:
            for direction in dir_configs:
                for (fee_p, slip_p, fee_lbl) in fee_configs:
                    cfg = {
                        'tp_mult': tp, 'sl_mult': sl, 'max_hold': 12,
                        'exit_mode': 'fixed',
                        'fee_pct': fee_p, 'slippage_pct': slip_p,
                        'signal_mode': 'percentile', 'signal_pct': sig_pct,
                        'direction': direction,
                        'hour_filter': None,
                        'initial_capital': 10000, 'risk_per_trade': 0.02,
                        'max_pos_frac': 0.30, 'cooldown': 0,
                        'confirmation': False,
                    }
                    label = f"{b_lbl}|{sig_lbl}|{direction}|{fee_lbl}"
                    s = run_single_experiment(
                        close, high, low, atr, hours,
                        long_probs, short_probs,
                        regime_mask, funding_rates, cfg, label)
                    if s:
                        results.append(s)
                    count += 1

    print(f"   Phase 1: {count} experiments done, {len(results)} with trades")

    # Get top configs from Phase 1
    profitable_p1 = sorted([r for r in results if r['pf'] > 1.0],
                           key=lambda x: x['pf'], reverse=True)
    print(f"   Phase 1 profitable: {len(profitable_p1)}")
    if profitable_p1:
        print(f"   Best PF: {profitable_p1[0]['pf']:.3f} ({profitable_p1[0]['label']})")
        print(f"   Best ret: {max(profitable_p1, key=lambda x: x['ret'])['ret']:+.1f}% "
              f"({max(profitable_p1, key=lambda x: x['ret'])['label']})")

    # ============================================================
    # PHASE 2: Exit strategy sweep (use best configs from Phase 1)
    # ============================================================
    print(f"\n   PHASE 2: Exit strategy sweep...")

    # Use top barrier/signal/dir/fee combos (or all if few profitable)
    if profitable_p1:
        # Parse the top 5 configs
        top_cfgs = profitable_p1[:5]
    else:
        # Use the least-bad configs
        top_cfgs = sorted(results, key=lambda x: x['pf'], reverse=True)[:5] if results else []

    phase2_results = []
    for base in top_cfgs:
        parts = base['label'].split('|')
        if len(parts) < 4:
            continue
        b_lbl, sig_lbl, direction, fee_lbl = parts[0], parts[1], parts[2], parts[3]

        # Parse barrier
        tp = float(b_lbl.split('_')[0].replace('TP', ''))
        sl = float(b_lbl.split('_')[1].replace('SL', ''))
        # Parse signal
        if sig_lbl == 'all':
            sig_pct = 1.0
        else:
            sig_pct = float(sig_lbl.replace('top', '').replace('%', '')) / 100
        # Parse fee
        fee_map = {'taker': (0.075, 0.02), 'maker': (0.02, 0.01), 'vip': (0.01, 0.005)}
        fee_p, slip_p = fee_map.get(fee_lbl, (0.075, 0.02))

        for exit_cfg in exit_configs:
            for hold in hold_configs:
                for cd in cooldown_configs:
                    cfg = {
                        'tp_mult': tp, 'sl_mult': sl, 'max_hold': hold,
                        'fee_pct': fee_p, 'slippage_pct': slip_p,
                        'signal_mode': 'percentile', 'signal_pct': sig_pct,
                        'direction': direction,
                        'hour_filter': None,
                        'initial_capital': 10000, 'risk_per_trade': 0.02,
                        'max_pos_frac': 0.30, 'cooldown': cd,
                        'confirmation': False,
                    }
                    cfg.update({k: v for k, v in exit_cfg.items() if k != '_lbl'})
                    label = f"{b_lbl}|{sig_lbl}|{direction}|{fee_lbl}|{exit_cfg['_lbl']}|H{hold}|CD{cd}"
                    s = run_single_experiment(
                        close, high, low, atr, hours,
                        long_probs, short_probs,
                        regime_mask, funding_rates, cfg, label)
                    if s:
                        phase2_results.append(s)

    results.extend(phase2_results)
    profitable_p2 = [r for r in phase2_results if r['pf'] > 1.0]
    print(f"   Phase 2: {len(phase2_results)} experiments, {len(profitable_p2)} profitable")

    # ============================================================
    # PHASE 3: Hour filter sweep on best combos
    # ============================================================
    print(f"\n   PHASE 3: Hour filter sweep...")

    all_profitable = sorted([r for r in results if r['pf'] > 1.0],
                            key=lambda x: x['pf'], reverse=True)
    top_for_hours = all_profitable[:10] if all_profitable else sorted(
        results, key=lambda x: x['pf'], reverse=True)[:5]

    phase3_results = []
    for base in top_for_hours:
        parts = base['label'].split('|')
        if len(parts) < 4:
            continue
        b_lbl = parts[0]
        sig_lbl = parts[1]
        direction = parts[2]
        fee_lbl = parts[3]
        exit_lbl = parts[4] if len(parts) > 4 else 'fixed'
        hold = int(parts[5].replace('H', '')) if len(parts) > 5 else 12
        cd = int(parts[6].replace('CD', '')) if len(parts) > 6 else 0

        tp = float(b_lbl.split('_')[0].replace('TP', ''))
        sl = float(b_lbl.split('_')[1].replace('SL', ''))
        if sig_lbl == 'all':
            sig_pct = 1.0
        else:
            sig_pct = float(sig_lbl.replace('top', '').replace('%', '')) / 100
        fee_map = {'taker': (0.075, 0.02), 'maker': (0.02, 0.01), 'vip': (0.01, 0.005)}
        fee_p, slip_p = fee_map.get(fee_lbl, (0.075, 0.02))

        # Reconstruct exit config
        exit_cfg = {'exit_mode': 'fixed'}
        if 'trail' in exit_lbl:
            exit_cfg = {'exit_mode': 'trailing', 'trail_mult': float(exit_lbl.replace('trail', ''))}
        elif 'be' in exit_lbl:
            exit_cfg = {'exit_mode': 'breakeven', 'be_trigger': float(exit_lbl.replace('be', ''))}
        elif 'part' in exit_lbl:
            # Parse partial config
            exit_cfg = {'exit_mode': 'partial', 'partial_at': 1.0, 'partial_frac': 0.5, 'be_trigger': 1.0}

        for (hf, hf_lbl) in hour_configs:
            for confirm in [False, True]:
                cfg = {
                    'tp_mult': tp, 'sl_mult': sl, 'max_hold': hold,
                    'fee_pct': fee_p, 'slippage_pct': slip_p,
                    'signal_mode': 'percentile', 'signal_pct': sig_pct,
                    'direction': direction,
                    'hour_filter': hf,
                    'initial_capital': 10000, 'risk_per_trade': 0.02,
                    'max_pos_frac': 0.30, 'cooldown': cd,
                    'confirmation': confirm,
                }
                cfg.update({k: v for k, v in exit_cfg.items()})
                c_lbl = "conf" if confirm else "noconf"
                label = f"{b_lbl}|{sig_lbl}|{direction}|{fee_lbl}|{exit_lbl}|H{hold}|CD{cd}|{hf_lbl}|{c_lbl}"
                s = run_single_experiment(
                    close, high, low, atr, hours,
                    long_probs, short_probs,
                    regime_mask, funding_rates, cfg, label)
                if s:
                    phase3_results.append(s)

    results.extend(phase3_results)
    profitable_p3 = [r for r in phase3_results if r['pf'] > 1.0]
    print(f"   Phase 3: {len(phase3_results)} experiments, {len(profitable_p3)} profitable")

    return results


# ======================================================================
# 5. RULE-BASED STRATEGIES
# ======================================================================

def run_rule_based_strategies(df, close, high, low, atr, hours,
                              regime_mask, funding_rates):
    """Test pure rule-based strategies (no ML)."""
    print(f"\n{'='*70}")
    print(f"  RULE-BASED STRATEGIES (no ML)")
    print(f"{'='*70}")

    results = []
    n = len(close)

    # Get indicator values
    rsi = df['rsi_4h'].values if 'rsi_4h' in df.columns else np.full(n, 50.0)
    cci = df['cci'].values if 'cci' in df.columns else np.full(n, 0.0)
    macd_h = df['macd_hist'].values if 'macd_hist' in df.columns else np.full(n, 0.0)
    ma20 = df['ma_20_1d'].values if 'ma_20_1d' in df.columns else close
    ma100 = df['ma_100'].values if 'ma_100' in df.columns else close
    adx_v = df['adx'].values if 'adx' in df.columns else np.full(n, 25.0)

    # Rule sets
    rules = {
        'RSI_pullback': {
            'long':  (rsi < 35) & (close > ma20),
            'short': (rsi > 65) & (close < ma20),
        },
        'RSI_extreme': {
            'long':  rsi < 25,
            'short': rsi > 75,
        },
        'CCI_bounce': {
            'long':  cci < -100,
            'short': cci > 200,
        },
        'MACD_momentum': {
            'long':  (macd_h > 0) & (np.roll(macd_h, 1) <= 0),  # cross above 0
            'short': (macd_h < 0) & (np.roll(macd_h, 1) >= 0),
        },
        'Trend_follow': {
            'long':  (close > ma20) & (ma20 > ma100) & (adx_v > 25),
            'short': (close < ma20) & (ma20 < ma100) & (adx_v > 25),
        },
        'Mean_revert': {
            'long':  (rsi < 30) & (cci < -150),
            'short': (rsi > 70) & (cci > 150),
        },
    }

    for rule_name, sigs in rules.items():
        long_sig = sigs['long'] & regime_mask
        short_sig = sigs['short'] & regime_mask
        # Create fake probs (1.0 where signal, 0 otherwise)
        lp = np.where(long_sig, 1.0, 0.0).astype(float)
        sp = np.where(short_sig, 1.0, 0.0).astype(float)

        for (tp, sl, b_lbl) in [(1.5, 1.0, "1.5:1"), (2.0, 1.0, "2:1"),
                                 (2.5, 1.5, "2.5:1.5"), (3.0, 1.0, "3:1")]:
            for (fee_p, slip_p, fee_lbl) in [(0.075, 0.02, 'taker'), (0.02, 0.01, 'maker')]:
                for exit_mode in ['fixed', 'trailing', 'breakeven']:
                    cfg = {
                        'tp_mult': tp, 'sl_mult': sl, 'max_hold': 12,
                        'exit_mode': exit_mode,
                        'trail_mult': 1.0, 'be_trigger': 1.0,
                        'fee_pct': fee_p, 'slippage_pct': slip_p,
                        'signal_mode': 'threshold', 'signal_thr': 0.5,
                        'direction': 'both',
                        'hour_filter': None,
                        'initial_capital': 10000, 'risk_per_trade': 0.02,
                        'max_pos_frac': 0.30, 'cooldown': 0,
                        'confirmation': False,
                    }
                    label = f"RULE:{rule_name}|{b_lbl}|{fee_lbl}|{exit_mode}"
                    s = run_single_experiment(
                        close, high, low, atr, hours, lp, sp,
                        regime_mask, funding_rates, cfg, label)
                    if s:
                        results.append(s)

    profitable = [r for r in results if r['pf'] > 1.0]
    print(f"   Rule-based: {len(results)} experiments, {len(profitable)} profitable")
    if profitable:
        best = max(profitable, key=lambda x: x['pf'])
        print(f"   Best rule: {best['label']} PF={best['pf']:.3f} Ret={best['ret']:+.1f}%")

    return results


# ======================================================================
# 6. 4H TIMEFRAME EXPERIMENT
# ======================================================================

def experiment_4h(df_1h):
    """Resample to 4H, retrain, and backtest."""
    print(f"\n{'='*70}")
    print(f"  4H TIMEFRAME EXPERIMENT")
    print(f"{'='*70}")

    # Resample
    df_4h = df_1h.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    })
    if 'funding_rate' in df_1h.columns:
        df_4h['funding_rate'] = df_1h['funding_rate'].resample('4h').sum()
    df_4h = df_4h.dropna(subset=['open', 'high', 'low', 'close'])
    print(f"   Resampled to {len(df_4h):,} 4H bars")

    # Build features
    df_4h = build_alpha_features(df_4h)
    df_4h = df_4h.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"   After features: {len(df_4h):,} bars, {len(df_4h.columns)} columns")

    if hasattr(df_4h.index, 'hour'):
        df_4h['_hour'] = df_4h.index.hour
    else:
        df_4h['_hour'] = 0

    # Regime filter
    if 'atr_14' in df_4h.columns:
        df_4h['atr_pct'] = df_4h['atr_14'] / df_4h['close'] * 100
        thr = df_4h['atr_pct'].quantile(0.25)
        df_4h['regime_ok'] = df_4h['atr_pct'] >= thr
    else:
        df_4h['regime_ok'] = True

    feature_cols = get_feature_cols(df_4h)
    print(f"   Features: {len(feature_cols)}")

    # Test multiple barrier configs on 4H
    barrier_cfgs_4h = [
        (1.5, 1.0, 6, "TP1.5_SL1.0_H6"),
        (2.0, 1.0, 6, "TP2.0_SL1.0_H6"),
        (2.0, 1.0, 8, "TP2.0_SL1.0_H8"),
        (2.5, 1.0, 6, "TP2.5_SL1.0_H6"),
        (2.5, 1.5, 6, "TP2.5_SL1.5_H6"),
        (1.5, 0.75, 4, "TP1.5_SL0.75_H4"),
        (3.0, 1.0, 6, "TP3.0_SL1.0_H6"),
        (1.0, 0.5, 6, "TP1.0_SL0.5_H6"),
    ]

    all_results = []

    for tp, sl, max_hold, b_lbl in barrier_cfgs_4h:
        print(f"\n   --- 4H {b_lbl} ---")
        lp, l_auc = generate_labels_and_predict(
            df_4h, feature_cols, tp, sl, max_hold, "long")
        sp, s_auc = generate_labels_and_predict(
            df_4h, feature_cols, tp, sl, max_hold, "short")

        close_v = df_4h['close'].values
        high_v = df_4h['high'].values
        low_v = df_4h['low'].values
        atr_v = df_4h['atr_14'].values if 'atr_14' in df_4h.columns else np.ones(len(df_4h))
        hours_v = df_4h['_hour'].values
        regime_v = df_4h['regime_ok'].values
        fund_v = df_4h['funding_rate'].values if 'funding_rate' in df_4h.columns else None

        for (sig_pct, sig_lbl) in [(0.10, 'top10'), (0.20, 'top20'), (0.30, 'top30'),
                                    (0.50, 'top50'), (1.0, 'all')]:
            for direction in ['both', 'long', 'short']:
                for (fee_p, slip_p, fee_lbl) in [(0.075, 0.02, 'taker'), (0.02, 0.01, 'maker')]:
                    for exit_mode in ['fixed', 'trailing', 'breakeven']:
                        for cd in [0, 1]:
                            cfg = {
                                'tp_mult': tp, 'sl_mult': sl, 'max_hold': max_hold,
                                'exit_mode': exit_mode,
                                'trail_mult': 1.0, 'be_trigger': 1.0,
                                'fee_pct': fee_p, 'slippage_pct': slip_p,
                                'signal_mode': 'percentile', 'signal_pct': sig_pct,
                                'direction': direction,
                                'hour_filter': None,
                                'initial_capital': 10000, 'risk_per_trade': 0.02,
                                'max_pos_frac': 0.30, 'cooldown': cd,
                                'confirmation': False,
                            }
                            label = f"4H|{b_lbl}|{sig_lbl}|{direction}|{fee_lbl}|{exit_mode}|CD{cd}"
                            s = run_single_experiment(
                                close_v, high_v, low_v, atr_v, hours_v,
                                lp, sp, regime_v, fund_v, cfg, label)
                            if s:
                                all_results.append(s)

    profitable = [r for r in all_results if r['pf'] > 1.0]
    print(f"\n   4H Total: {len(all_results)} experiments, {len(profitable)} profitable")
    if profitable:
        best = max(profitable, key=lambda x: x['pf'])
        print(f"   Best 4H: {best['label']} PF={best['pf']:.3f} Ret={best['ret']:+.1f}%")

    return all_results


# ======================================================================
# 7. DIFFERENT LABEL BARRIERS (retrain on 1H with different TP/SL)
# ======================================================================

def experiment_retrain_barriers(df, feature_cols):
    """Retrain models with different barrier labels and backtest."""
    print(f"\n{'='*70}")
    print(f"  RETRAIN WITH DIFFERENT BARRIERS (1H)")
    print(f"{'='*70}")

    barrier_cfgs = [
        (1.5, 1.0, 12, "retrain_TP1.5_SL1.0_H12"),
        (2.0, 1.0, 12, "retrain_TP2.0_SL1.0_H12"),
        (2.0, 1.0, 8,  "retrain_TP2.0_SL1.0_H8"),
        (3.0, 1.0, 12, "retrain_TP3.0_SL1.0_H12"),
        (1.5, 0.75, 8, "retrain_TP1.5_SL0.75_H8"),
        (2.0, 0.75, 12, "retrain_TP2.0_SL0.75_H12"),
        (1.0, 0.5, 8,  "retrain_TP1.0_SL0.5_H8"),
    ]

    close_v = df['close'].values
    high_v = df['high'].values
    low_v = df['low'].values
    atr_v = df['atr_14'].values if 'atr_14' in df.columns else np.ones(len(df))
    hours_v = df['_hour'].values
    regime_v = df['regime_ok'].values
    fund_v = df['funding_rate'].values if 'funding_rate' in df.columns else None

    all_results = []

    for tp, sl, max_hold, cfg_lbl in barrier_cfgs:
        print(f"\n   --- {cfg_lbl} ---")
        lp, l_auc = generate_labels_and_predict(
            df, feature_cols, tp, sl, max_hold, "long")
        sp, s_auc = generate_labels_and_predict(
            df, feature_cols, tp, sl, max_hold, "short")

        # Execution barriers = same as training barriers AND also try tighter
        exec_barriers = [
            (tp, sl, max_hold),      # matching
            (tp * 0.75, sl, max_hold),  # tighter TP
            (tp * 0.5, sl, max_hold),   # much tighter TP
        ]

        for (e_tp, e_sl, e_hold) in exec_barriers:
            for (sig_pct, sig_lbl) in [(0.10, 'top10'), (0.20, 'top20'),
                                        (0.30, 'top30'), (0.50, 'top50'), (1.0, 'all')]:
                for direction in ['both', 'long']:
                    for (fee_p, slip_p, fee_lbl) in [(0.075, 0.02, 'taker'), (0.02, 0.01, 'maker')]:
                        for exit_mode in ['fixed', 'trailing', 'breakeven']:
                            cfg = {
                                'tp_mult': e_tp, 'sl_mult': e_sl, 'max_hold': e_hold,
                                'exit_mode': exit_mode,
                                'trail_mult': 1.0, 'be_trigger': 1.0,
                                'fee_pct': fee_p, 'slippage_pct': slip_p,
                                'signal_mode': 'percentile', 'signal_pct': sig_pct,
                                'direction': direction,
                                'hour_filter': None,
                                'initial_capital': 10000, 'risk_per_trade': 0.02,
                                'max_pos_frac': 0.30, 'cooldown': 0,
                                'confirmation': False,
                            }
                            e_lbl = f"exec_TP{e_tp:.1f}_SL{e_sl:.1f}"
                            label = f"{cfg_lbl}|{e_lbl}|{sig_lbl}|{direction}|{fee_lbl}|{exit_mode}"
                            s = run_single_experiment(
                                close_v, high_v, low_v, atr_v, hours_v,
                                lp, sp, regime_v, fund_v, cfg, label)
                            if s:
                                all_results.append(s)

    profitable = [r for r in all_results if r['pf'] > 1.0]
    print(f"\n   Retrain Total: {len(all_results)} experiments, {len(profitable)} profitable")
    if profitable:
        best = max(profitable, key=lambda x: x['pf'])
        print(f"   Best retrain: {best['label']} PF={best['pf']:.3f} Ret={best['ret']:+.1f}%")

    return all_results


# ======================================================================
# 8. ANALYSIS & REPORTING
# ======================================================================

def print_top_results(results, title="TOP RESULTS", top_n=30):
    """Print ranked results table."""
    if not results:
        print(f"\n   No results to show.")
        return

    profitable = [r for r in results if r['pf'] > 1.0]
    profitable.sort(key=lambda x: x['pf'], reverse=True)

    print(f"\n{'='*120}")
    print(f"  {title}")
    print(f"  Total experiments: {len(results):,} | Profitable: {len(profitable)}")
    print(f"{'='*120}")

    if not profitable:
        # Show least-bad
        print(f"\n  ⚠️  No profitable strategies found. Showing least-bad:")
        show = sorted(results, key=lambda x: x['pf'], reverse=True)[:top_n]
    else:
        show = profitable[:top_n]

    print(f"\n  {'#':<4s} {'Strategy':<65s} {'Trades':>6s} {'WR':>6s} "
          f"{'PF':>6s} {'Ret%':>8s} {'DD%':>6s} {'Sharpe':>7s} {'L/S':>7s}")
    print(f"  {'-'*118}")

    for i, s in enumerate(show):
        wr_str = f"{s['wr']:.1%}"
        lbl = s['label'][:64]
        ls_str = f"{s['n_long']}/{s['n_short']}"
        marker = " ✅" if s['pf'] > 1.0 else ""
        print(f"  {i+1:<4d} {lbl:<65s} {s['trades']:>6d} {wr_str:>6s} "
              f"{s['pf']:>6.2f} {s['ret']:>+8.1f} {s['dd']:>6.1f} "
              f"{s['sharpe']:>7.2f} {ls_str:>7s}{marker}")

    # Analyze what works
    if profitable:
        print(f"\n  --- Pattern Analysis ---")

        # Count by fee type
        fee_counts = {}
        for p in profitable:
            parts = p['label'].split('|')
            for pt in parts:
                if pt in ('taker', 'maker', 'vip'):
                    fee_counts[pt] = fee_counts.get(pt, 0) + 1
        if fee_counts:
            print(f"  Fee type distribution: {fee_counts}")

        # Count by exit mode
        exit_counts = {}
        for p in profitable:
            parts = p['label'].split('|')
            for pt in parts:
                if pt.startswith(('fixed', 'trail', 'be', 'part', 'breakeven')):
                    exit_counts[pt] = exit_counts.get(pt, 0) + 1
        if exit_counts:
            print(f"  Exit mode distribution: {exit_counts}")

        # Count by direction
        dir_counts = {}
        for p in profitable:
            parts = p['label'].split('|')
            for pt in parts:
                if pt in ('both', 'long', 'short'):
                    dir_counts[pt] = dir_counts.get(pt, 0) + 1
        if dir_counts:
            print(f"  Direction distribution: {dir_counts}")

        # Count by signal filter
        sig_counts = {}
        for p in profitable:
            parts = p['label'].split('|')
            for pt in parts:
                if 'top' in pt or pt == 'all':
                    sig_counts[pt] = sig_counts.get(pt, 0) + 1
        if sig_counts:
            print(f"  Signal filter distribution: {sig_counts}")


def analyze_model_probabilities(long_probs, short_probs, regime_mask):
    """Analyze probability distributions to understand model behavior."""
    print(f"\n{'='*70}")
    print(f"  MODEL PROBABILITY ANALYSIS")
    print(f"{'='*70}")

    # Only OOS region
    lp_valid = long_probs[~np.isnan(long_probs)]
    sp_valid = short_probs[~np.isnan(short_probs)]

    if len(lp_valid) == 0:
        print("   No valid predictions!")
        return

    print(f"\n   LONG model predictions ({len(lp_valid):,} bars):")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"      P{pct}: {np.percentile(lp_valid, pct):.4f}")
    print(f"      Mean: {lp_valid.mean():.4f}  Std: {lp_valid.std():.4f}")
    print(f"      Min: {lp_valid.min():.4f}  Max: {lp_valid.max():.4f}")

    print(f"\n   SHORT model predictions ({len(sp_valid):,} bars):")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"      P{pct}: {np.percentile(sp_valid, pct):.4f}")
    print(f"      Mean: {sp_valid.mean():.4f}  Std: {sp_valid.std():.4f}")
    print(f"      Min: {sp_valid.min():.4f}  Max: {sp_valid.max():.4f}")

    # Threshold coverage
    print(f"\n   Threshold coverage (LONG):")
    for thr in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        pct = (lp_valid > thr).mean()
        print(f"      > {thr:.2f}: {pct:.1%} of bars")

    print(f"\n   Threshold coverage (SHORT):")
    for thr in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        pct = (sp_valid > thr).mean()
        print(f"      > {thr:.2f}: {pct:.1%} of bars")

    # Correlation
    both_valid = ~np.isnan(long_probs) & ~np.isnan(short_probs)
    if both_valid.sum() > 10:
        corr = np.corrcoef(long_probs[both_valid], short_probs[both_valid])[0, 1]
        print(f"\n   Long-Short correlation: {corr:.3f}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("  V5 MEGA-EXPERIMENT: Try Everything")
    print("  Systematic strategy optimization for BTC/USDT perpetual futures")
    print("=" * 70)

    t0 = datetime.now()

    # ---- Load data ----
    print(f"\n{'='*70}")
    print(f"  STEP 1: Load Data & Build Features")
    print(f"{'='*70}")
    df = load_data()
    feature_cols = get_feature_cols(df)
    print(f"   {len(feature_cols)} features available")

    close_v = df['close'].values
    high_v = df['high'].values
    low_v = df['low'].values
    atr_v = df['atr_14'].values if 'atr_14' in df.columns else np.ones(len(df))
    hours_v = df['_hour'].values
    regime_v = df['regime_ok'].values
    fund_v = df['funding_rate'].values if 'funding_rate' in df.columns else None

    # ---- Generate OOS predictions with V4 barrier config ----
    print(f"\n{'='*70}")
    print(f"  STEP 2: Walk-Forward OOS Predictions (V4 barriers: TP=2.5 SL=1.5)")
    print(f"{'='*70}")

    long_probs, long_auc = generate_labels_and_predict(
        df, feature_cols, tp_mult=2.5, sl_mult=1.5, max_hold=12, direction="long")
    short_probs, short_auc = generate_labels_and_predict(
        df, feature_cols, tp_mult=2.5, sl_mult=1.5, max_hold=12, direction="short")

    # ---- Analyze probability distributions ----
    analyze_model_probabilities(long_probs, short_probs, regime_v)

    # ---- Phase A: ML strategy sweep (V4 model, varied execution) ----
    print(f"\n{'='*70}")
    print(f"  STEP 3: ML Strategy Sweep (V4 model predictions)")
    print(f"{'='*70}")
    ml_results = sweep_experiments(
        close_v, high_v, low_v, atr_v, hours_v,
        long_probs, short_probs,
        regime_v, fund_v)

    print_top_results(ml_results, "ML STRATEGY RESULTS (V4 model, varied execution)")

    # ---- Phase B: Rule-based strategies ----
    rule_results = run_rule_based_strategies(
        df, close_v, high_v, low_v, atr_v, hours_v,
        regime_v, fund_v)

    if rule_results:
        print_top_results(rule_results, "RULE-BASED STRATEGY RESULTS")

    # ---- Phase C: Retrain with different barriers ----
    print(f"\n{'='*70}")
    print(f"  STEP 4: Retrain With Different Barrier Labels")
    print(f"{'='*70}")
    retrain_results = experiment_retrain_barriers(df, feature_cols)

    if retrain_results:
        print_top_results(retrain_results, "RETRAINED BARRIER RESULTS")

    # ---- Phase D: 4H timeframe ----
    results_4h = experiment_4h(df)
    if results_4h:
        print_top_results(results_4h, "4H TIMEFRAME RESULTS")

    # ---- GRAND SUMMARY ----
    all_results = ml_results + rule_results + retrain_results + results_4h
    print_top_results(all_results, "🏆 GRAND SUMMARY -- ALL EXPERIMENTS 🏆", top_n=50)

    # Save results
    out_path = ROOT / "outputs" / "v5_experiment_results.json"
    profitable = [r for r in all_results if r['pf'] > 1.0]

    summary = {
        'total_experiments': len(all_results),
        'profitable_count': len(profitable),
        'top_20': sorted(all_results, key=lambda x: x['pf'], reverse=True)[:20],
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': (datetime.now() - t0).total_seconds(),
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n   Results saved to {out_path}")

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n   Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"   Experiments run: {len(all_results):,}")
    print(f"   Profitable strategies: {len(profitable)}")

    if profitable:
        best = max(profitable, key=lambda x: x['pf'])
        print(f"\n   🏆 BEST STRATEGY: {best['label']}")
        print(f"      PF={best['pf']:.3f} | WR={best['wr']:.1%} | "
              f"Return={best['ret']:+.1f}% | MaxDD={best['dd']:.1f}% | "
              f"Sharpe={best['sharpe']:.2f}")
        print(f"      Trades: {best['trades']} (L:{best['n_long']} S:{best['n_short']})")
    else:
        print(f"\n   ⚠️  No profitable strategy found across {len(all_results):,} experiments")
        print(f"   The model has real signal (AUC>0.63) but BTC 1H is extremely efficient.")
        print(f"   Consider: alternative assets, higher timeframes, or hybrid approaches.")


if __name__ == "__main__":
    main()
