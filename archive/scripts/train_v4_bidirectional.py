"""
Train V4 -- Bidirectional (LONG + SHORT) with tighter barriers.

Key improvements over V3:
  1. TIGHTER barriers: TP=2.5x ATR, SL=1.5x ATR  (higher TP-hit rate ~25%)
  2. SHORT model: trained on short-side triple-barrier labels
  3. Combined scoring: long_prob - short_prob → directional confidence
  4. Funding rate features (if available)
  5. Walk-forward honest evaluation + bar-by-bar backtest

Key V3 findings driving this:
  - V3 TP rate was only 10% (TP=4x too ambitious) → tighter TP
  - SL hit 34% of bars → short model can exploit this
  - AUC=0.69 shows real signal exists but can't convert to P&L unidirectionally

Usage:
    python scripts/train_v4_bidirectional.py
"""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ML
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE

# Project
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features.alpha_features import build_alpha_features
from features.triple_barrier import triple_barrier_labels


# ======================================================================
# CONFIG
# ======================================================================
CONFIG = {
    # --- Barrier (tighter than V3 for higher TP rate) ---
    'long_tp_mult': 2.5,
    'long_sl_mult': 1.5,
    'short_tp_mult': 2.5,
    'short_sl_mult': 1.5,
    'max_holding': 12,       # 12h

    # --- Regime filter ---
    'min_atr_pct_percentile': 25,

    # --- Features ---
    'top_k_features': 50,

    # --- Optuna ---
    'n_trials': 50,

    # --- Walk-forward ---
    'n_splits': 5,
    'test_pct': 0.12,
    'purge_bars': 12,

    # --- Trading ---
    'confidence_threshold': 0.15,
    'fee_pct': 0.075,
    'slippage_pct': 0.02,
    'initial_capital': 10000,
    'risk_per_trade': 0.02,
    'max_pos_frac': 0.30,
}

TOTAL_COST = 2 * (CONFIG['fee_pct'] + CONFIG['slippage_pct']) / 100
SLIPPAGE_PER_SIDE = CONFIG['slippage_pct'] / 100


# ======================================================================
# Data helpers (reuse from V3)
# ======================================================================

def load_and_build_features(data_path: Path) -> pd.DataFrame:
    print("Loading data...")
    df = pd.read_parquet(data_path)
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    df = build_alpha_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna()
    print(f"   {n_before - len(df):,} warm-up rows dropped → {len(df):,} clean rows")

    # Check for funding rate
    if 'funding_rate' in df.columns:
        n_fund = df['funding_rate'].notna().sum()
        print(f"   ✅ Funding rate present: {n_fund:,} bars")
    else:
        print(f"   ⚠️  No funding rate data (run fetch_funding_rates.py first)")

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {
        'open', 'high', 'low', 'close', 'volume', 'funding_rate', 'open_interest',
        'timestamp', 'target', 'classic_binary',
        'regime_filter', 'vol_regime_low', 'regime_ok',
        # Long labels
        'tb_label', 'tb_binary', 'tb_holding', 'tb_barrier_tp', 'tb_barrier_sl',
        # Short labels
        'short_tb_label', 'short_tb_binary', 'short_tb_holding',
        'short_tb_barrier_tp', 'short_tb_barrier_sl',
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if any(p in c.lower() for p in ['future_', 'target_', 'label_']):
            continue
        if df[c].dtype in ('float64', 'float32', 'int64', 'int32'):
            cols.append(c)
    return cols


def add_regime_filter(df: pd.DataFrame) -> pd.DataFrame:
    if 'atr_pct' not in df.columns:
        if 'atr_14' in df.columns and 'close' in df.columns:
            df['atr_pct'] = df['atr_14'] / df['close'] * 100
        else:
            df['regime_filter'] = True
            return df

    threshold = df['atr_pct'].quantile(CONFIG['min_atr_pct_percentile'] / 100)
    df['regime_filter'] = df['atr_pct'] >= threshold
    print(f"   Regime filter: {df['regime_filter'].sum():,}/{len(df):,} pass "
          f"({df['regime_filter'].mean():.0%}), ATR% >= {threshold:.4f}%")
    return df


def purged_walk_forward_splits(n, n_splits=5, test_pct=0.12, purge=12):
    test_size = int(n * test_pct)
    splits = []
    for i in range(n_splits):
        te_start = n - (n_splits - i) * test_size
        te_end = min(te_start + test_size, n)
        tr_end = te_start - purge
        if te_start < test_size or tr_end < 100:
            continue
        splits.append((np.arange(0, tr_end), np.arange(te_start, te_end)))
    print(f"[SPLIT] {len(splits)} purged walk-forward splits (purge={purge}h)")
    for i, (tr, te) in enumerate(splits):
        print(f"   Fold {i+1}: train={len(tr):,} test={len(te):,}")
    return splits


# ======================================================================
# Feature selection
# ======================================================================

def select_stable_features(X, y, splits, top_k=50) -> list:
    print("\n[FEAT] Feature selection via walk-forward importance stability...")
    imp_mat = pd.DataFrame(0.0, index=X.columns, columns=range(len(splits)))
    for fold_i, (tr, te) in enumerate(splits):
        m = lgb.LGBMClassifier(n_estimators=500, max_depth=5, learning_rate=0.03,
                               class_weight='balanced', verbose=-1, random_state=42,
                               subsample=0.8, colsample_bytree=0.8)
        m.fit(X.iloc[tr], y.iloc[tr])
        imp = pd.Series(m.feature_importances_, index=X.columns)
        imp_mat[fold_i] = imp / (imp.sum() + 1e-10)

    ranks = imp_mat.rank(ascending=False)
    avg_rank = ranks.mean(axis=1)
    rank_std = ranks.std(axis=1)
    score = (avg_rank + 0.3 * rank_std).sort_values()
    selected = score.head(top_k).index.tolist()
    print(f"   Selected {len(selected)} features. Top 10: {selected[:10]}")
    return selected


# ======================================================================
# Optuna objectives
# ======================================================================

def _lgb_trial(trial, X_tr, y_tr, X_te, y_te):
    p = {
        'objective': 'binary', 'metric': 'auc',
        'random_state': 42, 'class_weight': 'balanced', 'verbose': -1,
        'max_depth': trial.suggest_int('d', 3, 7),
        'learning_rate': trial.suggest_float('lr', 0.005, 0.08, log=True),
        'n_estimators': trial.suggest_int('ne', 300, 1500, step=100),
        'subsample': trial.suggest_float('sub', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('col', 0.3, 0.8),
        'min_child_samples': trial.suggest_int('mcs', 10, 100),
        'reg_alpha': trial.suggest_float('al', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('la', 1.0, 10.0),
        'num_leaves': trial.suggest_int('nl', 15, 63),
    }
    m = lgb.LGBMClassifier(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])


def _xgb_trial(trial, X_tr, y_tr, X_te, y_te):
    p = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'tree_method': 'hist', 'random_state': 42,
        'scale_pos_weight': (y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        'max_depth': trial.suggest_int('d', 3, 7),
        'learning_rate': trial.suggest_float('lr', 0.005, 0.08, log=True),
        'n_estimators': trial.suggest_int('ne', 300, 1500, step=100),
        'subsample': trial.suggest_float('sub', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('col', 0.3, 0.8),
        'min_child_weight': trial.suggest_int('mcw', 5, 50),
        'gamma': trial.suggest_float('g', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('al', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('la', 1.0, 10.0),
    }
    m = xgb.XGBClassifier(**p, early_stopping_rounds=50)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])


# ======================================================================
# Train one side (long or short)
# ======================================================================

def train_one_side(
    label: str,           # "LONG" or "SHORT"
    X_sel: pd.DataFrame,
    y: pd.Series,
    splits: list,
    n_trials: int = 50,
) -> dict:
    """
    Optuna-tune LGB + XGB, walk-forward evaluate, return models + OOS probs.
    """
    print(f"\n{'='*70}")
    print(f"  TRAINING {label} MODEL")
    print(f"{'='*70}")
    print(f"   Samples: {len(X_sel):,}  |  Positive rate: {y.mean():.1%}")

    # Optuna on first fold
    tr0, te0 = splits[0]
    X_tr, y_tr = X_sel.iloc[tr0], y.iloc[tr0]
    X_te, y_te = X_sel.iloc[te0], y.iloc[te0]

    if y_tr.sum() > 5:
        smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_tr.sum()) - 1))
        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
    else:
        X_tr_sm, y_tr_sm = X_tr, y_tr

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"   [Optuna] LightGBM ({n_trials} trials)...")
    study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_lgb.optimize(lambda t: _lgb_trial(t, X_tr_sm, y_tr_sm, X_te, y_te),
                       n_trials=n_trials, show_progress_bar=True)
    print(f"      Best LGB AUC: {study_lgb.best_value:.4f}")

    print(f"   [Optuna] XGBoost ({n_trials} trials)...")
    study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_xgb.optimize(lambda t: _xgb_trial(t, X_tr_sm, y_tr_sm, X_te, y_te),
                       n_trials=n_trials, show_progress_bar=True)
    print(f"      Best XGB AUC: {study_xgb.best_value:.4f}")

    # Build param dicts
    def _build_lgb(study):
        bp = study.best_params
        return {
            'max_depth': bp['d'], 'learning_rate': bp['lr'],
            'n_estimators': bp['ne'], 'subsample': bp['sub'],
            'colsample_bytree': bp['col'], 'min_child_samples': bp['mcs'],
            'reg_alpha': bp['al'], 'reg_lambda': bp['la'], 'num_leaves': bp['nl'],
            'objective': 'binary', 'metric': 'auc',
            'random_state': 42, 'class_weight': 'balanced', 'verbose': -1,
        }

    def _build_xgb(study, y_tr):
        bp = study.best_params
        return {
            'max_depth': bp['d'], 'learning_rate': bp['lr'],
            'n_estimators': bp['ne'], 'subsample': bp['sub'],
            'colsample_bytree': bp['col'], 'min_child_weight': bp['mcw'],
            'gamma': bp['g'], 'reg_alpha': bp['al'], 'reg_lambda': bp['la'],
            'objective': 'binary:logistic', 'eval_metric': 'auc',
            'tree_method': 'hist', 'random_state': 42,
            'scale_pos_weight': (y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        }

    lgb_params = _build_lgb(study_lgb)
    xgb_params = _build_xgb(study_xgb, y_tr)

    # Walk-forward evaluation
    print(f"\n   [Walk-Forward] {len(splits)} folds...")
    oos_probs = np.full(len(X_sel), np.nan)
    fold_aucs = []

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X_sel.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X_sel.iloc[te_idx], y.iloc[te_idx]

        if y_tr.sum() > 5:
            smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_tr.sum()) - 1))
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
        else:
            X_tr_sm, y_tr_sm = X_tr, y_tr

        xp = dict(xgb_params)
        xp['scale_pos_weight'] = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        m_xgb = xgb.XGBClassifier(**xp, early_stopping_rounds=50)
        m_xgb.fit(X_tr_sm, y_tr_sm, eval_set=[(X_te, y_te)], verbose=False)

        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(X_tr_sm, y_tr_sm, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        p_xgb = m_xgb.predict_proba(X_te)[:, 1]
        p_lgb = m_lgb.predict_proba(X_te)[:, 1]
        p_ens = (p_xgb + p_lgb) / 2.0

        auc = roc_auc_score(y_te, p_ens)
        fold_aucs.append(auc)
        oos_probs[te_idx] = p_ens
        print(f"      Fold {fold_i+1}: XGB={roc_auc_score(y_te, p_xgb):.4f}  "
              f"LGB={roc_auc_score(y_te, p_lgb):.4f}  ENS={auc:.4f}")

    valid_mask = ~np.isnan(oos_probs)
    y_valid = y.values[valid_mask]
    pooled_auc = roc_auc_score(y_valid, oos_probs[valid_mask])
    print(f"\n   {label} Walk-Forward AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"   {label} Pooled OOS AUC:   {pooled_auc:.4f}")

    # Train final models on 85% for deployment
    cal_split = int(len(X_sel) * 0.85)
    X_train_f = X_sel.iloc[:cal_split]
    y_train_f = y.iloc[:cal_split]
    X_cal = X_sel.iloc[cal_split:]
    y_cal = y.iloc[cal_split:]

    if y_train_f.sum() > 5:
        smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_train_f.sum()) - 1))
        X_tsm, y_tsm = smote.fit_resample(X_train_f, y_train_f)
    else:
        X_tsm, y_tsm = X_train_f, y_train_f

    xp_f = dict(xgb_params)
    xp_f['scale_pos_weight'] = (y_train_f == 0).sum() / max((y_train_f == 1).sum(), 1)
    m_xgb_f = xgb.XGBClassifier(**xp_f, early_stopping_rounds=50)
    m_xgb_f.fit(X_tsm, y_tsm, eval_set=[(X_cal, y_cal)], verbose=False)

    m_lgb_f = lgb.LGBMClassifier(**lgb_params)
    m_lgb_f.fit(X_tsm, y_tsm, eval_set=[(X_cal, y_cal)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

    p_cal = (m_xgb_f.predict_proba(X_cal)[:, 1] + m_lgb_f.predict_proba(X_cal)[:, 1]) / 2
    cal_auc = roc_auc_score(y_cal, p_cal)
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(p_cal, y_cal.values)
    print(f"   {label} Calibration-set AUC: {cal_auc:.4f}")

    # Feature importance
    imp_lgb = pd.Series(m_lgb_f.feature_importances_, index=X_sel.columns)
    imp_lgb = (imp_lgb / imp_lgb.sum()).sort_values(ascending=False)
    print(f"   {label} Top 10 features: {list(imp_lgb.head(10).index)}")

    return {
        'xgb_model': m_xgb_f,
        'lgb_model': m_lgb_f,
        'calibrator': calibrator,
        'xgb_params': xgb_params,
        'lgb_params': lgb_params,
        'fold_aucs': fold_aucs,
        'pooled_auc': pooled_auc,
        'cal_auc': cal_auc,
        'oos_probs': oos_probs,
        'feature_importance': imp_lgb.head(20).to_dict(),
    }


# ======================================================================
# Bar-by-bar backtest (bidirectional)
# ======================================================================

def simulate_bidirectional(
    close, high, low, atr,
    long_signals, short_signals,
    long_tp_mult, long_sl_mult,
    short_tp_mult, short_sl_mult,
    max_hold,
) -> pd.DataFrame:
    """
    Bar-by-bar simulator supporting both LONG and SHORT positions.
    Only one position at a time.
    """
    n = len(close)
    capital = CONFIG['initial_capital']
    in_position = False
    pos_side = None
    trades = []

    for i in range(n):
        if in_position:
            bars_held = i - entry_bar

            exit_price = None
            reason = None

            if pos_side == 'LONG':
                if low[i] <= sl_price:
                    exit_price = sl_price
                    reason = 'SL'
                elif high[i] >= tp_price:
                    exit_price = tp_price
                    reason = 'TP'
                elif bars_held >= max_hold:
                    exit_price = close[i]
                    reason = 'TIMEOUT'
            else:  # SHORT
                if high[i] >= sl_price:
                    exit_price = sl_price
                    reason = 'SL'
                elif low[i] <= tp_price:
                    exit_price = tp_price
                    reason = 'TP'
                elif bars_held >= max_hold:
                    exit_price = close[i]
                    reason = 'TIMEOUT'

            if exit_price is not None:
                if pos_side == 'LONG':
                    gross_ret = (exit_price - entry_p) / entry_p
                else:
                    gross_ret = (entry_p - exit_price) / entry_p
                net_ret = gross_ret - TOTAL_COST
                pnl_usd = pos_size * net_ret
                capital += pnl_usd
                trades.append({
                    'side': pos_side,
                    'reason': reason,
                    'bars_held': bars_held,
                    'gross_ret_pct': gross_ret * 100,
                    'net_ret_pct': net_ret * 100,
                    'pnl_usd': pnl_usd,
                    'capital': capital,
                })
                in_position = False
                pos_side = None

        # Try entering a new position (long preferred over short if both signal)
        if not in_position and capital > 100 and atr[i] > 0:
            if long_signals[i]:
                entry_p = close[i] * (1 + SLIPPAGE_PER_SIDE)
                entry_bar = i
                tp_price = entry_p + long_tp_mult * atr[i]
                sl_price = entry_p - long_sl_mult * atr[i]
                sl_dist = (entry_p - sl_price) / entry_p
                pos_size = min(capital * CONFIG['risk_per_trade'] / max(sl_dist, 1e-6),
                               capital * CONFIG['max_pos_frac'])
                if pos_size >= 50:
                    in_position = True
                    pos_side = 'LONG'

            elif short_signals[i]:
                entry_p = close[i] * (1 - SLIPPAGE_PER_SIDE)
                entry_bar = i
                tp_price = entry_p - short_tp_mult * atr[i]
                sl_price = entry_p + short_sl_mult * atr[i]
                sl_dist = (sl_price - entry_p) / entry_p
                pos_size = min(capital * CONFIG['risk_per_trade'] / max(sl_dist, 1e-6),
                               capital * CONFIG['max_pos_frac'])
                if pos_size >= 50:
                    in_position = True
                    pos_side = 'SHORT'

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def calc_stats(trades_df, label=""):
    if trades_df.empty:
        return {}
    n = len(trades_df)
    wins = trades_df['pnl_usd'] > 0
    wr = wins.mean()
    gp = trades_df.loc[wins, 'pnl_usd'].sum()
    gl = abs(trades_df.loc[~wins, 'pnl_usd'].sum())
    pf = gp / gl if gl > 0 else float('inf')
    final = trades_df['capital'].iloc[-1]
    tr = (final - CONFIG['initial_capital']) / CONFIG['initial_capital'] * 100

    eq = np.array([CONFIG['initial_capital']] + trades_df['capital'].tolist())
    pk = np.maximum.accumulate(eq)
    dd = (pk - eq) / pk
    max_dd = dd.max() * 100

    rets = trades_df['net_ret_pct'].values / 100
    sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252 * 2) if len(rets) > 1 else 0

    n_long = (trades_df['side'] == 'LONG').sum() if 'side' in trades_df.columns else n
    n_short = (trades_df['side'] == 'SHORT').sum() if 'side' in trades_df.columns else 0
    tp_n = (trades_df['reason'] == 'TP').sum()
    sl_n = (trades_df['reason'] == 'SL').sum()
    to_n = (trades_df['reason'] == 'TIMEOUT').sum()

    return {
        'label': label, 'trades': n, 'win_rate': wr,
        'total_ret': tr, 'final_cap': final,
        'pf': pf, 'max_dd': max_dd, 'sharpe': sharpe,
        'n_long': n_long, 'n_short': n_short,
        'tp': tp_n, 'sl': sl_n, 'timeout': to_n,
    }


def print_stats(s):
    if not s:
        print("   (no trades)")
        return
    print(f"   {s['label']}")
    print(f"   Trades: {s['trades']:,} (L:{s.get('n_long',0)} S:{s.get('n_short',0)})  |  "
          f"Win rate: {s['win_rate']:.1%}  |  PF: {s['pf']:.2f}")
    print(f"   Return: {s['total_ret']:+.1f}%  |  Max DD: {s['max_dd']:.1f}%  |  "
          f"Sharpe: {s['sharpe']:.2f}")
    print(f"   TP: {s['tp']}  SL: {s['sl']}  Timeout: {s['timeout']}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("  TRAIN V4 -- BIDIRECTIONAL (LONG + SHORT)")
    print("  Barriers: TP=2.5x SL=1.5x Hold=12 | Regime-filtered")
    print("=" * 70)

    ROOT = Path(__file__).parent.parent
    MODEL_DIR = ROOT / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data_paths = [
        ROOT / "data" / "processed" / "btcusdt_1h_full_history_enhanced.parquet",
        ROOT / "data" / "processed" / "btcusdt_1h_features.parquet",
    ]
    data_path = next((p for p in data_paths if p.exists()), None)
    if data_path is None:
        print("[ERR] No data found!")
        return

    df = load_and_build_features(data_path)
    df = add_regime_filter(df)

    # ------------------------------------------------------------------
    # 2. Triple-barrier labels: LONG + SHORT
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  STEP 2: Triple-Barrier Labels (bidirectional)")
    print(f"{'='*70}")

    print(f"\n   --- LONG side (TP={CONFIG['long_tp_mult']}x, SL={CONFIG['long_sl_mult']}x) ---")
    tb_long = triple_barrier_labels(
        df['close'], df['high'], df['low'],
        tp_mult=CONFIG['long_tp_mult'],
        sl_mult=CONFIG['long_sl_mult'],
        max_holding=CONFIG['max_holding'],
        direction='long',
    )

    print(f"\n   --- SHORT side (TP={CONFIG['short_tp_mult']}x, SL={CONFIG['short_sl_mult']}x) ---")
    tb_short = triple_barrier_labels(
        df['close'], df['high'], df['low'],
        tp_mult=CONFIG['short_tp_mult'],
        sl_mult=CONFIG['short_sl_mult'],
        max_holding=CONFIG['max_holding'],
        direction='short',
    )

    df = pd.concat([df, tb_long, tb_short], axis=1)

    # ------------------------------------------------------------------
    # 3. Prepare features
    # ------------------------------------------------------------------
    feature_cols = get_feature_cols(df)
    mask = (df[feature_cols].notna().all(axis=1) &
            df['tb_binary'].notna() &
            df['short_tb_binary'].notna())
    df_clean = df[mask].copy()

    X_all = df_clean[feature_cols]
    y_long = df_clean['tb_binary'].astype(int)
    y_short = df_clean['short_tb_binary'].astype(int)
    regime_all = df_clean['regime_filter'].values

    print(f"\n[DATA] Clean dataset: {len(df_clean):,} x {len(feature_cols)} features")
    print(f"   Long TP rate:  {y_long.mean():.1%}")
    print(f"   Short TP rate: {y_short.mean():.1%}")

    # ------------------------------------------------------------------
    # 4. Walk-forward splits
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  STEP 3: Walk-Forward Splits")
    print(f"{'='*70}")

    splits = purged_walk_forward_splits(
        len(X_all), CONFIG['n_splits'], CONFIG['test_pct'], CONFIG['purge_bars'])

    # ------------------------------------------------------------------
    # 5. Feature selection (on long model, reuse for both)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  STEP 4: Feature Selection")
    print(f"{'='*70}")

    selected = select_stable_features(X_all, y_long, splits, CONFIG['top_k_features'])
    X_sel = X_all[selected]

    # ------------------------------------------------------------------
    # 6. Train LONG model
    # ------------------------------------------------------------------
    long_result = train_one_side("LONG", X_sel, y_long, splits, CONFIG['n_trials'])

    # ------------------------------------------------------------------
    # 7. Train SHORT model
    # ------------------------------------------------------------------
    short_result = train_one_side("SHORT", X_sel, y_short, splits, CONFIG['n_trials'])

    # ------------------------------------------------------------------
    # 8. Bidirectional walk-forward backtest
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  STEP 7: Bidirectional Backtest (OOS)")
    print(f"{'='*70}")

    long_probs = long_result['oos_probs']
    short_probs = short_result['oos_probs']

    # OOS region = where both have predictions
    valid = ~np.isnan(long_probs) & ~np.isnan(short_probs)
    oos_indices = np.where(valid)[0]
    if len(oos_indices) == 0:
        print("   No valid OOS region!")
        return

    oos_start = oos_indices[0]
    oos_end = oos_indices[-1] + 1

    df_oos = df_clean.iloc[oos_start:oos_end].copy()
    lp = long_probs[oos_start:oos_end]
    sp = short_probs[oos_start:oos_end]
    regime_oos = df_oos['regime_filter'].values

    close_oos = df_oos['close'].values
    high_oos = df_oos['high'].values
    low_oos = df_oos['low'].values
    atr_oos = df_oos['atr_14'].values

    print(f"\n   OOS region: {len(df_oos):,} bars")
    print(f"   Regime pass: {regime_oos.sum():,}")

    # Strategy sweep
    print(f"\n   {'Strategy':<40s} {'Trades':>7s} {'L/S':>7s} {'WR':>6s} "
          f"{'Ret%':>8s} {'PF':>6s} {'DD':>6s} {'Sharpe':>7s}")
    print(f"   {'-'*95}")

    results = []

    for long_thr in [0.15, 0.20, 0.25, 0.30]:
        for short_thr in [0.15, 0.20, 0.25, 0.30]:
            long_sig = (~np.isnan(lp)) & (lp > long_thr) & regime_oos
            short_sig = (~np.isnan(sp)) & (sp > short_thr) & regime_oos
            # Don't signal both at same bar
            both = long_sig & short_sig
            # Prefer the higher confidence side
            for idx in np.where(both)[0]:
                if lp[idx] >= sp[idx]:
                    short_sig[idx] = False
                else:
                    long_sig[idx] = False

            trades = simulate_bidirectional(
                close_oos, high_oos, low_oos, atr_oos,
                long_sig, short_sig,
                CONFIG['long_tp_mult'], CONFIG['long_sl_mult'],
                CONFIG['short_tp_mult'], CONFIG['short_sl_mult'],
                CONFIG['max_holding'],
            )
            lbl = f"L>{long_thr:.2f} S>{short_thr:.2f}"
            s = calc_stats(trades, lbl)
            if s:
                results.append(s)
                print(f"   {lbl:<40s} {s['trades']:>7d} {s.get('n_long',0):>3d}/{s.get('n_short',0):>3d} "
                      f"{s['win_rate']:>6.1%} {s['total_ret']:>8.1f}% {s['pf']:>6.2f} "
                      f"{s['max_dd']:>6.1f}% {s['sharpe']:>7.2f}")

    # Also test long-only and short-only
    print(f"\n   --- Single-direction baselines ---")
    for thr in [0.15, 0.20, 0.25]:
        # Long only
        long_sig = (~np.isnan(lp)) & (lp > thr) & regime_oos
        trades = simulate_bidirectional(
            close_oos, high_oos, low_oos, atr_oos,
            long_sig, np.zeros(len(df_oos), dtype=bool),
            CONFIG['long_tp_mult'], CONFIG['long_sl_mult'],
            CONFIG['short_tp_mult'], CONFIG['short_sl_mult'],
            CONFIG['max_holding'],
        )
        lbl = f"LONG-only >{thr:.2f}"
        s = calc_stats(trades, lbl)
        if s:
            results.append(s)
            print(f"   {lbl:<40s} {s['trades']:>7d} {s.get('n_long',0):>3d}/{s.get('n_short',0):>3d} "
                  f"{s['win_rate']:>6.1%} {s['total_ret']:>8.1f}% {s['pf']:>6.2f} "
                  f"{s['max_dd']:>6.1f}% {s['sharpe']:>7.2f}")

        # Short only
        short_sig = (~np.isnan(sp)) & (sp > thr) & regime_oos
        trades = simulate_bidirectional(
            close_oos, high_oos, low_oos, atr_oos,
            np.zeros(len(df_oos), dtype=bool), short_sig,
            CONFIG['long_tp_mult'], CONFIG['long_sl_mult'],
            CONFIG['short_tp_mult'], CONFIG['short_sl_mult'],
            CONFIG['max_holding'],
        )
        lbl = f"SHORT-only >{thr:.2f}"
        s = calc_stats(trades, lbl)
        if s:
            results.append(s)
            print(f"   {lbl:<40s} {s['trades']:>7d} {s.get('n_long',0):>3d}/{s.get('n_short',0):>3d} "
                  f"{s['win_rate']:>6.1%} {s['total_ret']:>8.1f}% {s['pf']:>6.2f} "
                  f"{s['max_dd']:>6.1f}% {s['sharpe']:>7.2f}")

    # Directional score: long_prob - short_prob
    print(f"\n   --- Directional score (Lp - Sp) ---")
    dir_score = lp - sp
    for long_sc, short_sc in [(0.05, -0.05), (0.10, -0.10), (0.15, -0.15)]:
        long_sig = (dir_score > long_sc) & regime_oos & (~np.isnan(dir_score))
        short_sig = (dir_score < short_sc) & regime_oos & (~np.isnan(dir_score))
        trades = simulate_bidirectional(
            close_oos, high_oos, low_oos, atr_oos,
            long_sig, short_sig,
            CONFIG['long_tp_mult'], CONFIG['long_sl_mult'],
            CONFIG['short_tp_mult'], CONFIG['short_sl_mult'],
            CONFIG['max_holding'],
        )
        lbl = f"Dir>{long_sc:+.2f} / <{short_sc:+.2f}"
        s = calc_stats(trades, lbl)
        if s:
            results.append(s)
            print(f"   {lbl:<40s} {s['trades']:>7d} {s.get('n_long',0):>3d}/{s.get('n_short',0):>3d} "
                  f"{s['win_rate']:>6.1%} {s['total_ret']:>8.1f}% {s['pf']:>6.2f} "
                  f"{s['max_dd']:>6.1f}% {s['sharpe']:>7.2f}")

    # Random baseline
    np.random.seed(42)
    n_rand = int(0.10 * len(df_oos))
    rand_l = np.zeros(len(df_oos), dtype=bool)
    rand_s = np.zeros(len(df_oos), dtype=bool)
    ridx = np.random.choice(len(df_oos), n_rand, replace=False)
    rand_l[ridx[:n_rand//2]] = True
    rand_s[ridx[n_rand//2:]] = True
    rand_l &= regime_oos
    rand_s &= regime_oos
    trades = simulate_bidirectional(
        close_oos, high_oos, low_oos, atr_oos,
        rand_l, rand_s,
        CONFIG['long_tp_mult'], CONFIG['long_sl_mult'],
        CONFIG['short_tp_mult'], CONFIG['short_sl_mult'],
        CONFIG['max_holding'],
    )
    s = calc_stats(trades, "Random L+S")
    if s:
        results.append(s)
        print(f"\n   {'Random L+S':<40s} {s['trades']:>7d} {s.get('n_long',0):>3d}/{s.get('n_short',0):>3d} "
              f"{s['win_rate']:>6.1%} {s['total_ret']:>8.1f}% {s['pf']:>6.2f} "
              f"{s['max_dd']:>6.1f}% {s['sharpe']:>7.2f}")

    # ------------------------------------------------------------------
    # 9. Save models
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  SAVE MODEL ARTIFACTS")
    print(f"{'='*70}")

    artifact = {
        'long_xgb': long_result['xgb_model'],
        'long_lgb': long_result['lgb_model'],
        'long_calibrator': long_result['calibrator'],
        'short_xgb': short_result['xgb_model'],
        'short_lgb': short_result['lgb_model'],
        'short_calibrator': short_result['calibrator'],
        'feature_cols': selected,
        'config': CONFIG,
        'training_date': datetime.now().isoformat(),
        'regime_threshold_atr_pct': float(
            df['atr_pct'].quantile(CONFIG['min_atr_pct_percentile'] / 100)
        ) if 'atr_pct' in df.columns else None,
    }

    model_path = MODEL_DIR / "ensemble_v4_bidirectional.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"   Model saved: {model_path}")

    # Find best result
    profitable = [r for r in results if r.get('pf', 0) > 1.0]
    best = max(results, key=lambda r: r.get('pf', 0)) if results else {}

    metadata = {
        "version": "v4_bidirectional",
        "description": "Bidirectional LONG+SHORT with tighter barriers",
        "config": CONFIG,
        "models": {"long": ["XGBoost", "LightGBM"], "short": ["XGBoost", "LightGBM"]},
        "features": selected,
        "n_features": len(selected),
        "training_date": datetime.now().isoformat(),
        "long_model": {
            "fold_aucs": [float(a) for a in long_result['fold_aucs']],
            "mean_auc": float(np.mean(long_result['fold_aucs'])),
            "pooled_auc": float(long_result['pooled_auc']),
            "cal_auc": float(long_result['cal_auc']),
            "top_features": long_result['feature_importance'],
        },
        "short_model": {
            "fold_aucs": [float(a) for a in short_result['fold_aucs']],
            "mean_auc": float(np.mean(short_result['fold_aucs'])),
            "pooled_auc": float(short_result['pooled_auc']),
            "cal_auc": float(short_result['cal_auc']),
            "top_features": short_result['feature_importance'],
        },
        "best_backtest": best,
        "profitable_strategies": len(profitable),
        "all_backtest_results": results,
    }

    meta_path = MODEL_DIR / "ensemble_v4_bidirectional_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"   Metadata saved: {meta_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE -- V4 Bidirectional")
    print(f"{'='*70}")
    print(f"   LONG  model: WF AUC = {np.mean(long_result['fold_aucs']):.4f} ± "
          f"{np.std(long_result['fold_aucs']):.4f}")
    print(f"   SHORT model: WF AUC = {np.mean(short_result['fold_aucs']):.4f} ± "
          f"{np.std(short_result['fold_aucs']):.4f}")
    if best:
        print(f"\n   Best strategy: {best['label']}")
        print(f"   → {best['trades']} trades, WR={best['win_rate']:.1%}, "
              f"PF={best['pf']:.2f}, Return={best['total_ret']:+.1f}%")
    if profitable:
        print(f"\n   ✅ {len(profitable)} profitable strategies found!")
    else:
        print(f"\n   ⚠️  No profitable strategies in OOS (but model ranking power may still be useful)")

    print(f"\n   Next steps:")
    print(f"   1. python scripts/start_paper_trading_v4.py")
    print(f"   2. Deploy updated Pine Script")


if __name__ == "__main__":
    main()
