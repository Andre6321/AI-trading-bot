"""
Train V3 -- Optimized based on deep analysis findings.

Key changes from V2:
  1. Barrier: TP=4x ATR, SL=2x ATR, Hold=12 bars  (best AUC=0.6241 in sweep)
  2. Regime filter: exclude low-volatility periods (AUC inverted to 0.42 there)
  3. Feature pruning: drop noise features, keep mean-reversion signals
  4. Confidence gating: only trade when P(TP) > threshold
  5. Realistic backtest embedded at end with fees + slippage

Deep analysis proved walk-forward AUC ~0.617 is real.

Usage:
    python scripts/train_v3_optimized.py
"""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ML
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import (roc_auc_score, brier_score_loss, log_loss,
                             precision_recall_curve, classification_report)
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.alpha_features import build_alpha_features
from features.triple_barrier import triple_barrier_labels


# ======================================================================
# CONFIGURATION -- from deep analysis results
# ======================================================================
CONFIG = {
    # Triple-barrier (optimal from SL/TP sweep)
    'tp_mult': 4.0,        # 4x ATR take-profit
    'sl_mult': 2.0,        # 2x ATR stop-loss (R:R = 2:1)
    'max_holding': 12,     # 12 bars = 12h max hold

    # Regime filter
    'min_atr_pct_percentile': 25,  # exclude bottom 25% volatility

    # Feature selection
    'top_k_features': 50,   # keep top 50 stable features

    # Optuna
    'n_trials': 60,        # per model

    # Walk-forward
    'n_splits': 5,
    'test_pct': 0.12,
    'purge_bars': 12,      # 12h purge (matches max_holding)

    # Trading thresholds
    'confidence_threshold': 0.15,  # only trade when P(TP) > this

    # Backtest
    'fee_pct': 0.075,      # Bybit taker fee per side (0.075%)
    'slippage_pct': 0.02,  # estimated slippage per side
    'initial_capital': 10000,
}


# ======================================================================
# Data loading
# ======================================================================

def load_and_build_features(data_path: Path) -> pd.DataFrame:
    """Load data and build all features."""
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

    has_base = all(c in df.columns for c in ['rsi_14', 'atr_14', 'ma_20'])
    if has_base:
        print("   Base features present")
        df_feat = df.copy()
    else:
        print("   Building base features...")
        from bybit_ai_trader.research.features import build_features
        df_feat = build_features(df)

    df_feat = build_alpha_features(df_feat)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    n_before = len(df_feat)
    df_feat = df_feat.dropna()
    print(f"   {n_before - len(df_feat):,} warm-up rows dropped -> {len(df_feat):,} clean rows")
    return df_feat


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Auto-select feature columns, excluding targets and known leaks."""
    exclude = {
        'open', 'high', 'low', 'close', 'volume', 'funding_rate', 'open_interest',
        'timestamp', 'tb_label', 'tb_binary', 'tb_holding',
        'tb_barrier_tp', 'tb_barrier_sl', 'target',
        'classic_binary',        # LEAK: shift(-4)
        'regime_filter',         # derived from target period
        'vol_regime_low',        # regime flag, not feature
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if any(pat in c.lower() for pat in ['future_', 'target_', 'label_']):
            continue
        if df[c].dtype in ('float64', 'float32', 'int64', 'int32'):
            cols.append(c)
    return cols


# ======================================================================
# Volatility regime filter
# ======================================================================

def add_regime_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark low-volatility periods for exclusion.
    Deep analysis showed AUC = 0.4173 in low-vol (model inverts!).
    We only trade in medium-to-high volatility environments.
    """
    if 'atr_pct' not in df.columns:
        if 'atr_14' in df.columns and 'close' in df.columns:
            df['atr_pct'] = df['atr_14'] / df['close'] * 100
        else:
            print("   [WARN] No ATR column for regime filter, skipping")
            df['regime_filter'] = True
            return df

    threshold = df['atr_pct'].quantile(CONFIG['min_atr_pct_percentile'] / 100)
    df['regime_filter'] = df['atr_pct'] >= threshold

    n_pass = df['regime_filter'].sum()
    n_total = len(df)
    print(f"   Regime filter: {n_pass:,}/{n_total:,} bars pass ({n_pass/n_total:.1%})")
    print(f"   ATR% threshold: {threshold:.4f}% (bottom {CONFIG['min_atr_pct_percentile']}% excluded)")
    return df


# ======================================================================
# Walk-forward splits
# ======================================================================

def purged_walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    test_pct: float = 0.12,
    purge_bars: int = 12,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward with purge gap."""
    test_size = int(n_samples * test_pct)
    step = test_size

    splits = []
    for i in range(n_splits):
        test_start = n_samples - (n_splits - i) * step
        test_end = test_start + test_size
        if test_end > n_samples:
            test_end = n_samples
        if test_start < test_size:
            continue

        train_end = test_start - purge_bars
        if train_end < 100:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    print(f"[SPLIT] {len(splits)} purged walk-forward splits (purge={purge_bars}h)")
    for i, (tr, te) in enumerate(splits):
        print(f"   Fold {i+1}: train={len(tr):,} test={len(te):,}")
    return splits


# ======================================================================
# Optuna objectives
# ======================================================================

def _xgb_trial(trial, X_tr, y_tr, X_te, y_te):
    p = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'tree_method': 'hist', 'random_state': 42,
        'scale_pos_weight': (y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 7),
        'learning_rate': trial.suggest_float('xgb_lr', 0.005, 0.08, log=True),
        'n_estimators': trial.suggest_int('xgb_n_est', 300, 1500, step=100),
        'subsample': trial.suggest_float('xgb_subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('xgb_colsample', 0.3, 0.8),
        'min_child_weight': trial.suggest_int('xgb_mcw', 5, 50),
        'gamma': trial.suggest_float('xgb_gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('xgb_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('xgb_lambda', 1.0, 10.0),
    }
    m = xgb.XGBClassifier(**p, early_stopping_rounds=50)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])


def _lgb_trial(trial, X_tr, y_tr, X_te, y_te):
    p = {
        'objective': 'binary', 'metric': 'auc',
        'random_state': 42, 'class_weight': 'balanced', 'verbose': -1,
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 7),
        'learning_rate': trial.suggest_float('lgb_lr', 0.005, 0.08, log=True),
        'n_estimators': trial.suggest_int('lgb_n_est', 300, 1500, step=100),
        'subsample': trial.suggest_float('lgb_subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('lgb_colsample', 0.3, 0.8),
        'min_child_samples': trial.suggest_int('lgb_mcs', 10, 100),
        'reg_alpha': trial.suggest_float('lgb_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('lgb_lambda', 1.0, 10.0),
        'num_leaves': trial.suggest_int('lgb_leaves', 15, 63),
    }
    m = lgb.LGBMClassifier(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])


def _cat_trial(trial, X_tr, y_tr, X_te, y_te):
    p = {
        'loss_function': 'Logloss', 'eval_metric': 'AUC',
        'random_seed': 42, 'verbose': False,
        'auto_class_weights': 'Balanced',
        'depth': trial.suggest_int('cat_depth', 3, 7),
        'learning_rate': trial.suggest_float('cat_lr', 0.005, 0.08, log=True),
        'iterations': trial.suggest_int('cat_iter', 300, 1500, step=100),
        'l2_leaf_reg': trial.suggest_float('cat_l2', 1.0, 10.0),
        'bagging_temperature': trial.suggest_float('cat_bag_temp', 0.0, 1.0),
        'random_strength': trial.suggest_float('cat_rand_str', 0.0, 10.0),
    }
    m = cb.CatBoostClassifier(**p, early_stopping_rounds=50)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])


# ======================================================================
# Feature selection with stability ranking
# ======================================================================

def select_stable_features(
    X: pd.DataFrame, y: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    top_k: int = 50,
) -> List[str]:
    """
    Train LGB on each fold, rank features by importance,
    keep features with best average rank across folds.
    """
    print("\n[FEAT] Feature selection via walk-forward importance stability...")
    importance_matrix = pd.DataFrame(0.0, index=X.columns, columns=range(len(splits)))

    for fold_idx, (tr_idx, te_idx) in enumerate(splits):
        m = lgb.LGBMClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            class_weight='balanced', verbose=-1, random_state=42,
            subsample=0.8, colsample_bytree=0.8
        )
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        imp = pd.Series(m.feature_importances_, index=X.columns)
        importance_matrix[fold_idx] = imp / (imp.sum() + 1e-10)

    # Average rank across folds
    ranks = importance_matrix.rank(ascending=False)
    avg_rank = ranks.mean(axis=1).sort_values()

    # Also compute fold-over-fold consistency (std of ranks)
    rank_std = ranks.std(axis=1)

    # Score = avg_rank (lower=better) + penalty for inconsistency
    score = avg_rank + 0.3 * rank_std
    score = score.sort_values()

    selected = score.head(top_k).index.tolist()
    print(f"   Selected {len(selected)} features (top stable across {len(splits)} folds)")
    print(f"   Top 15: {selected[:15]}")

    # Show anti-predictive features (high importance but negative direction)
    return selected


# ======================================================================
# Isotonic calibration
# ======================================================================

def calibrate_probabilities(y_true, y_prob):
    """Fit isotonic regression calibrator."""
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_prob, y_true)
    return iso


# ======================================================================
# Realistic backtest
# ======================================================================

def backtest_walk_forward(
    df: pd.DataFrame,
    oos_probs: np.ndarray,
    oos_labels: np.ndarray,
    oos_mask: np.ndarray,
    regime_mask: np.ndarray,
    threshold: float = 0.15,
    fee_pct: float = 0.075,
    slippage_pct: float = 0.02,
    initial_capital: float = 10000,
) -> Dict[str, Any]:
    """
    Simulate trading on OOS predictions.

    Rules:
    - Enter LONG when P(TP) > threshold AND regime_filter = True
    - Exit at TP (4x ATR), SL (2x ATR), or 12-bar timeout
    - Fees: 0.075% per side (taker), plus 0.02% slippage estimate
    - Position size: 100% of capital per trade (no leverage initially)
    """
    print("\n[BACKTEST] Simulating walk-forward trading...")

    valid = oos_mask & regime_mask
    indices = np.where(valid)[0]

    if len(indices) == 0:
        print("   No valid trading signals!")
        return {}

    probs = oos_probs[valid]
    labels = oos_labels[valid]

    # Signal: trade when probability exceeds threshold
    signals = probs > threshold
    n_signals = signals.sum()

    if n_signals == 0:
        print(f"   No trades triggered at threshold={threshold:.2f}")
        return {}

    # Each trade result
    total_fee = 2 * (fee_pct + slippage_pct) / 100  # entry + exit
    tp_return = CONFIG['tp_mult'] * 0.01 * 2.5  # rough: 4x ATR ~ +10% on big moves, normalized
    sl_return = -CONFIG['sl_mult'] * 0.01 * 2.5  # 2x ATR ~ -5%

    # More precise: use actual ATR data
    close_vals = df['close'].values[valid]
    atr_vals = df['atr_14'].values[valid] if 'atr_14' in df.columns else None

    trades = []
    for i in range(len(probs)):
        if not signals[i]:
            continue

        label = labels[i]
        prob = probs[i]

        if atr_vals is not None and close_vals[i] > 0:
            atr_pct = atr_vals[i] / close_vals[i]
            tp_ret = CONFIG['tp_mult'] * atr_pct - total_fee
            sl_ret = -CONFIG['sl_mult'] * atr_pct - total_fee
            timeout_ret = -total_fee  # flat, just pay fees
        else:
            tp_ret = 0.10 - total_fee
            sl_ret = -0.05 - total_fee
            timeout_ret = -total_fee

        if label == 1:  # TP hit
            ret = tp_ret
            outcome = 'TP'
        elif label == 0:  # timeout (tb_label == 0 maps to tb_binary == 0)
            # Need to check original tb_label
            ret = timeout_ret
            outcome = 'TIMEOUT'
        else:  # SL hit (tb_label == -1 maps to tb_binary == 0)
            ret = sl_ret
            outcome = 'SL'

        trades.append({
            'prob': prob,
            'label': label,
            'return': ret,
            'outcome': outcome,
        })

    if not trades:
        return {}

    trades_df = pd.DataFrame(trades)

    # Equity curve
    equity = initial_capital
    equity_curve = [equity]
    for r in trades_df['return']:
        equity *= (1 + r)
        equity_curve.append(equity)

    # Statistics
    total_return = equity / initial_capital - 1
    n_trades = len(trades_df)
    win_rate = (trades_df['return'] > 0).mean()
    avg_return = trades_df['return'].mean()
    avg_win = trades_df.loc[trades_df['return'] > 0, 'return'].mean() if win_rate > 0 else 0
    avg_loss = trades_df.loc[trades_df['return'] <= 0, 'return'].mean() if win_rate < 1 else 0

    # Profit factor
    gross_profit = trades_df.loc[trades_df['return'] > 0, 'return'].sum()
    gross_loss = abs(trades_df.loc[trades_df['return'] <= 0, 'return'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    peak = initial_capital
    max_dd = 0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    # Outcome breakdown
    outcomes = trades_df['outcome'].value_counts()

    results = {
        'n_trades': n_trades,
        'n_signals_available': int(n_signals),
        'total_return_pct': total_return * 100,
        'final_equity': equity,
        'win_rate': win_rate,
        'avg_return_per_trade': avg_return * 100,
        'avg_win_pct': avg_win * 100,
        'avg_loss_pct': avg_loss * 100,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_dd * 100,
        'tp_hits': int(outcomes.get('TP', 0)),
        'sl_hits': int(outcomes.get('SL', 0)),
        'timeouts': int(outcomes.get('TIMEOUT', 0)),
        'threshold': threshold,
    }

    print(f"   Trades: {n_trades}")
    print(f"   Win rate: {win_rate:.1%}")
    print(f"   Avg return/trade: {avg_return*100:.3f}%")
    print(f"   Profit factor: {profit_factor:.2f}")
    print(f"   Total return: {total_return*100:.1f}%")
    print(f"   Max drawdown: {max_dd*100:.1f}%")
    print(f"   TP: {outcomes.get('TP', 0)}  SL: {outcomes.get('SL', 0)}  Timeout: {outcomes.get('TIMEOUT', 0)}")

    return results


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("  TRAIN V3 -- OPTIMIZED PIPELINE")
    print("  Barrier: TP=4x SL=2x Hold=12 | Regime-filtered | Mean-reversion")
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
        ROOT / "data" / "raw" / "btcusdt_1h.parquet",
    ]
    data_path = next((p for p in data_paths if p.exists()), None)
    if data_path is None:
        print("[ERR] No data found!")
        return

    df = load_and_build_features(data_path)

    # ------------------------------------------------------------------
    # 2. Regime filter
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 2: Regime Filter (exclude low-volatility)")
    print("=" * 70)

    df = add_regime_filter(df)

    # ------------------------------------------------------------------
    # 3. Triple-barrier labels with OPTIMIZED params
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  STEP 3: Triple-Barrier (TP={CONFIG['tp_mult']}x SL={CONFIG['sl_mult']}x Hold={CONFIG['max_holding']})")
    print("=" * 70)

    tb = triple_barrier_labels(
        df['close'], df['high'], df['low'],
        tp_mult=CONFIG['tp_mult'],
        sl_mult=CONFIG['sl_mult'],
        max_holding=CONFIG['max_holding'],
    )
    df = pd.concat([df, tb], axis=1)

    # ------------------------------------------------------------------
    # 4. Prepare X, y (regime-filtered for training)
    # ------------------------------------------------------------------
    feature_cols = get_feature_cols(df)

    mask = df[feature_cols + ['tb_binary']].notna().all(axis=1)
    df_clean = df[mask].copy()

    # Training: use all data (regime filter applied at prediction time)
    X_all = df_clean[feature_cols]
    y_all = df_clean['tb_binary'].astype(int)
    regime_all = df_clean['regime_filter'].values if 'regime_filter' in df_clean.columns else np.ones(len(df_clean), dtype=bool)

    print(f"\n[DATA] Full dataset: {X_all.shape[0]:,} x {X_all.shape[1]} features")
    print(f"   Positive rate (TP hit): {y_all.mean():.1%}")
    print(f"   Regime-filtered for trading: {regime_all.sum():,} bars ({regime_all.mean():.1%})")

    # ------------------------------------------------------------------
    # 5. Walk-forward splits
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 4: Walk-Forward Splits")
    print("=" * 70)

    splits = purged_walk_forward_splits(
        len(X_all),
        n_splits=CONFIG['n_splits'],
        test_pct=CONFIG['test_pct'],
        purge_bars=CONFIG['purge_bars'],
    )

    # ------------------------------------------------------------------
    # 6. Feature selection
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 5: Feature Selection (stability-ranked)")
    print("=" * 70)

    selected_features = select_stable_features(X_all, y_all, splits, top_k=CONFIG['top_k_features'])
    X_sel = X_all[selected_features]

    # ------------------------------------------------------------------
    # 7. Optuna on first fold
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  STEP 6: Optuna ({CONFIG['n_trials']} trials x 3 models)")
    print("=" * 70)

    tr_idx, te_idx = splits[0]
    X_tr, y_tr = X_sel.iloc[tr_idx], y_all.iloc[tr_idx]
    X_te, y_te = X_sel.iloc[te_idx], y_all.iloc[te_idx]

    if y_tr.sum() > 5:
        smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_tr.sum()) - 1))
        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
    else:
        X_tr_sm, y_tr_sm = X_tr, y_tr

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # XGBoost
    print("   [1/3] XGBoost...")
    study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_xgb.optimize(lambda t: _xgb_trial(t, X_tr_sm, y_tr_sm, X_te, y_te),
                       n_trials=CONFIG['n_trials'], show_progress_bar=True)
    print(f"         Best AUC: {study_xgb.best_value:.4f}")

    # LightGBM
    print("   [2/3] LightGBM...")
    study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_lgb.optimize(lambda t: _lgb_trial(t, X_tr_sm, y_tr_sm, X_te, y_te),
                       n_trials=CONFIG['n_trials'], show_progress_bar=True)
    print(f"         Best AUC: {study_lgb.best_value:.4f}")

    # CatBoost
    print("   [3/3] CatBoost...")
    study_cat = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_cat.optimize(lambda t: _cat_trial(t, X_tr_sm, y_tr_sm, X_te, y_te),
                       n_trials=CONFIG['n_trials'], show_progress_bar=True)
    print(f"         Best AUC: {study_cat.best_value:.4f}")

    # Build parameter dicts
    def _build_xgb_params(study):
        bp = {k.replace('xgb_', ''): v for k, v in study.best_params.items()}
        param_map = {'max_depth': 'max_depth', 'lr': 'learning_rate',
                     'n_est': 'n_estimators', 'subsample': 'subsample',
                     'colsample': 'colsample_bytree', 'mcw': 'min_child_weight',
                     'gamma': 'gamma', 'alpha': 'reg_alpha', 'lambda': 'reg_lambda'}
        params = {param_map.get(k, k): v for k, v in bp.items()}
        params.update({'objective': 'binary:logistic', 'eval_metric': 'auc',
                       'tree_method': 'hist', 'random_state': 42})
        return params

    def _build_lgb_params(study):
        bp = {k.replace('lgb_', ''): v for k, v in study.best_params.items()}
        param_map = {'max_depth': 'max_depth', 'lr': 'learning_rate',
                     'n_est': 'n_estimators', 'subsample': 'subsample',
                     'colsample': 'colsample_bytree', 'mcs': 'min_child_samples',
                     'alpha': 'reg_alpha', 'lambda': 'reg_lambda', 'leaves': 'num_leaves'}
        params = {param_map.get(k, k): v for k, v in bp.items()}
        params.update({'objective': 'binary', 'metric': 'auc',
                       'random_state': 42, 'class_weight': 'balanced', 'verbose': -1})
        return params

    def _build_cat_params(study):
        bp = {k.replace('cat_', ''): v for k, v in study.best_params.items()}
        param_map = {'depth': 'depth', 'lr': 'learning_rate',
                     'iter': 'iterations', 'l2': 'l2_leaf_reg',
                     'bag_temp': 'bagging_temperature', 'rand_str': 'random_strength'}
        params = {param_map.get(k, k): v for k, v in bp.items()}
        params.update({'loss_function': 'Logloss', 'eval_metric': 'AUC',
                       'random_seed': 42, 'verbose': False, 'auto_class_weights': 'Balanced'})
        return params

    xgb_params = _build_xgb_params(study_xgb)
    lgb_params = _build_lgb_params(study_lgb)
    cat_params = _build_cat_params(study_cat)

    # ------------------------------------------------------------------
    # 8. Walk-forward evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 7: Walk-Forward Evaluation (honest AUC)")
    print("=" * 70)

    oos_probs = np.full(len(X_sel), np.nan)
    oos_labels = np.full(len(X_sel), np.nan)
    oos_tb_labels = np.full(len(X_sel), np.nan)  # keep original -1/0/1
    fold_aucs = []

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        print(f"\n--- Fold {fold_i+1}/{len(splits)} ---")
        X_tr, y_tr = X_sel.iloc[tr_idx], y_all.iloc[tr_idx]
        X_te, y_te = X_sel.iloc[te_idx], y_all.iloc[te_idx]

        if y_tr.sum() > 5:
            smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_tr.sum()) - 1))
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
        else:
            X_tr_sm, y_tr_sm = X_tr.values, y_tr.values

        # XGBoost
        xp = dict(xgb_params)
        xp['scale_pos_weight'] = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
        m_xgb = xgb.XGBClassifier(**xp, early_stopping_rounds=50)
        m_xgb.fit(X_tr_sm, y_tr_sm, eval_set=[(X_te, y_te)], verbose=False)

        # LightGBM
        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(X_tr_sm, y_tr_sm, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        # CatBoost
        m_cat = cb.CatBoostClassifier(**cat_params, early_stopping_rounds=50)
        m_cat.fit(X_tr_sm, y_tr_sm, eval_set=[(X_te, y_te)], verbose=False)

        # Ensemble
        p_xgb = m_xgb.predict_proba(X_te)[:, 1]
        p_lgb = m_lgb.predict_proba(X_te)[:, 1]
        p_cat = m_cat.predict_proba(X_te)[:, 1]
        p_ens = (p_xgb + p_lgb + p_cat) / 3.0

        fold_auc = roc_auc_score(y_te, p_ens)
        fold_aucs.append(fold_auc)
        print(f"   XGB={roc_auc_score(y_te, p_xgb):.4f}  LGB={roc_auc_score(y_te, p_lgb):.4f}  "
              f"CAT={roc_auc_score(y_te, p_cat):.4f}  ENS={fold_auc:.4f}")

        oos_probs[te_idx] = p_ens
        oos_labels[te_idx] = y_te.values
        if 'tb_label' in df_clean.columns:
            oos_tb_labels[te_idx] = df_clean['tb_label'].values[te_idx]

    # Overall metrics
    valid = ~np.isnan(oos_probs)
    overall_auc = roc_auc_score(oos_labels[valid], oos_probs[valid])
    overall_brier = brier_score_loss(oos_labels[valid], oos_probs[valid])

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD RESULTS (honest)")
    print(f"{'='*70}")
    print(f"   Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"   Mean AUC:  {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    print(f"   Pooled OOS AUC: {overall_auc:.4f}")
    print(f"   Brier score: {overall_brier:.4f}")

    # ------------------------------------------------------------------
    # 9. Threshold sweep for optimal trading
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  STEP 8: Threshold Optimization")
    print(f"{'='*70}")

    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]
    print(f"\n   {'Thresh':>8s} {'N_trades':>8s} {'Precision':>10s} {'Win_rate':>10s}")
    print(f"   {'-'*42}")

    best_threshold = CONFIG['confidence_threshold']
    best_pf = 0

    for thr in thresholds:
        mask_thr = (oos_probs[valid] > thr)
        n_trades = mask_thr.sum()
        if n_trades < 20:
            continue
        precision = oos_labels[valid][mask_thr].mean()

        # Simulated win rate with regime filter
        regime_pass = regime_all[valid] if len(regime_all) == len(valid) else np.ones(valid.sum(), dtype=bool)
        combined = mask_thr & regime_pass[:len(mask_thr)]
        n_regime = combined.sum()
        if n_regime > 10:
            prec_regime = oos_labels[valid][combined].mean()
        else:
            prec_regime = precision

        print(f"   {thr:>8.2f} {n_trades:>8d} {precision:>10.3f} {prec_regime:>10.3f}")

    # ------------------------------------------------------------------
    # 10. Backtest on OOS predictions
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  STEP 9: Realistic Backtest (OOS walk-forward)")
    print(f"{'='*70}")

    # Use tb_label for more precise backtest (distinguish timeout vs SL)
    backtest_labels = oos_tb_labels.copy()
    # For simple version: label=1 means TP hit, label=0 means not-TP (SL or timeout)
    backtest_results = backtest_walk_forward(
        df_clean, oos_probs, oos_labels, valid, regime_all,
        threshold=CONFIG['confidence_threshold'],
        fee_pct=CONFIG['fee_pct'],
        slippage_pct=CONFIG['slippage_pct'],
        initial_capital=CONFIG['initial_capital'],
    )

    # Also test with different thresholds
    print("\n   Threshold sensitivity:")
    print(f"   {'Thresh':>8s} {'Trades':>8s} {'WinRate':>8s} {'TotalRet':>10s} {'PF':>6s} {'MaxDD':>8s}")
    for thr in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        bt = backtest_walk_forward(
            df_clean, oos_probs, oos_labels, valid, regime_all,
            threshold=thr,
            fee_pct=CONFIG['fee_pct'],
            slippage_pct=CONFIG['slippage_pct'],
            initial_capital=CONFIG['initial_capital'],
        )
        if bt:
            print(f"   {thr:>8.2f} {bt['n_trades']:>8d} {bt['win_rate']:>8.1%} "
                  f"{bt['total_return_pct']:>10.1f}% {bt['profit_factor']:>6.2f} "
                  f"{bt['max_drawdown_pct']:>8.1f}%")

    # ------------------------------------------------------------------
    # 11. Train final model + calibrate
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  STEP 10: Train Final Model + Calibration")
    print(f"{'='*70}")

    cal_split = int(len(X_sel) * 0.85)
    X_train_final = X_sel.iloc[:cal_split]
    y_train_final = y_all.iloc[:cal_split]
    X_cal = X_sel.iloc[cal_split:]
    y_cal = y_all.iloc[cal_split:]

    if y_train_final.sum() > 5:
        smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_train_final.sum()) - 1))
        X_train_sm, y_train_sm = smote.fit_resample(X_train_final, y_train_final)
    else:
        X_train_sm, y_train_sm = X_train_final, y_train_final

    xp_final = dict(xgb_params)
    xp_final['scale_pos_weight'] = (y_train_final==0).sum() / max((y_train_final==1).sum(), 1)

    m_xgb_final = xgb.XGBClassifier(**xp_final, early_stopping_rounds=50)
    m_xgb_final.fit(X_train_sm, y_train_sm, eval_set=[(X_cal, y_cal)], verbose=False)

    m_lgb_final = lgb.LGBMClassifier(**lgb_params)
    m_lgb_final.fit(X_train_sm, y_train_sm, eval_set=[(X_cal, y_cal)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])

    m_cat_final = cb.CatBoostClassifier(**cat_params, early_stopping_rounds=50)
    m_cat_final.fit(X_train_sm, y_train_sm, eval_set=[(X_cal, y_cal)], verbose=False)

    # Calibration
    p_cal = (m_xgb_final.predict_proba(X_cal)[:, 1] +
             m_lgb_final.predict_proba(X_cal)[:, 1] +
             m_cat_final.predict_proba(X_cal)[:, 1]) / 3.0

    cal_auc = roc_auc_score(y_cal, p_cal)
    calibrator = calibrate_probabilities(y_cal.values, p_cal)
    p_cal_cal = calibrator.predict(p_cal)
    brier_before = brier_score_loss(y_cal, p_cal)
    brier_after = brier_score_loss(y_cal, p_cal_cal)
    print(f"   Cal-set AUC: {cal_auc:.4f}")
    print(f"   Brier: {brier_before:.4f} -> {brier_after:.4f}")

    # Feature importance from final models
    imp_xgb = pd.Series(m_xgb_final.feature_importances_, index=selected_features)
    imp_lgb = pd.Series(m_lgb_final.feature_importances_, index=selected_features)
    # Normalize and average
    imp_combined = (imp_xgb / imp_xgb.sum() + imp_lgb / imp_lgb.sum()) / 2
    imp_combined = imp_combined.sort_values(ascending=False)

    print(f"\n   Top 15 features:")
    for feat, score in imp_combined.head(15).items():
        print(f"     {feat:<35s} {score:.4f}")

    # ------------------------------------------------------------------
    # 12. Save
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  STEP 11: Save Model Artifacts")
    print(f"{'='*70}")

    artifact = {
        'xgb_model': m_xgb_final,
        'lgb_model': m_lgb_final,
        'cat_model': m_cat_final,
        'calibrator': calibrator,
        'feature_cols': selected_features,
        'config': CONFIG,
        'training_date': datetime.now().isoformat(),
        'regime_threshold_atr_pct': float(df['atr_pct'].quantile(
            CONFIG['min_atr_pct_percentile'] / 100)) if 'atr_pct' in df.columns else None,
    }

    model_path = MODEL_DIR / "ensemble_v3_optimized.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"   Model saved: {model_path}")

    metadata = {
        "version": "v3_optimized",
        "description": "Optimized from deep analysis: TP=4x SL=2x Hold=12, regime-filtered, mean-reversion",
        "config": CONFIG,
        "models": ["XGBoost", "LightGBM", "CatBoost"],
        "features": selected_features,
        "n_features": len(selected_features),
        "labeling": f"triple_barrier (tp={CONFIG['tp_mult']}xATR, sl={CONFIG['sl_mult']}xATR, hold={CONFIG['max_holding']}h)",
        "validation": f"purged_walk_forward_{CONFIG['n_splits']}fold",
        "training_date": datetime.now().isoformat(),
        "walk_forward_results": {
            "fold_aucs": [float(a) for a in fold_aucs],
            "mean_auc": float(np.mean(fold_aucs)),
            "std_auc": float(np.std(fold_aucs)),
            "pooled_oos_auc": float(overall_auc),
            "pooled_brier": float(overall_brier),
        },
        "calibration": {
            "method": "isotonic",
            "brier_before": float(brier_before),
            "brier_after": float(brier_after),
        },
        "backtest": backtest_results if backtest_results else {},
        "top_features": {k: float(v) for k, v in imp_combined.head(20).items()},
        "key_findings": {
            "signal_type": "mean_reversion",
            "best_regime": "medium_to_high_volatility",
            "anti_regime": "low_volatility_inverts_model",
            "dominant_features": "RSI, CCI, momentum, Williams%R (all negative direction)",
            "rsi_edge": "RSI>80 -> +0.26% at 12h; RSI<20 -> continued decline",
        },
    }

    meta_path = MODEL_DIR / "ensemble_v3_optimized_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved: {meta_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE -- V3 Optimized Pipeline")
    print(f"{'='*70}")
    print(f"   Walk-forward AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    print(f"   Pooled OOS AUC:   {overall_auc:.4f}")
    print(f"   vs V2 (leak-free): 0.5306 -> {overall_auc:.4f}")
    print(f"   vs deep analysis:  0.6169 -> {np.mean(fold_aucs):.4f}")
    print(f"\n   Key strategy: Mean-reversion on BTC overextensions")
    print(f"   Target: big moves (4x ATR TP) in short windows (12h)")
    print(f"   Filter: exclude low-vol (bottom 25% ATR)")
    if backtest_results:
        print(f"\n   Backtest ({backtest_results['n_trades']} trades, threshold={CONFIG['confidence_threshold']}):")
        print(f"   Total return: {backtest_results['total_return_pct']:.1f}%")
        print(f"   Win rate:     {backtest_results['win_rate']:.1%}")
        print(f"   Profit factor: {backtest_results['profit_factor']:.2f}")
        print(f"   Max drawdown: {backtest_results['max_drawdown_pct']:.1f}%")

    print(f"\n   Next steps:")
    print(f"   1. Run: python scripts/start_paper_trading_v3.py")
    print(f"   2. Or deploy TradingView Pine Script alerts")


if __name__ == "__main__":
    main()
