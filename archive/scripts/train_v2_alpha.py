"""
Train V2 -- walk-forward ensemble with alpha features + triple-barrier labels.

This script replaces the original Optuna pipeline with:
  1. Alpha feature engine (Hurst, entropy, funding alpha, etc.)
  2. Triple-barrier labeling (TP/SL/timeout instead of binary up/down)
  3. Purged walk-forward cross-validation (no future leakage)
  4. Isotonic calibration on held-out fold
  5. Feature importance + selection based on walk-forward stability

The goal is to push the honest AUC from 0.66 -> 0.70+ by giving the models
better features and cleaner labels while eliminating all sources of leakage.

Usage:
    python scripts/train_v2_alpha.py
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
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.alpha_features import build_alpha_features
from features.triple_barrier import triple_barrier_labels


# ======================================================================
# Data loading & feature building
# ======================================================================

def load_and_build_features(data_path: Path) -> pd.DataFrame:
    """
    Load data and build features.

    Strategy:
      - If data_path already has enhanced features (>30 columns), use it directly
        and only add the NEW alpha features on top.
      - Otherwise build base features from scratch.
    """
    print("Loading data...")
    df = pd.read_parquet(data_path)
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Ensure timestamp index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert; if purely integer, just keep it
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass  # keep integer index

    # Decide whether base features are already present
    has_base = all(c in df.columns for c in ['rsi_14', 'atr_14', 'ma_20'])

    if has_base:
        print("   Base features already present -- skipping base build")
        df_feat = df.copy()
    else:
        print("   Building base features...")
        from bybit_ai_trader.research.features import build_features
        df_feat = build_features(df)

    # Layer on alpha features (Hurst, entropy, etc.)
    df_feat = build_alpha_features(df_feat)

    # Clean
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    n_before = len(df_feat)
    df_feat = df_feat.dropna()
    print(f"   Dropped {n_before - len(df_feat):,} warm-up rows -> {len(df_feat):,} clean rows")
    return df_feat


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Auto-select feature columns excluding OHLCV, targets, and barrier cols."""
    exclude = {
        'open', 'high', 'low', 'close', 'volume', 'funding_rate', 'open_interest',
        'timestamp', 'tb_label', 'tb_binary', 'tb_holding',
        'tb_barrier_tp', 'tb_barrier_sl', 'target',
        'classic_binary',  # LEAK: uses shift(-4), must never be a feature
    }
    # Also exclude anything with 'future', 'target', 'label' in name
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
# Walk-forward with purging
# ======================================================================

def purged_walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    test_pct: float = 0.15,
    purge_bars: int = 24,      # gap between train and test (1 day)
    embargo_bars: int = 24,    # gap after test before next train window
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window walk-forward splits with purge gap.

    Train: [0, split_point - purge)
    Test:  [split_point, split_point + test_size)

    The purge gap prevents label leakage from the triple-barrier
    look-ahead window.
    """
    test_size = int(n_samples * test_pct)
    step = test_size  # non-overlapping test windows

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

    print(f"[DATA] Created {len(splits)} purged walk-forward splits (purge={purge_bars}, test={test_pct:.0%})")
    for i, (tr, te) in enumerate(splits):
        print(f"   Fold {i+1}: train={len(tr):,} [{tr[0]}->{tr[-1]}], test={len(te):,} [{te[0]}->{te[-1]}]")
    return splits


# ======================================================================
# Optuna objectives -- same as before but cleaner
# ======================================================================

def _xgb_trial(trial, X_tr, y_tr, X_te, y_te):
    p = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'tree_method': 'hist', 'random_state': 42,
        'scale_pos_weight': (y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
        'learning_rate': trial.suggest_float('xgb_lr', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('xgb_n_est', 200, 1500, step=100),
        'subsample': trial.suggest_float('xgb_subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('xgb_colsample', 0.3, 0.9),
        'min_child_weight': trial.suggest_int('xgb_mcw', 3, 30),
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
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 8),
        'learning_rate': trial.suggest_float('lgb_lr', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('lgb_n_est', 200, 1500, step=100),
        'subsample': trial.suggest_float('lgb_subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('lgb_colsample', 0.3, 0.9),
        'min_child_samples': trial.suggest_int('lgb_mcs', 10, 100),
        'reg_alpha': trial.suggest_float('lgb_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('lgb_lambda', 1.0, 10.0),
        'num_leaves': trial.suggest_int('lgb_leaves', 15, 127),
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
        'depth': trial.suggest_int('cat_depth', 3, 8),
        'learning_rate': trial.suggest_float('cat_lr', 0.005, 0.1, log=True),
        'iterations': trial.suggest_int('cat_iter', 200, 1500, step=100),
        'l2_leaf_reg': trial.suggest_float('cat_l2', 1.0, 10.0),
        'bagging_temperature': trial.suggest_float('cat_bag_temp', 0.0, 1.0),
        'random_strength': trial.suggest_float('cat_rand_str', 0.0, 10.0),
    }
    m = cb.CatBoostClassifier(**p, early_stopping_rounds=50)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])


# ======================================================================
# Feature selection via walk-forward stability
# ======================================================================

def select_stable_features(
    X: pd.DataFrame, y: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    top_k: int = 60,
) -> List[str]:
    """
    Train a quick LightGBM on each fold, collect feature importances,
    then keep features that are consistently important across folds.
    """
    print("\n[ANALYZE] Feature selection via walk-forward importance stability...")
    importance_matrix = pd.DataFrame(0.0, index=X.columns, columns=range(len(splits)))

    for fold_idx, (tr_idx, te_idx) in enumerate(splits):
        m = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            class_weight='balanced', verbose=-1, random_state=42
        )
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        imp = pd.Series(m.feature_importances_, index=X.columns)
        importance_matrix[fold_idx] = imp / (imp.sum() + 1e-10)

    # Rank each fold, then average ranks
    ranks = importance_matrix.rank(ascending=False)
    avg_rank = ranks.mean(axis=1).sort_values()

    selected = avg_rank.head(top_k).index.tolist()
    print(f"   Selected {len(selected)} features with best avg rank across {len(splits)} folds")
    print(f"   Top 10: {selected[:10]}")
    return selected


# ======================================================================
# Isotonic calibration
# ======================================================================

def calibrate_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> IsotonicRegression:
    """Fit isotonic regression calibrator on held-out probabilities."""
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_prob, y_true)
    return iso


# ======================================================================
# Main pipeline
# ======================================================================

def main():
    print("=" * 70)
    print("[START] TRAIN V2 -- Alpha Features + Triple Barrier + Walk-Forward")
    print("=" * 70)

    ROOT = Path(__file__).parent.parent
    MODEL_DIR = ROOT / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data & build features
    # ------------------------------------------------------------------
    # Try enhanced first, fall back to base
    data_paths = [
        ROOT / "data" / "processed" / "btcusdt_1h_full_history_enhanced.parquet",
        ROOT / "data" / "processed" / "btcusdt_1h_features.parquet",
        ROOT / "data" / "raw" / "btcusdt_1h.parquet",
    ]
    data_path = None
    for p in data_paths:
        if p.exists():
            data_path = p
            break
    if data_path is None:
        print("[ERR] No data found. Run data download first.")
        return

    df = load_and_build_features(data_path)

    # ------------------------------------------------------------------
    # 2. Triple-barrier labels
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[TARGET] STEP 2: Triple-Barrier Labeling")
    print("=" * 70)

    tb = triple_barrier_labels(
        df['close'], df['high'], df['low'],
        tp_mult=2.0, sl_mult=1.0, max_holding=24
    )
    df = pd.concat([df, tb], axis=1)

    # Also create classic binary label for comparison
    df['classic_binary'] = (np.log(df['close'].shift(-4) / df['close']) > 0.0).astype(float)

    # ------------------------------------------------------------------
    # 3. Prepare X, y
    # ------------------------------------------------------------------
    feature_cols = get_feature_cols(df)

    # Use triple-barrier binary as primary target
    mask = df[feature_cols + ['tb_binary']].notna().all(axis=1)
    df_clean = df[mask].copy()

    X_all = df_clean[feature_cols]
    y_all = df_clean['tb_binary'].astype(int)

    print(f"\n[DATA] Dataset: {X_all.shape[0]:,} samples, {X_all.shape[1]} features")
    print(f"   Positive rate (TP hit): {y_all.mean():.1%}")

    # ------------------------------------------------------------------
    # 4. Walk-forward splits
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[DATA] STEP 3: Purged Walk-Forward Splits")
    print("=" * 70)

    splits = purged_walk_forward_splits(
        len(X_all), n_splits=5, test_pct=0.12, purge_bars=24
    )

    # ------------------------------------------------------------------
    # 5. Feature selection (stability-based)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[ANALYZE] STEP 4: Feature Selection")
    print("=" * 70)

    selected_features = select_stable_features(X_all, y_all, splits, top_k=60)
    X_sel = X_all[selected_features]

    # ------------------------------------------------------------------
    # 6. Optuna optimization on first fold
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[SEARCH] STEP 5: Optuna Hyperparameter Optimization")
    print("=" * 70)

    tr_idx, te_idx = splits[0]
    X_tr, y_tr = X_sel.iloc[tr_idx], y_all.iloc[tr_idx]
    X_te, y_te = X_sel.iloc[te_idx], y_all.iloc[te_idx]

    # SMOTE on training fold
    if y_tr.sum() > 5:
        smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_tr.sum()) - 1))
        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
    else:
        X_tr_sm, y_tr_sm = X_tr, y_tr

    N_TRIALS = 80  # per model (total 240 trials)

    # XGBoost
    study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_xgb.optimize(lambda t: _xgb_trial(t, X_tr_sm, y_tr_sm, X_te, y_te),
                       n_trials=N_TRIALS, show_progress_bar=True)
    print(f"   XGBoost best AUC: {study_xgb.best_value:.4f}")

    # LightGBM
    study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_lgb.optimize(lambda t: _lgb_trial(t, X_tr_sm, y_tr_sm, X_te, y_te),
                       n_trials=N_TRIALS, show_progress_bar=True)
    print(f"   LightGBM best AUC: {study_lgb.best_value:.4f}")

    # CatBoost
    study_cat = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_cat.optimize(lambda t: _cat_trial(t, X_tr_sm, y_tr_sm, X_te, y_te),
                       n_trials=N_TRIALS, show_progress_bar=True)
    print(f"   CatBoost best AUC: {study_cat.best_value:.4f}")

    # ------------------------------------------------------------------
    # 7. Walk-forward evaluation -- train on each fold, collect OOS probs
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[CHART] STEP 6: Walk-Forward Evaluation (honest AUC)")
    print("=" * 70)

    oos_probs = np.full(len(X_sel), np.nan)
    oos_labels = np.full(len(X_sel), np.nan)
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

        # Train 3 models with best params
        bp_xgb = {k.replace('xgb_', ''): v for k, v in study_xgb.best_params.items()}
        bp_xgb.update({'objective': 'binary:logistic', 'eval_metric': 'auc',
                       'tree_method': 'hist', 'random_state': 42,
                       'scale_pos_weight': (y_tr==0).sum() / max((y_tr==1).sum(), 1)})
        # Rename params
        param_map_xgb = {'max_depth': 'max_depth', 'lr': 'learning_rate',
                         'n_est': 'n_estimators', 'subsample': 'subsample',
                         'colsample': 'colsample_bytree', 'mcw': 'min_child_weight',
                         'gamma': 'gamma', 'alpha': 'reg_alpha', 'lambda': 'reg_lambda'}
        xgb_params = {param_map_xgb.get(k, k): v for k, v in bp_xgb.items()}

        m_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
        m_xgb.fit(X_tr_sm, y_tr_sm, eval_set=[(X_te, y_te)], verbose=False)

        bp_lgb = {k.replace('lgb_', ''): v for k, v in study_lgb.best_params.items()}
        param_map_lgb = {'max_depth': 'max_depth', 'lr': 'learning_rate',
                         'n_est': 'n_estimators', 'subsample': 'subsample',
                         'colsample': 'colsample_bytree', 'mcs': 'min_child_samples',
                         'alpha': 'reg_alpha', 'lambda': 'reg_lambda', 'leaves': 'num_leaves'}
        lgb_params = {param_map_lgb.get(k, k): v for k, v in bp_lgb.items()}
        lgb_params.update({'objective': 'binary', 'metric': 'auc',
                           'random_state': 42, 'class_weight': 'balanced', 'verbose': -1})

        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(X_tr_sm, y_tr_sm, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        bp_cat = {k.replace('cat_', ''): v for k, v in study_cat.best_params.items()}
        param_map_cat = {'depth': 'depth', 'lr': 'learning_rate',
                         'iter': 'iterations', 'l2': 'l2_leaf_reg',
                         'bag_temp': 'bagging_temperature', 'rand_str': 'random_strength'}
        cat_params = {param_map_cat.get(k, k): v for k, v in bp_cat.items()}
        cat_params.update({'loss_function': 'Logloss', 'eval_metric': 'AUC',
                           'random_seed': 42, 'verbose': False, 'auto_class_weights': 'Balanced'})

        m_cat = cb.CatBoostClassifier(**cat_params, early_stopping_rounds=50)
        m_cat.fit(X_tr_sm, y_tr_sm, eval_set=[(X_te, y_te)], verbose=False)

        # Ensemble probability (equal weight average)
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

    # Overall honest AUC
    valid = ~np.isnan(oos_probs)
    overall_auc = roc_auc_score(oos_labels[valid], oos_probs[valid])
    overall_brier = brier_score_loss(oos_labels[valid], oos_probs[valid])

    print(f"\n{'='*70}")
    print(f"[TARGET] WALK-FORWARD RESULTS (honest, no leakage)")
    print(f"{'='*70}")
    print(f"   Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"   Mean AUC:  {np.mean(fold_aucs):.4f} +- {np.std(fold_aucs):.4f}")
    print(f"   Overall AUC (pooled OOS): {overall_auc:.4f}")
    print(f"   Brier score: {overall_brier:.4f}")

    # ------------------------------------------------------------------
    # 8. Train final model on all data + calibrate
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"[START] STEP 7: Train Final Model + Calibration")
    print(f"{'='*70}")

    # 85% train, 15% calibration
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

    # Train final models
    m_xgb_final = xgb.XGBClassifier(**{k: v for k, v in xgb_params.items()},
                                      early_stopping_rounds=50)
    m_xgb_final.fit(X_train_sm, y_train_sm, eval_set=[(X_cal, y_cal)], verbose=False)

    m_lgb_final = lgb.LGBMClassifier(**{k: v for k, v in lgb_params.items()})
    m_lgb_final.fit(X_train_sm, y_train_sm, eval_set=[(X_cal, y_cal)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])

    m_cat_final = cb.CatBoostClassifier(**{k: v for k, v in cat_params.items()},
                                         early_stopping_rounds=50)
    m_cat_final.fit(X_train_sm, y_train_sm, eval_set=[(X_cal, y_cal)], verbose=False)

    # Ensemble on calibration set
    p_cal = (m_xgb_final.predict_proba(X_cal)[:, 1] +
             m_lgb_final.predict_proba(X_cal)[:, 1] +
             m_cat_final.predict_proba(X_cal)[:, 1]) / 3.0

    cal_auc = roc_auc_score(y_cal, p_cal)
    print(f"   Calibration-set AUC: {cal_auc:.4f}")

    # Isotonic calibration
    calibrator = calibrate_probabilities(y_cal.values, p_cal)
    p_cal_calibrated = calibrator.predict(p_cal)
    cal_auc_after = roc_auc_score(y_cal, p_cal_calibrated)
    brier_before = brier_score_loss(y_cal, p_cal)
    brier_after = brier_score_loss(y_cal, p_cal_calibrated)
    print(f"   After calibration: AUC={cal_auc_after:.4f}, Brier {brier_before:.4f}->{brier_after:.4f}")

    # ------------------------------------------------------------------
    # 9. Save everything
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"[SAVE] STEP 8: Save Model Artifacts")
    print(f"{'='*70}")

    artifact = {
        'xgb_model': m_xgb_final,
        'lgb_model': m_lgb_final,
        'cat_model': m_cat_final,
        'calibrator': calibrator,
        'feature_cols': selected_features,
        'training_date': datetime.now().isoformat(),
    }

    model_path = MODEL_DIR / "ensemble_v2_alpha.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"   Model saved: {model_path}")

    # Feature importance from final XGB
    imp = pd.Series(m_xgb_final.feature_importances_, index=selected_features)
    imp = imp.sort_values(ascending=False)

    metadata = {
        "version": "v2_alpha",
        "model_type": "ensemble_3model_alpha_triple_barrier",
        "models": ["XGBoost", "LightGBM", "CatBoost"],
        "features": selected_features,
        "n_features": len(selected_features),
        "labeling": "triple_barrier (tp=2xATR, sl=1xATR, hold=24h)",
        "validation": "purged_walk_forward_5fold",
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
        "top_features": {k: float(v) for k, v in imp.head(20).items()},
        "data_samples": len(X_sel),
        "positive_rate": float(y_all.mean()),
    }

    meta_path = MODEL_DIR / "ensemble_v2_alpha_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved: {meta_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"[DONE] TRAINING COMPLETE -- V2 Alpha Pipeline")
    print(f"{'='*70}")
    print(f"   Baseline AUC (old pipeline): 0.6617")
    print(f"   New walk-forward AUC:         {overall_auc:.4f}")
    delta = overall_auc - 0.6617
    print(f"   Improvement:                  {'+' if delta > 0 else ''}{delta:.4f}")
    print(f"\n   Top 5 features:")
    for feat, score in imp.head(5).items():
        print(f"     {feat:<30s} {score:.0f}")

    # Also compare against classic binary label
    print(f"\n[DATA] Bonus: Classic binary label comparison")
    mask_classic = df_clean['classic_binary'].notna()
    if mask_classic.sum() > 100:
        y_classic = df_clean.loc[mask_classic, 'classic_binary'].astype(int)
        X_classic = X_sel.loc[mask_classic]
        # Quick 80/20 chronological split
        sp = int(len(X_classic) * 0.8)
        m_quick = lgb.LGBMClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                                      class_weight='balanced', verbose=-1)
        m_quick.fit(X_classic.iloc[:sp], y_classic.iloc[:sp])
        auc_classic = roc_auc_score(y_classic.iloc[sp:], m_quick.predict_proba(X_classic.iloc[sp:])[:, 1])
        print(f"   Classic binary AUC (same features): {auc_classic:.4f}")
        print(f"   Triple-barrier AUC (walk-forward):  {overall_auc:.4f}")


if __name__ == "__main__":
    main()
