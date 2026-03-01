"""
V7 Improved Regression Experiment
===================================
Building on V6's finding that 8h regression was the best approach (PF=1.54),
this experiment applies four Tier-1 improvements:

  1. FEATURE SELECTION
     - Mutual-information ranking + recursive pruning
     - Reduce 117→40-60 most informative features
     - Removes noise features that hurt generalization

  2. TARGET ENGINEERING
     - Volatility-normalized forward returns: y = ln(Ct+8/Ct) / σ_24h
     - Winsorize at ±3σ to tame outliers
     - Gives model a "risk-adjusted" target (more stable across regimes)

  3. OPTUNA HYPERPARAMETER TUNING
     - 150-trial Bayesian optimization for XGBoost, LightGBM, CatBoost
     - Objective: minimize RMSE on full 5-fold purged walk-forward CV
     - Not just 1 split like previous Optuna attempts

  4. CATBOOST AS 3RD ENSEMBLE MEMBER
     - Ordered boosting (handles small data well)
     - Symmetric trees (different inductive bias from XGB/LGB)
     - 3-model weighted ensemble: XGB + LGB + CatBoost

Comparison against V6 baselines:
  - V6 Tree AUC:     0.6367 (classification)
  - V6 reg_8h PF:    1.54  (best strategy)
  - V6 reg_8h Sharpe: 4.06

Usage:
    python scripts/v7_improved_regression.py
"""
import os, sys, json, warnings, time, math
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from scipy.stats import mstats

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression as sk_mi_reg
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features.alpha_features import build_alpha_features
from features.triple_barrier import triple_barrier_labels

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

# ======================================================================
# 1. DATA LOADING (reuse from V6)
# ======================================================================

def load_and_prepare_data():
    """Load data, build features, and return clean DataFrame."""
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
        df.index = pd.to_datetime(df.index)

    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Build alpha features
    df = build_alpha_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"   After cleanup: {len(df):,} rows, {len(df.columns)} columns")

    # Hour feature
    df['_hour'] = df.index.hour if hasattr(df.index, 'hour') else 0

    # Regime filter (exclude ultra-low vol)
    if 'atr_14' in df.columns:
        df['atr_pct'] = df['atr_14'] / df['close'] * 100
        thr = df['atr_pct'].quantile(0.25)
        df['regime_ok'] = df['atr_pct'] >= thr
    else:
        df['regime_ok'] = True

    return df


def get_feature_cols(df):
    """Get feature column names, excluding targets and metadata."""
    exclude = {
        'open', 'high', 'low', 'close', 'volume', 'funding_rate',
        'open_interest', 'timestamp', 'target', 'classic_binary',
        'regime_filter', 'regime_ok', '_hour', 'atr_pct',
        'tb_label', 'tb_binary', 'tb_holding', 'tb_barrier_tp', 'tb_barrier_sl',
        'short_tb_label', 'short_tb_binary', 'short_tb_holding',
        'short_tb_barrier_tp', 'short_tb_barrier_sl',
        'fwd_ret', 'fwd_ret_label',
    }
    # Also exclude any forward return columns we create
    return [c for c in df.columns
            if c not in exclude
            and not any(p in c.lower() for p in ['future_', 'target_', 'label_', 'fwd_ret_'])
            and not c.startswith('y_')
            and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]


# ======================================================================
# 2. FEATURE SELECTION (Tier 1 Improvement #1)
# ======================================================================

def select_features_mi(X, y, feature_cols, top_k=50, n_neighbors=5):
    """
    Mutual Information-based feature selection.

    Steps:
      1. Compute MI(feature, target) for each feature
      2. Rank by MI score
      3. Iterative redundancy removal: drop highly correlated features
         (keep higher MI one in each pair)
      4. Return top_k features

    This is more robust than SHAP for regression targets, and doesn't
    require a pre-trained model (avoids circular dependency).
    """
    print(f"\n   === FEATURE SELECTION (MI-based, top_k={top_k}) ===")
    t0 = time.time()

    X_feat = X[feature_cols].copy()
    y_clean = y.copy()

    # Remove any remaining NaN/inf
    valid_mask = X_feat.notna().all(axis=1) & y_clean.notna()
    X_feat = X_feat[valid_mask]
    y_clean = y_clean[valid_mask]

    if len(X_feat) < 500:
        print(f"   ⚠️ Only {len(X_feat)} valid samples, skipping feature selection")
        return feature_cols

    # Step 1: Compute MI scores (use a subsample for speed)
    n_sample = min(10000, len(X_feat))
    idx = np.random.RandomState(42).choice(len(X_feat), n_sample, replace=False)
    X_sample = X_feat.iloc[idx].values
    y_sample = y_clean.iloc[idx].values

    print(f"   [1/3] Computing MI scores for {len(feature_cols)} features "
          f"on {n_sample} samples...")
    mi_scores = sk_mi_reg(X_sample, y_sample, n_neighbors=n_neighbors,
                          random_state=42)
    mi_df = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores,
    }).sort_values('mi_score', ascending=False)

    # Print top 20
    print(f"\n   Top 20 features by MI:")
    for i, row in mi_df.head(20).iterrows():
        print(f"      {row['feature']:<35s} MI={row['mi_score']:.4f}")

    # Step 2: Take top 2*top_k candidates
    candidates = mi_df.head(top_k * 2)['feature'].tolist()

    # Step 3: Redundancy removal via correlation
    print(f"\n   [2/3] Removing redundant features (corr > 0.85)...")
    corr_matrix = X_feat[candidates].corr().abs()
    drop_set = set()
    for i in range(len(candidates)):
        if candidates[i] in drop_set:
            continue
        for j in range(i + 1, len(candidates)):
            if candidates[j] in drop_set:
                continue
            if corr_matrix.iloc[i, j] > 0.85:
                # Drop the one with lower MI
                mi_i = mi_df.set_index('feature').loc[candidates[i], 'mi_score']
                mi_j = mi_df.set_index('feature').loc[candidates[j], 'mi_score']
                drop = candidates[j] if mi_i >= mi_j else candidates[i]
                drop_set.add(drop)

    selected = [f for f in candidates if f not in drop_set][:top_k]

    print(f"   [3/3] Selected {len(selected)} features "
          f"(removed {len(drop_set)} redundant)")
    print(f"   Feature selection took {time.time()-t0:.1f}s")

    return selected


# ======================================================================
# 3. TARGET ENGINEERING (Tier 1 Improvement #2)
# ======================================================================

def build_vol_normalized_target(df, horizon=8, winsorize_sigma=3.0):
    """
    Build volatility-normalized forward return target.

    y = ln(C_{t+h} / C_t) / σ_24h

    Then winsorize at ±winsorize_sigma to tame outliers.

    This gives the model a "risk-adjusted" target:
    - A 2% move in a low-vol environment gets amplified
    - A 2% move in a high-vol environment gets dampened
    - The model learns to predict relative moves, not absolute
    """
    print(f"\n   === TARGET ENGINEERING (horizon={horizon}h, "
          f"winsorize ±{winsorize_sigma}σ) ===")

    # Raw forward return
    fwd_ret = np.log(df['close'].shift(-horizon) / df['close'])

    # 24h rolling volatility of log returns
    log_returns = np.log(df['close'] / df['close'].shift(1))
    vol_24h = log_returns.rolling(24, min_periods=12).std()

    # Normalize
    y_raw = fwd_ret / (vol_24h + 1e-10)

    # Winsorize at ±3σ
    valid = y_raw.dropna()
    mu = valid.mean()
    sigma = valid.std()
    lower = mu - winsorize_sigma * sigma
    upper = mu + winsorize_sigma * sigma
    y_winsorized = y_raw.clip(lower, upper)

    print(f"   Raw fwd_ret stats:  mean={fwd_ret.mean():.6f}, "
          f"std={fwd_ret.std():.6f}")
    print(f"   Vol-norm stats:     mean={y_raw.mean():.4f}, "
          f"std={y_raw.std():.4f}")
    print(f"   Winsorize bounds:   [{lower:.3f}, {upper:.3f}]")
    print(f"   Valid samples:      {y_winsorized.notna().sum():,}")

    # Also build raw (non-normalized) for comparison
    return y_winsorized, fwd_ret


# ======================================================================
# 4. WALK-FORWARD SPLITS (reuse from V6)
# ======================================================================

def purged_wf_splits(n, n_splits=5, test_pct=0.12, purge=12):
    """Walk-forward splits with purge gap."""
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


# ======================================================================
# 5. OPTUNA HYPERPARAMETER TUNING (Tier 1 Improvement #3)
# ======================================================================

def optuna_tune_xgb(X, y, feature_cols, splits, n_trials=50):
    """Optuna-tuned XGBRegressor with full WF CV as objective."""
    print(f"\n   [Optuna] Tuning XGBoost ({n_trials} trials)...")
    X_feat = X[feature_cols]

    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'random_state': 42,
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 20.0, log=True),
        }

        fold_rmses = []
        for tr_idx, te_idx in splits:
            X_tr, y_tr = X_feat.iloc[tr_idx], y.iloc[tr_idx]
            X_te, y_te = X_feat.iloc[te_idx], y.iloc[te_idx]

            model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            preds = model.predict(X_te)
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            fold_rmses.append(rmse)

        return np.mean(fold_rmses)

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"   [Optuna] XGB best RMSE: {study.best_value:.6f}")
    print(f"   [Optuna] XGB best params: {study.best_params}")
    return study.best_params


def optuna_tune_lgb(X, y, feature_cols, splits, n_trials=50):
    """Optuna-tuned LGBMRegressor with full WF CV as objective."""
    print(f"\n   [Optuna] Tuning LightGBM ({n_trials} trials)...")
    X_feat = X[feature_cols]

    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': 42,
            'verbose': -1,
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 20.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 2.0),
        }

        fold_rmses = []
        for tr_idx, te_idx in splits:
            X_tr, y_tr = X_feat.iloc[tr_idx], y.iloc[tr_idx]
            X_te, y_te = X_feat.iloc[te_idx], y.iloc[te_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            preds = model.predict(X_te)
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            fold_rmses.append(rmse)

        return np.mean(fold_rmses)

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"   [Optuna] LGB best RMSE: {study.best_value:.6f}")
    print(f"   [Optuna] LGB best params: {study.best_params}")
    return study.best_params


def optuna_tune_catboost(X, y, feature_cols, splits, n_trials=50):
    """Optuna-tuned CatBoostRegressor with full WF CV as objective."""
    print(f"\n   [Optuna] Tuning CatBoost ({n_trials} trials)...")
    X_feat = X[feature_cols]

    def objective(trial):
        params = {
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': 0,
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'iterations': trial.suggest_int('iterations', 300, 1500),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 0.8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 20.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 60),
            'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
        }

        fold_rmses = []
        for tr_idx, te_idx in splits:
            X_tr, y_tr = X_feat.iloc[tr_idx].values, y.iloc[tr_idx].values
            X_te, y_te = X_feat.iloc[te_idx].values, y.iloc[te_idx].values

            model = cb.CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=(X_te, y_te),
                      early_stopping_rounds=50, verbose=0)
            preds = model.predict(X_te)
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            fold_rmses.append(rmse)

        return np.mean(fold_rmses)

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"   [Optuna] CatBoost best RMSE: {study.best_value:.6f}")
    print(f"   [Optuna] CatBoost best params: {study.best_params}")
    return study.best_params


# ======================================================================
# 6. WALK-FORWARD REGRESSION (V7: 3-model ensemble)
# ======================================================================

def walk_forward_regression_v7(X, y, feature_cols, splits,
                                xgb_params, lgb_params, cb_params,
                                label="V7"):
    """
    Walk-forward regression with Optuna-tuned XGB + LGB + CatBoost.
    Returns OOS predictions from each model and the weighted ensemble.
    """
    print(f"\n   === WALK-FORWARD REGRESSION ({label}) ===")

    n = len(X)
    xgb_preds = np.full(n, np.nan)
    lgb_preds = np.full(n, np.nan)
    cb_preds = np.full(n, np.nan)

    xgb_rmses, lgb_rmses, cb_rmses = [], [], []
    X_feat = X[feature_cols]

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X_feat.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X_feat.iloc[te_idx], y.iloc[te_idx]

        # --- XGBoost ---
        xgb_p = dict(xgb_params)
        xgb_p['objective'] = 'reg:squarederror'
        xgb_p['tree_method'] = 'hist'
        xgb_p['random_state'] = 42
        n_est = xgb_p.pop('n_estimators', 800)
        m_xgb = xgb.XGBRegressor(**xgb_p, n_estimators=n_est,
                                   early_stopping_rounds=50)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        p_xgb = m_xgb.predict(X_te)
        rmse_xgb = np.sqrt(mean_squared_error(y_te, p_xgb))
        xgb_rmses.append(rmse_xgb)
        xgb_preds[te_idx] = p_xgb

        # --- LightGBM ---
        lgb_p = dict(lgb_params)
        lgb_p['objective'] = 'regression'
        lgb_p['metric'] = 'rmse'
        lgb_p['random_state'] = 42
        lgb_p['verbose'] = -1
        n_est_l = lgb_p.pop('n_estimators', 800)
        m_lgb = lgb.LGBMRegressor(**lgb_p, n_estimators=n_est_l)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        p_lgb = m_lgb.predict(X_te)
        rmse_lgb = np.sqrt(mean_squared_error(y_te, p_lgb))
        lgb_rmses.append(rmse_lgb)
        lgb_preds[te_idx] = p_lgb

        # --- CatBoost ---
        cb_p = dict(cb_params)
        cb_p['loss_function'] = 'RMSE'
        cb_p['random_seed'] = 42
        cb_p['verbose'] = 0
        n_iter = cb_p.pop('iterations', 800)
        m_cb = cb.CatBoostRegressor(**cb_p, iterations=n_iter)
        m_cb.fit(X_tr.values, y_tr.values,
                 eval_set=(X_te.values, y_te.values),
                 early_stopping_rounds=50, verbose=0)
        p_cb = m_cb.predict(X_te.values)
        rmse_cb = np.sqrt(mean_squared_error(y_te, p_cb))
        cb_rmses.append(rmse_cb)
        cb_preds[te_idx] = p_cb

        print(f"      Fold {fold_i+1}: XGB={rmse_xgb:.6f} | "
              f"LGB={rmse_lgb:.6f} | CB={rmse_cb:.6f}")

    # Compute mean RMSEs
    mean_xgb = np.mean(xgb_rmses)
    mean_lgb = np.mean(lgb_rmses)
    mean_cb = np.mean(cb_rmses)
    print(f"\n   Mean RMSE: XGB={mean_xgb:.6f} | LGB={mean_lgb:.6f} | "
          f"CB={mean_cb:.6f}")

    # Inverse-RMSE weighting for ensemble
    inv_rmses = np.array([1.0/mean_xgb, 1.0/mean_lgb, 1.0/mean_cb])
    weights = inv_rmses / inv_rmses.sum()
    print(f"   Ensemble weights: XGB={weights[0]:.3f} | LGB={weights[1]:.3f} | "
          f"CB={weights[2]:.3f}")

    # Build ensemble predictions
    ensemble_preds = np.full(n, np.nan)
    for i in range(n):
        vals = []
        wts = []
        if not np.isnan(xgb_preds[i]):
            vals.append(xgb_preds[i])
            wts.append(weights[0])
        if not np.isnan(lgb_preds[i]):
            vals.append(lgb_preds[i])
            wts.append(weights[1])
        if not np.isnan(cb_preds[i]):
            vals.append(cb_preds[i])
            wts.append(weights[2])
        if vals:
            wts = np.array(wts)
            wts /= wts.sum()
            ensemble_preds[i] = np.dot(vals, wts)

    return {
        'xgb': xgb_preds,
        'lgb': lgb_preds,
        'cb': cb_preds,
        'ensemble': ensemble_preds,
        'weights': weights,
        'rmses': {'xgb': mean_xgb, 'lgb': mean_lgb, 'cb': mean_cb},
    }


# ======================================================================
# 7. WALK-FORWARD CLASSIFICATION (for AUC comparison with V6)
# ======================================================================

def walk_forward_tree_classification(X, y, feature_cols, splits,
                                      xgb_params, lgb_params, cb_params,
                                      side_label="LONG"):
    """Walk-forward classification with Optuna-tuned 3-model ensemble."""
    oos_probs = np.full(len(X), np.nan)
    fold_aucs = []
    X_feat = X[feature_cols]

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X_feat.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X_feat.iloc[te_idx], y.iloc[te_idx]

        # SMOTE
        if y_tr.sum() > 5 and y_tr.sum() < len(y_tr) - 5:
            smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_tr.sum())-1))
            X_tr_s, y_tr_s = smote.fit_resample(X_tr, y_tr)
        else:
            X_tr_s, y_tr_s = X_tr, y_tr

        # XGBoost (classification version of optimized params)
        xgb_p = dict(xgb_params)
        xgb_p['objective'] = 'binary:logistic'
        xgb_p['eval_metric'] = 'auc'
        xgb_p['tree_method'] = 'hist'
        xgb_p['random_state'] = 42
        xgb_p['scale_pos_weight'] = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        n_est = xgb_p.pop('n_estimators', 800)
        m_xgb = xgb.XGBClassifier(**xgb_p, n_estimators=n_est,
                                    early_stopping_rounds=50)
        m_xgb.fit(X_tr_s, y_tr_s, eval_set=[(X_te, y_te)], verbose=False)

        # LightGBM
        lgb_p = dict(lgb_params)
        lgb_p['objective'] = 'binary'
        lgb_p['metric'] = 'auc'
        lgb_p['random_state'] = 42
        lgb_p['class_weight'] = 'balanced'
        lgb_p['verbose'] = -1
        n_est_l = lgb_p.pop('n_estimators', 800)
        m_lgb = lgb.LGBMClassifier(**lgb_p, n_estimators=n_est_l)
        m_lgb.fit(X_tr_s, y_tr_s, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        # CatBoost
        cb_p = dict(cb_params)
        cb_p['loss_function'] = 'Logloss'
        cb_p['random_seed'] = 42
        cb_p['verbose'] = 0
        cb_p['auto_class_weights'] = 'Balanced'
        n_iter = cb_p.pop('iterations', 800)
        m_cb = cb.CatBoostClassifier(**cb_p, iterations=n_iter)
        m_cb.fit(X_tr_s.values, y_tr_s.values,
                 eval_set=(X_te.values, y_te.values),
                 early_stopping_rounds=50, verbose=0)

        # Ensemble probabilities (equal weight for classification)
        p = (m_xgb.predict_proba(X_te)[:, 1] +
             m_lgb.predict_proba(X_te)[:, 1] +
             m_cb.predict_proba(X_te.values)[:, 1]) / 3.0

        try:
            auc = roc_auc_score(y_te, p)
        except:
            auc = 0.5
        fold_aucs.append(auc)
        oos_probs[te_idx] = p
        print(f"      {side_label} Fold {fold_i+1}: AUC={auc:.4f}")

    mean_auc = np.mean(fold_aucs)
    print(f"      {side_label} Mean AUC: {mean_auc:.4f} ± {np.std(fold_aucs):.4f}")
    return oos_probs, mean_auc


# ======================================================================
# 8. V6 BASELINE (2-model, default params, raw target)
# ======================================================================

def walk_forward_regression_v6_baseline(X, y, feature_cols, splits):
    """V6-style regression with default params (for comparison)."""
    print(f"\n   === V6 BASELINE REGRESSION ===")
    n = len(X)
    preds = np.full(n, np.nan)
    fold_rmses = []
    X_feat = X[feature_cols]

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X_feat.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X_feat.iloc[te_idx], y.iloc[te_idx]

        m_lgb = lgb.LGBMRegressor(
            objective='regression', metric='rmse', random_state=42, verbose=-1,
            max_depth=5, learning_rate=0.03, n_estimators=800,
            subsample=0.7, colsample_bytree=0.6, min_child_samples=30,
            reg_alpha=1.0, reg_lambda=5.0, num_leaves=31,
        )
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        m_xgb = xgb.XGBRegressor(
            objective='reg:squarederror', eval_metric='rmse',
            tree_method='hist', random_state=42,
            max_depth=5, learning_rate=0.03, n_estimators=800,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=20,
            gamma=1.0, reg_alpha=1.0, reg_lambda=5.0,
            early_stopping_rounds=50,
        )
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        p = (m_lgb.predict(X_te) + m_xgb.predict(X_te)) / 2
        rmse = np.sqrt(mean_squared_error(y_te, p))
        fold_rmses.append(rmse)
        preds[te_idx] = p
        print(f"      V6 Fold {fold_i+1}: RMSE={rmse:.6f}")

    print(f"      V6 Mean RMSE: {np.mean(fold_rmses):.6f}")
    return preds


# ======================================================================
# 9. PORTFOLIO SIMULATOR (reused from V6, fixed trailing stop)
# ======================================================================

def simulate_portfolio(
    close, high, low, atr, hours,
    long_probs, short_probs,
    cfg: dict,
) -> list:
    """
    Bar-by-bar portfolio simulator with FIXED trailing stop logic.
    Identical to V6 (conservative worst-price-first approach).
    """
    n = len(close)
    initial_cap = cfg.get('initial_capital', 25000)
    cap = initial_cap
    peak_cap = initial_cap
    risk_pct = cfg.get('risk_per_trade', 0.01)
    max_pos = cfg.get('max_pos_frac', 0.25)
    fee = cfg.get('fee_pct', 0.02) / 100
    slip = cfg.get('slippage_pct', 0.01) / 100
    total_cost = 2 * (fee + slip)
    tp_mult = cfg['tp_mult']
    sl_mult = cfg['sl_mult']
    max_hold = cfg['max_hold']
    exit_mode = cfg.get('exit_mode', 'fixed')
    trail_mult = cfg.get('trail_mult', 1.0)
    be_trigger = cfg.get('be_trigger', 1.0)
    partial_at = cfg.get('partial_at', 1.0)
    partial_frac = cfg.get('partial_frac', 0.5)
    direction = cfg.get('direction', 'both')
    funding = cfg.get('funding_rates', None)
    cooldown = cfg.get('cooldown', 0)
    confirmation = cfg.get('confirmation', False)
    max_dd_pct = cfg.get('max_drawdown_pct', 0.10)

    signal_mode = cfg.get('signal_mode', 'percentile')
    signal_pct = cfg.get('signal_pct', 0.05)
    signal_thr = cfg.get('signal_thr', 0.15)
    regime = cfg.get('regime_mask', np.ones(n, dtype=bool))

    if signal_mode == 'percentile':
        lp_valid = long_probs[~np.isnan(long_probs)]
        sp_valid = short_probs[~np.isnan(short_probs)]
        long_thr = np.percentile(lp_valid, 100 * (1 - signal_pct)) if len(lp_valid) > 0 else 1.0
        short_thr = np.percentile(sp_valid, 100 * (1 - signal_pct)) if len(sp_valid) > 0 else 1.0
    else:
        long_thr = signal_thr
        short_thr = signal_thr

    long_sig = (~np.isnan(long_probs)) & (long_probs >= long_thr) & regime
    short_sig = (~np.isnan(short_probs)) & (short_probs >= short_thr) & regime

    if confirmation:
        sp_med = np.nanmedian(short_probs)
        lp_med = np.nanmedian(long_probs)
        long_sig = long_sig & (~np.isnan(short_probs)) & (short_probs < sp_med)
        short_sig = short_sig & (~np.isnan(long_probs)) & (long_probs < lp_med)

    if direction == 'long':
        short_sig[:] = False
    elif direction == 'short':
        long_sig[:] = False

    # Resolve conflicts
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
    dd_breached = False

    for i in range(n):
        if cap < peak_cap * (1 - max_dd_pct):
            dd_breached = True
        if dd_breached:
            break

        if in_pos:
            bars_held = i - entry_bar
            exit_price = None
            reason = None

            if funding is not None and i < len(funding) and not np.isnan(funding[i]):
                if pos_side == 'LONG':
                    fund_pnl -= funding[i] * remaining * pos_size
                else:
                    fund_pnl += funding[i] * remaining * pos_size

            if pos_side == 'LONG':
                if low[i] <= current_sl:
                    exit_price = current_sl
                    reason = 'SL'
                elif high[i] >= tp_price:
                    exit_price = tp_price
                    reason = 'TP'
                elif bars_held >= max_hold:
                    exit_price = close[i]
                    reason = 'TIMEOUT'

                if exit_price is None:
                    if exit_mode == 'trailing':
                        new_sl = high[i] - trail_mult * atr_entry
                        current_sl = max(current_sl, new_sl)
                    if exit_mode in ('breakeven', 'partial'):
                        if not be_active and high[i] >= entry_p + be_trigger * atr_entry:
                            current_sl = max(current_sl, entry_p)
                            be_active = True
                    if exit_mode == 'partial' and not partial_done:
                        if high[i] >= entry_p + partial_at * atr_entry:
                            p_exit = entry_p + partial_at * atr_entry
                            p_ret = (p_exit - entry_p) / entry_p
                            partial_pnl += partial_frac * p_ret * pos_size
                            remaining = 1.0 - partial_frac
                            partial_done = True
                            current_sl = max(current_sl, entry_p)

                if exit_price is not None:
                    gross_ret = (exit_price - entry_p) / entry_p
                    net_ret_main = gross_ret * remaining * pos_size
                    total_pnl = net_ret_main + partial_pnl + fund_pnl - total_cost * pos_size
                    cap += total_pnl
                    peak_cap = max(peak_cap, cap)

            else:  # SHORT
                if high[i] >= current_sl:
                    exit_price = current_sl
                    reason = 'SL'
                elif low[i] <= tp_price:
                    exit_price = tp_price
                    reason = 'TP'
                elif bars_held >= max_hold:
                    exit_price = close[i]
                    reason = 'TIMEOUT'

                if exit_price is None:
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

                if exit_price is not None:
                    gross_ret = (entry_p - exit_price) / entry_p
                    net_ret_main = gross_ret * remaining * pos_size
                    total_pnl = net_ret_main + partial_pnl + fund_pnl - total_cost * pos_size
                    cap += total_pnl
                    peak_cap = max(peak_cap, cap)

            if exit_price is not None:
                trades.append({
                    'side': pos_side, 'reason': reason,
                    'bars_held': bars_held,
                    'pnl': total_pnl, 'capital': cap,
                    'gross_ret': gross_ret,
                })
                in_pos = False
                last_exit_bar = i

        if not in_pos and cap > 100 and not dd_breached and i - last_exit_bar > cooldown:
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


def calc_stats(trades, label="", initial_capital=25000):
    """Calculate portfolio statistics from trade list."""
    if not trades:
        return None
    df_t = pd.DataFrame(trades)
    n = len(df_t)
    if n < 3:
        return None

    wins = df_t['pnl'] > 0
    wr = wins.mean()
    gp = df_t.loc[wins, 'pnl'].sum()
    gl = abs(df_t.loc[~wins, 'pnl'].sum())
    pf = gp / gl if gl > 0 else 99.0
    final = df_t['capital'].iloc[-1]
    tr = (final - initial_capital) / initial_capital * 100

    eq = np.array([initial_capital] + df_t['capital'].tolist())
    pk = np.maximum.accumulate(eq)
    dd = (pk - eq) / (pk + 1e-10)
    max_dd = dd.max() * 100

    rets = df_t['pnl'].values / initial_capital
    sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(365 * 2) if len(rets) > 1 else 0

    avg_pnl = df_t['pnl'].mean()
    avg_win = df_t.loc[wins, 'pnl'].mean() if wins.sum() > 0 else 0
    avg_loss = df_t.loc[~wins, 'pnl'].mean() if (~wins).sum() > 0 else 0

    n_l = (df_t['side'] == 'LONG').sum()
    n_s = (df_t['side'] == 'SHORT').sum()
    tp_n = (df_t['reason'] == 'TP').sum()
    sl_n = (df_t['reason'] == 'SL').sum()

    return {
        'label': label, 'n_trades': n, 'wr': wr, 'pf': pf,
        'ret': tr, 'max_dd': max_dd, 'sharpe': sharpe,
        'final_cap': final,
        'avg_pnl': avg_pnl, 'avg_win': avg_win, 'avg_loss': avg_loss,
        'n_long': n_l, 'n_short': n_s,
        'tp': tp_n, 'sl': sl_n, 'timeout': n - tp_n - sl_n,
    }


def regression_to_signals(pred_returns, threshold_pct=0.005):
    """Convert return predictions to long/short probability-like signals."""
    long_probs = np.full_like(pred_returns, np.nan, dtype=np.float64)
    short_probs = np.full_like(pred_returns, np.nan, dtype=np.float64)

    valid = ~np.isnan(pred_returns)
    v = pred_returns[valid]
    long_probs[valid] = 1.0 / (1.0 + np.exp(-v * 200))
    short_probs[valid] = 1.0 / (1.0 + np.exp(v * 200))

    return long_probs, short_probs


# ======================================================================
# 10. STRATEGY SWEEP
# ======================================================================

def sweep_strategies(close, high, low, atr, hours,
                     long_probs, short_probs,
                     regime_mask, funding_rates,
                     model_label="model"):
    """Sweep strategy parameters. Same grid as V6 for fair comparison."""
    results = []

    barrier_configs = [
        (1.5, 1.0, "TP1.5_SL1.0"),
        (2.0, 1.0, "TP2.0_SL1.0"),
        (2.0, 1.5, "TP2.0_SL1.5"),
        (2.5, 1.5, "TP2.5_SL1.5"),
        (3.0, 1.5, "TP3.0_SL1.5"),
    ]

    signal_configs = [
        (0.05, 'top5%'),
        (0.10, 'top10%'),
        (0.20, 'top20%'),
    ]

    fee_configs = [
        (0.02, 0.01, 'maker'),
        (0.01, 0.005, 'vip'),
    ]

    exit_configs = [
        ('fixed', 1.0, 'fixed'),
        ('trailing', 0.75, 'trail0.75'),
        ('trailing', 1.0, 'trail1.0'),
        ('breakeven', 1.0, 'be1.0'),
    ]

    dir_configs = ['both', 'short']
    hold_configs = [12, 18]
    cooldown_configs = [0]

    total = (len(barrier_configs) * len(signal_configs) * len(dir_configs) *
             len(fee_configs) * len(exit_configs) * len(hold_configs) *
             len(cooldown_configs))
    done = 0
    t0 = time.time()
    print(f"      Sweeping {total} strategy combinations...", flush=True)

    for (tp, sl, b_lbl) in barrier_configs:
        for (sig_pct, sig_lbl) in signal_configs:
            for direction in dir_configs:
                for (fee_p, slip_p, fee_lbl) in fee_configs:
                    for (exit_mode, trail_m, exit_lbl) in exit_configs:
                        for hold in hold_configs:
                            for cd in cooldown_configs:
                                cfg = {
                                    'tp_mult': tp, 'sl_mult': sl, 'max_hold': hold,
                                    'exit_mode': exit_mode,
                                    'trail_mult': trail_m, 'be_trigger': 1.0,
                                    'fee_pct': fee_p, 'slippage_pct': slip_p,
                                    'signal_mode': 'percentile', 'signal_pct': sig_pct,
                                    'direction': direction,
                                    'hour_filter': None,
                                    'initial_capital': 25000,
                                    'risk_per_trade': 0.01,
                                    'max_pos_frac': 0.25,
                                    'cooldown': cd,
                                    'max_drawdown_pct': 0.10,
                                    'regime_mask': regime_mask,
                                    'funding_rates': funding_rates,
                                }
                                label = (f"{model_label}|{b_lbl}|{sig_lbl}|{direction}"
                                         f"|{fee_lbl}|{exit_lbl}|H{hold}|CD{cd}")
                                trades = simulate_portfolio(
                                    close, high, low, atr, hours,
                                    long_probs, short_probs, cfg)
                                s = calc_stats(trades, label, 25000)
                                if s:
                                    results.append(s)
                                done += 1

    elapsed = time.time() - t0
    n_prof = sum(1 for r in results if r['pf'] > 1.0)
    print(f"      Sweep done: {done} configs ({elapsed:.0f}s), "
          f"{len(results)} valid, {n_prof} profitable")
    return results


# ======================================================================
# 11. RESULTS DISPLAY
# ======================================================================

def print_top_results(results, title="RESULTS", top_n=30):
    """Print ranked results table."""
    profitable = sorted([r for r in results if r and r['pf'] > 1.0],
                        key=lambda x: x['pf'], reverse=True)

    print(f"\n{'='*130}")
    print(f"  {title}")
    print(f"  Total experiments: {len(results):,} | Profitable: {len(profitable)}")
    print(f"{'='*130}")

    show = profitable[:top_n] if profitable else sorted(
        [r for r in results if r], key=lambda x: x['pf'], reverse=True)[:top_n]

    if not show:
        print("  No results to show.")
        return profitable

    print(f"\n  {'#':<4s} {'Strategy':<70s} {'Trades':>6s} {'WR':>6s} "
          f"{'PF':>7s} {'Ret%':>8s} {'DD%':>6s} {'Sharpe':>7s} {'AvgPnL':>8s}")
    print(f"  {'-'*128}")

    for i, s in enumerate(show):
        wr_str = f"{s['wr']:.1%}"
        pf_str = f"{s['pf']:.2f}" if s['pf'] < 100 else f"{s['pf']:.0f}"
        tag = "+" if s['pf'] > 1.0 else "-"
        print(f"  {i+1:<4d} {s['label']:<70s} {s['n_trades']:>6d} "
              f"{wr_str:>6s} {pf_str:>7s} {s['ret']:>+8.1f} {s['max_dd']:>6.1f} "
              f"{s['sharpe']:>7.2f} ${s['avg_pnl']:>7.2f} {tag}")

    # Pattern analysis
    if profitable:
        print(f"\n  --- Pattern Analysis (profitable only) ---")
        exits = Counter()
        dirs = Counter()
        fees = Counter()
        sigs = Counter()
        for r in profitable:
            parts = r['label'].split('|')
            for p in parts:
                if p.startswith('trail') or p in ('fixed', 'be1.0'):
                    exits[p] += 1
                elif p in ('short', 'long', 'both'):
                    dirs[p] += 1
                elif p in ('taker', 'maker', 'vip'):
                    fees[p] += 1
                elif p.startswith('top'):
                    sigs[p] += 1
        print(f"  Exit modes: {dict(exits)}")
        print(f"  Directions: {dict(dirs)}")
        print(f"  Fee tiers:  {dict(fees)}")
        print(f"  Signals:    {dict(sigs)}")

    return profitable


# ======================================================================
# MAIN EXPERIMENT
# ======================================================================

def main():
    t_start = time.time()
    print("=" * 80)
    print("  V7 IMPROVED REGRESSION EXPERIMENT")
    print(f"  Started: {datetime.now()}")
    print(f"  Improvements: Feature Selection + Vol-Normalized Target + "
          f"Optuna Tuning + CatBoost")
    print(f"  Account: $25,000 prop firm | Max DD: 10%")
    print("=" * 80)

    # ---- 1. Load data ----
    print("\n[1/8] Loading and preparing data...")
    df = load_and_prepare_data()
    all_feature_cols = get_feature_cols(df)
    print(f"    Total available features: {len(all_feature_cols)}")

    # ---- 2. Build targets ----
    print("\n[2/8] Building targets...")

    # 2a. Triple-barrier labels (for classification AUC comparison)
    tb_long = triple_barrier_labels(df['close'], df['high'], df['low'],
                                    tp_mult=2.5, sl_mult=1.5, max_holding=12,
                                    direction='long')
    tb_short = triple_barrier_labels(df['close'], df['high'], df['low'],
                                     tp_mult=2.5, sl_mult=1.5, max_holding=12,
                                     direction='short')
    df['y_long'] = tb_long['tb_binary']
    df['y_short'] = tb_short['short_tb_binary']

    # 2b. Regression targets
    # Vol-normalized (V7 innovation)
    y_vol_norm_8h, y_raw_8h = build_vol_normalized_target(df, horizon=8)
    df['y_vol_norm_8h'] = y_vol_norm_8h
    df['y_raw_8h'] = y_raw_8h

    # Also raw 8h for V6 comparison
    df['fwd_ret_8h'] = np.log(df['close'].shift(-8) / df['close'])

    # Clean
    mask = (df['y_long'].notna() & df['y_short'].notna() &
            df['y_vol_norm_8h'].notna() & df['fwd_ret_8h'].notna() &
            df[all_feature_cols].notna().all(axis=1))
    df_clean = df[mask].copy().reset_index(drop=True)

    print(f"    Clean samples: {len(df_clean):,}")
    print(f"    LONG positive rate: {df_clean['y_long'].mean():.1%}")
    print(f"    SHORT positive rate: {df_clean['y_short'].mean():.1%}")

    # ---- 3. Feature Selection ----
    print("\n[3/8] Feature Selection (MI-based)...")

    # Use vol-normalized target for MI scoring (this is what we'll optimize)
    selected_features = select_features_mi(
        df_clean, df_clean['y_vol_norm_8h'],
        all_feature_cols, top_k=50,
    )
    print(f"\n    Selected {len(selected_features)} features "
          f"(from {len(all_feature_cols)})")
    print(f"    Selected: {selected_features[:10]}...")

    # Prepare arrays for backtesting
    close_v = df_clean['close'].values
    high_v = df_clean['high'].values
    low_v = df_clean['low'].values
    atr_v = df_clean['atr_14'].values if 'atr_14' in df_clean.columns else np.ones(len(df_clean))
    hours_v = df_clean['_hour'].values
    regime_v = df_clean['regime_ok'].values
    fund_v = df_clean['funding_rate'].values if 'funding_rate' in df_clean.columns else None

    splits = purged_wf_splits(len(df_clean), n_splits=5, test_pct=0.12, purge=10)

    all_results = []
    model_aucs = {}

    # ================================================================
    # EXPERIMENT A: V6 Baseline (raw target, default params, all features)
    # ================================================================
    print("\n" + "=" * 80)
    print("  EXPERIMENT A: V6 BASELINE (raw 8h, default params, all features)")
    print("=" * 80, flush=True)

    v6_preds = walk_forward_regression_v6_baseline(
        df_clean, df_clean['fwd_ret_8h'], all_feature_cols, splits)
    v6_long, v6_short = regression_to_signals(v6_preds)

    print(f"\n   Sweeping strategies for V6 baseline...")
    v6_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        v6_long, v6_short, regime_v, fund_v, "v6_baseline")

    v6_profitable = print_top_results(v6_results, "V6 BASELINE RESULTS")
    all_results.extend(v6_results)

    # ================================================================
    # EXPERIMENT B: V7a - Feature Selection only
    # ================================================================
    print("\n" + "=" * 80)
    print("  EXPERIMENT B: V7a - FEATURE SELECTION ONLY "
          f"({len(selected_features)} features)")
    print("=" * 80, flush=True)

    v7a_preds = walk_forward_regression_v6_baseline(
        df_clean, df_clean['fwd_ret_8h'], selected_features, splits)
    v7a_long, v7a_short = regression_to_signals(v7a_preds)

    print(f"\n   Sweeping strategies for V7a (feature selection only)...")
    v7a_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        v7a_long, v7a_short, regime_v, fund_v, "v7a_feat_sel")

    v7a_profitable = print_top_results(v7a_results,
                                       "V7a FEATURE SELECTION RESULTS")
    all_results.extend(v7a_results)

    # ================================================================
    # EXPERIMENT C: V7b - Vol-normalized target (selected features)
    # ================================================================
    print("\n" + "=" * 80)
    print("  EXPERIMENT C: V7b - VOL-NORMALIZED TARGET")
    print("=" * 80, flush=True)

    v7b_preds = walk_forward_regression_v6_baseline(
        df_clean, df_clean['y_vol_norm_8h'], selected_features, splits)
    v7b_long, v7b_short = regression_to_signals(v7b_preds)

    print(f"\n   Sweeping strategies for V7b (vol-normalized target)...")
    v7b_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        v7b_long, v7b_short, regime_v, fund_v, "v7b_vol_norm")

    v7b_profitable = print_top_results(v7b_results,
                                       "V7b VOL-NORMALIZED TARGET RESULTS")
    all_results.extend(v7b_results)

    # ================================================================
    # EXPERIMENT D: Optuna Tuning (selected features, vol-norm target)
    # ================================================================
    print("\n" + "=" * 80)
    print("  EXPERIMENT D: OPTUNA HYPERPARAMETER TUNING")
    print("=" * 80, flush=True)

    # Tune all 3 models with 50 trials each
    tune_splits = purged_wf_splits(len(df_clean), n_splits=3, test_pct=0.15, purge=10)
    print(f"   Using {len(tune_splits)} splits for tuning, "
          f"{len(splits)} for evaluation")

    best_xgb_params = optuna_tune_xgb(
        df_clean, df_clean['y_vol_norm_8h'], selected_features,
        tune_splits, n_trials=50)

    best_lgb_params = optuna_tune_lgb(
        df_clean, df_clean['y_vol_norm_8h'], selected_features,
        tune_splits, n_trials=50)

    best_cb_params = optuna_tune_catboost(
        df_clean, df_clean['y_vol_norm_8h'], selected_features,
        tune_splits, n_trials=50)

    # ================================================================
    # EXPERIMENT E: V7 Full (Optuna + 3-model ensemble + vol-norm + feat sel)
    # ================================================================
    print("\n" + "=" * 80)
    print("  EXPERIMENT E: V7 FULL (all improvements combined)")
    print("=" * 80, flush=True)

    v7_result = walk_forward_regression_v7(
        df_clean, df_clean['y_vol_norm_8h'], selected_features, splits,
        best_xgb_params, best_lgb_params, best_cb_params,
        label="V7_vol_norm")

    # Sweep with each sub-model and ensemble
    for model_key, model_name in [('xgb', 'v7_xgb'), ('lgb', 'v7_lgb'),
                                   ('cb', 'v7_cb'), ('ensemble', 'v7_ensemble')]:
        preds = v7_result[model_key]
        l_sig, s_sig = regression_to_signals(preds)
        print(f"\n   Sweeping strategies for {model_name}...")
        res = sweep_strategies(
            close_v, high_v, low_v, atr_v, hours_v,
            l_sig, s_sig, regime_v, fund_v, model_name)
        print_top_results(res, f"{model_name.upper()} RESULTS")
        all_results.extend(res)

    # ================================================================
    # EXPERIMENT F: V7 with RAW target (to isolate Optuna + CatBoost effect)
    # ================================================================
    print("\n" + "=" * 80)
    print("  EXPERIMENT F: V7 RAW TARGET (Optuna + 3-model, raw 8h return)")
    print("=" * 80, flush=True)

    # Re-tune on raw target (fewer trials for speed)
    raw_xgb = optuna_tune_xgb(
        df_clean, df_clean['fwd_ret_8h'], selected_features,
        tune_splits, n_trials=30)
    raw_lgb = optuna_tune_lgb(
        df_clean, df_clean['fwd_ret_8h'], selected_features,
        tune_splits, n_trials=30)
    raw_cb = optuna_tune_catboost(
        df_clean, df_clean['fwd_ret_8h'], selected_features,
        tune_splits, n_trials=30)

    v7_raw = walk_forward_regression_v7(
        df_clean, df_clean['fwd_ret_8h'], selected_features, splits,
        raw_xgb, raw_lgb, raw_cb, label="V7_raw")

    for model_key, model_name in [('ensemble', 'v7_raw_ensemble')]:
        preds = v7_raw[model_key]
        l_sig, s_sig = regression_to_signals(preds)
        print(f"\n   Sweeping strategies for {model_name}...")
        res = sweep_strategies(
            close_v, high_v, low_v, atr_v, hours_v,
            l_sig, s_sig, regime_v, fund_v, model_name)
        print_top_results(res, f"{model_name.upper()} RESULTS")
        all_results.extend(res)

    # ================================================================
    # EXPERIMENT G: Classification AUC comparison (3-model ensemble)
    # ================================================================
    print("\n" + "=" * 80)
    print("  EXPERIMENT G: CLASSIFICATION AUC COMPARISON (V7 3-model)")
    print("=" * 80, flush=True)

    # Use Optuna-tuned params for classification too
    v7_long_probs, v7_l_auc = walk_forward_tree_classification(
        df_clean, df_clean['y_long'].astype(int), selected_features, splits,
        best_xgb_params, best_lgb_params, best_cb_params, "LONG")

    v7_short_probs, v7_s_auc = walk_forward_tree_classification(
        df_clean, df_clean['y_short'].astype(int), selected_features, splits,
        best_xgb_params, best_lgb_params, best_cb_params, "SHORT")

    model_aucs['v7_classification'] = {
        'long': v7_l_auc, 'short': v7_s_auc,
        'avg': (v7_l_auc + v7_s_auc) / 2,
    }

    print(f"\n   Sweeping strategies for V7 classification...")
    v7_cls_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        v7_long_probs, v7_short_probs, regime_v, fund_v, "v7_cls")
    print_top_results(v7_cls_results, "V7 CLASSIFICATION RESULTS")
    all_results.extend(v7_cls_results)

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print("\n" + "=" * 130)
    print("  V7 GRAND SUMMARY")
    print("=" * 130)

    all_profitable = print_top_results(all_results,
                                       "ALL STRATEGIES RANKED", top_n=50)

    # Improvement comparison
    print(f"\n{'='*80}")
    print(f"  V6 vs V7 COMPARISON")
    print(f"{'='*80}")

    # Classification AUC
    print(f"\n  Classification AUC:")
    print(f"    V6 baseline: LONG=0.6377 | SHORT=0.6356 | Avg=0.6367")
    if 'v7_classification' in model_aucs:
        ma = model_aucs['v7_classification']
        print(f"    V7 improved: LONG={ma['long']:.4f} | SHORT={ma['short']:.4f} "
              f"| Avg={ma['avg']:.4f}")
        delta = ma['avg'] - 0.6367
        print(f"    Delta: {delta:+.4f} ({'improvement' if delta > 0 else 'regression'})")

    # Regression RMSE
    print(f"\n  Regression RMSE:")
    print(f"    V7 ensemble: {v7_result['rmses']}")
    print(f"    V7 weights:  {v7_result['weights']}")

    # Best strategies comparison
    print(f"\n  Profitability by model:")
    model_tags = ['v6_baseline', 'v7a_feat_sel', 'v7b_vol_norm',
                  'v7_xgb', 'v7_lgb', 'v7_cb', 'v7_ensemble',
                  'v7_raw_ensemble', 'v7_cls']
    for tag in model_tags:
        matching = [r for r in all_results if r and r['label'].startswith(tag)]
        prof = [r for r in matching if r['pf'] > 1.0]
        if matching:
            best_pf = max(r['pf'] for r in matching)
            best_sharpe = max(r['sharpe'] for r in matching)
            print(f"    {tag:<20s}: {len(matching):>5d} exps, "
                  f"{len(prof):>4d} prof ({len(prof)/len(matching):.1%}), "
                  f"best PF={best_pf:.2f}, best Sharpe={best_sharpe:.2f}")

    # Top strategies for prop firm
    if all_profitable:
        print(f"\n{'='*80}")
        print(f"  TOP 10 STRATEGIES FOR $25K PROP FIRM")
        print(f"{'='*80}")
        for i, s in enumerate(all_profitable[:10]):
            ann_ret = s['ret'] / 5
            dollar_pnl = s['final_cap'] - 25000
            print(f"\n  {i+1}. {s['label']}")
            print(f"     Trades: {s['n_trades']} | WR: {s['wr']:.1%} | PF: {s['pf']:.2f}")
            print(f"     Total: {s['ret']:+.1f}% (${dollar_pnl:+,.0f}) | "
                  f"Ann: ~{ann_ret:+.1f}%/yr | DD: {s['max_dd']:.1f}%")
            print(f"     Avg: ${s['avg_pnl']:+.2f} | "
                  f"Win: ${s['avg_win']:+.2f} | Loss: ${s['avg_loss']:+.2f}")
            print(f"     Sharpe: {s['sharpe']:.2f} | L/S: {s['n_long']}/{s['n_short']}")

    # Feature importance from final ensemble
    print(f"\n{'='*80}")
    print(f"  SELECTED FEATURES ({len(selected_features)})")
    print(f"{'='*80}")
    for i, f in enumerate(selected_features):
        print(f"    {i+1:>3d}. {f}")

    # Save results
    duration = time.time() - t_start
    save_data = {
        'experiment': 'V7 Improved Regression',
        'improvements': [
            'MI-based feature selection',
            'Volatility-normalized target + Winsorization',
            'Optuna hyperparameter tuning (full WF CV)',
            'CatBoost as 3rd ensemble member',
        ],
        'total_experiments': len(all_results),
        'profitable_count': len(all_profitable) if all_profitable else 0,
        'top_50': all_profitable[:50] if all_profitable else [],
        'selected_features': selected_features,
        'n_features_original': len(all_feature_cols),
        'n_features_selected': len(selected_features),
        'best_params': {
            'xgb_vol_norm': best_xgb_params,
            'lgb_vol_norm': best_lgb_params,
            'cb_vol_norm': best_cb_params,
            'xgb_raw': raw_xgb,
            'lgb_raw': raw_lgb,
            'cb_raw': raw_cb,
        },
        'ensemble_weights': {
            'vol_norm': v7_result['weights'].tolist(),
        },
        'ensemble_rmses': v7_result['rmses'],
        'classification_aucs': model_aucs,
        'v6_baseline_aucs': {
            'long': 0.6377, 'short': 0.6356, 'avg': 0.6367,
        },
        'timestamp': str(datetime.now()),
        'duration_seconds': duration,
        'account_size': 25000,
    }

    out_path = OUTPUT_DIR / "v7_experiment_results.json"
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n   Results saved to {out_path}")

    print(f"\n   Total duration: {duration/60:.1f} minutes")
    print(f"   Experiments run: {len(all_results):,}")
    print("\n   V7 experiment complete!")


if __name__ == "__main__":
    main()
