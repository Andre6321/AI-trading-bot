"""
V8 Advanced Regression Experiment
===================================
Building on V7's finding that vol-normalized target was the key win (PF 1.46→1.58),
this experiment applies five Tier-1 research-backed improvements:

  1. HUBER LOSS (instead of RMSE)
     - Quadratic for small errors, linear for large
     - Down-weights unpredictable tail moves in crypto
     - Focuses model on the "predictable middle" of returns

  2. CATBOOST RMSEWithUncertainty (heteroscedastic regression)
     - Jointly predicts mean return AND log-variance
     - Provides free model confidence estimate
     - Trade filter: only enter when signal/noise exceeds threshold

  3. MULTI-QUANTILE REGRESSION (signal confidence filter)
     - Train CatBoost with MultiQuantile loss (α = 0.1, 0.25, 0.5, 0.75, 0.9)
     - Only SHORT when 75th percentile prediction is still negative
     - More principled than raw prediction thresholding

  4. EXPONENTIAL DECAY SAMPLE WEIGHTING
     - Recent data weighted more heavily: w_i = exp(-λ * (t_now - t_i))
     - Crypto market microstructure evolves rapidly
     - λ tuned alongside model hyperparameters

  5. ADVERSARIAL VALIDATION SAMPLE WEIGHTING
     - Train classifier to distinguish old vs recent data
     - Upweight training samples that "look like" recent data
     - Adapts to concept drift before performance degrades

V7 Results (our baseline):
  - Best: v7b_vol_norm  PF=1.58, Sharpe=4.08, 231 trades (SHORT-only)
  - Vol-normalized target = key win
  - Feature selection alone HURT (PF 1.46→1.24)
  - CatBoost 3rd ensemble = no value (weights 0.333/0.333/0.333)

V8 Strategy:
  - Keep vol-normalized target (proven winner)
  - Keep ALL 117 features (feature selection hurt in V7)
  - Use V7's Optuna-tuned params as starting point
  - Test each improvement independently AND combined

Usage:
    python scripts/v8_advanced_regression.py
"""
import os, sys, json, warnings, time, math, signal
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings('ignore')

# Force unbuffered stdout so Tee-Object / redirect can see progress
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Ignore SIGINT (Ctrl+C) to prevent stale KeyboardInterrupt from PowerShell
# Python 3.14 + Windows propagates ^C across terminals via ctypes callbacks
signal.signal(signal.SIGINT, signal.SIG_IGN)

from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features.alpha_features import build_alpha_features
from features.triple_barrier import triple_barrier_labels

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"


# ======================================================================
# 1. DATA LOADING (reuse from V7)
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
    return [c for c in df.columns
            if c not in exclude
            and not any(p in c.lower() for p in ['future_', 'target_', 'label_', 'fwd_ret_'])
            and not c.startswith('y_')
            and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]


# ======================================================================
# 2. TARGET ENGINEERING (from V7 - proven winner)
# ======================================================================

def build_vol_normalized_target(df, horizon=8, winsorize_sigma=3.0):
    """
    Build volatility-normalized forward return target.
    y = ln(C_{t+h} / C_t) / σ_24h, winsorized at ±3σ
    """
    print(f"\n   === TARGET: vol-normalized (horizon={horizon}h, "
          f"winsorize ±{winsorize_sigma}σ) ===")

    fwd_ret = np.log(df['close'].shift(-horizon) / df['close'])
    log_returns = np.log(df['close'] / df['close'].shift(1))
    vol_24h = log_returns.rolling(24, min_periods=12).std()

    y_raw = fwd_ret / (vol_24h + 1e-10)

    valid = y_raw.dropna()
    mu = valid.mean()
    sigma = valid.std()
    lower = mu - winsorize_sigma * sigma
    upper = mu + winsorize_sigma * sigma
    y_winsorized = y_raw.clip(lower, upper)

    print(f"   Vol-norm bounds: [{lower:.3f}, {upper:.3f}]")
    print(f"   Vol-norm std: {y_raw.std():.4f}")

    return y_winsorized, fwd_ret


# ======================================================================
# 3. WALK-FORWARD SPLITS
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
# 4. V8 IMPROVEMENT #1: HUBER LOSS
# ======================================================================

def optuna_tune_huber(X, y, feature_cols, splits, n_trials=50,
                      sample_weights=None):
    """
    Optuna-tune XGB + LGB + CatBoost with HUBER loss.
    Also tunes the Huber delta parameter.
    """
    print(f"\n   [Optuna-Huber] Tuning 3 models ({n_trials} trials each)...")
    X_feat = X[feature_cols]

    # --- XGBoost with Pseudo-Huber ---
    def xgb_objective(trial):
        huber_slope = trial.suggest_float('huber_slope', 0.5, 5.0)
        params = {
            'objective': 'reg:pseudohubererror',
            'huber_slope': huber_slope,
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
            sw = sample_weights[tr_idx] if sample_weights is not None else None
            n_est = params.pop('n_estimators')
            model = xgb.XGBRegressor(**params, n_estimators=n_est,
                                      early_stopping_rounds=50)
            params['n_estimators'] = n_est
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False,
                      sample_weight=sw)
            preds = model.predict(X_te)
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            fold_rmses.append(rmse)
        return np.mean(fold_rmses)

    study_xgb = optuna.create_study(direction='minimize',
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=False,
                       catch=(BaseException,))
    try:
        xgb_params = study_xgb.best_params
        print(f"   [Optuna-Huber] XGB best RMSE: {study_xgb.best_value:.6f}, "
              f"huber_slope={xgb_params.get('huber_slope', 'N/A')}")
    except ValueError:
        print("   [Optuna-Huber] XGB: ALL trials failed, using defaults")
        xgb_params = {'huber_slope': 2.0, 'max_depth': 5, 'learning_rate': 0.03,
                      'n_estimators': 800, 'subsample': 0.7, 'colsample_bytree': 0.6,
                      'reg_alpha': 0.5, 'reg_lambda': 1.0, 'min_child_weight': 10}

    # --- LightGBM with Huber ---
    def lgb_objective(trial):
        alpha = trial.suggest_float('alpha', 0.5, 5.0)
        params = {
            'objective': 'huber',
            'alpha': alpha,
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
            sw = sample_weights[tr_idx] if sample_weights is not None else None
            n_est = params.pop('n_estimators')
            model = lgb.LGBMRegressor(**params, n_estimators=n_est)
            params['n_estimators'] = n_est
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                      callbacks=[lgb.early_stopping(50, verbose=False)],
                      sample_weight=sw)
            preds = model.predict(X_te)
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            fold_rmses.append(rmse)
        return np.mean(fold_rmses)

    study_lgb = optuna.create_study(direction='minimize',
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb.optimize(lgb_objective, n_trials=n_trials, show_progress_bar=False,
                       catch=(BaseException,))
    try:
        lgb_params = study_lgb.best_params
        print(f"   [Optuna-Huber] LGB best RMSE: {study_lgb.best_value:.6f}, "
              f"alpha={lgb_params.get('alpha', 'N/A')}")
    except ValueError:
        print("   [Optuna-Huber] LGB: ALL trials failed, using defaults")
        lgb_params = {'alpha': 2.0, 'num_leaves': 63, 'learning_rate': 0.03,
                      'n_estimators': 800, 'subsample': 0.7, 'colsample_bytree': 0.6,
                      'reg_alpha': 0.5, 'reg_lambda': 1.0, 'min_child_samples': 20}

    # --- CatBoost with Huber ---
    def cb_objective(trial):
        delta = trial.suggest_float('delta', 0.5, 5.0)
        params = {
            'loss_function': f'Huber:delta={delta}',
            'random_seed': 42,
            'verbose': 0,
            'allow_writing_files': False,
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
            sw = sample_weights[tr_idx] if sample_weights is not None else None
            n_iter = params.pop('iterations')
            model = cb.CatBoostRegressor(**params, iterations=n_iter)
            params['iterations'] = n_iter
            pool_tr = cb.Pool(X_tr, y_tr, weight=sw)
            pool_te = cb.Pool(X_te, y_te)
            model.fit(pool_tr, eval_set=pool_te,
                      early_stopping_rounds=50, verbose=0)
            preds = model.predict(X_te)
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            fold_rmses.append(rmse)
        return np.mean(fold_rmses)

    study_cb = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_cb.optimize(cb_objective, n_trials=n_trials, show_progress_bar=False,
                      catch=(BaseException,))
    try:
        cb_params = study_cb.best_params
        print(f"   [Optuna-Huber] CB best RMSE: {study_cb.best_value:.6f}, "
              f"delta={cb_params.get('delta', 'N/A')}")
    except ValueError:
        print("   [Optuna-Huber] CB: ALL trials failed, using defaults")
        cb_params = {'delta': 2.0, 'depth': 6, 'learning_rate': 0.03,
                     'iterations': 800, 'subsample': 0.7, 'colsample_bylevel': 0.6,
                     'l2_leaf_reg': 1.0, 'min_data_in_leaf': 20,
                     'random_strength': 1.0, 'bagging_temperature': 1.0}

    return xgb_params, lgb_params, cb_params


# ======================================================================
# 5. V8 IMPROVEMENT #2: CatBoost RMSEWithUncertainty
# ======================================================================

def train_uncertainty_catboost(X, y, feature_cols, splits, n_trials=40):
    """
    CatBoost with RMSEWithUncertainty loss.
    Jointly predicts [mean, log_sigma].
    Returns predictions + uncertainty for confidence filtering.
    """
    print(f"\n   === UNCERTAINTY MODEL (CatBoost RMSEWithUncertainty) ===")
    X_feat = X[feature_cols]

    # Tune hyperparams
    def objective(trial):
        params = {
            'loss_function': 'RMSEWithUncertainty',
            'random_seed': 42,
            'verbose': 0,
            'posterior_sampling': False,
            'allow_writing_files': False,
            'depth': trial.suggest_int('depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
            'iterations': trial.suggest_int('iterations', 400, 1500),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 0.8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 20.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 60),
            'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
        }
        fold_rmses = []
        for tr_idx, te_idx in splits:
            X_tr, y_tr = X_feat.iloc[tr_idx].values, y.iloc[tr_idx].values
            X_te, y_te = X_feat.iloc[te_idx].values, y.iloc[te_idx].values
            n_iter = params.pop('iterations')
            model = cb.CatBoostRegressor(**params, iterations=n_iter)
            params['iterations'] = n_iter
            model.fit(X_tr, y_tr, eval_set=(X_te, y_te),
                      early_stopping_rounds=50, verbose=0)
            preds = model.predict(X_te)  # shape: (n, 2) -> [mean, log_var]
            if preds.ndim == 2:
                pred_means = preds[:, 0]
            else:
                pred_means = preds
            rmse = np.sqrt(mean_squared_error(y_te, pred_means))
            fold_rmses.append(rmse)
        return np.mean(fold_rmses)

    # Use fewer tuning splits for speed
    tune_splits = purged_wf_splits(len(X), n_splits=3, test_pct=0.15, purge=10)

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   catch=(BaseException,))
    try:
        best_params = study.best_params
        print(f"   [Uncertainty] Best RMSE: {study.best_value:.6f}")
    except ValueError:
        print("   [Uncertainty] ALL trials failed, using defaults")
        best_params = {'depth': 6, 'learning_rate': 0.03, 'iterations': 800,
                       'subsample': 0.7, 'colsample_bylevel': 0.6,
                       'l2_leaf_reg': 1.0, 'min_data_in_leaf': 20,
                       'random_strength': 1.0, 'bagging_temperature': 1.0}

    # Walk-forward predictions with uncertainty
    n = len(X)
    pred_means = np.full(n, np.nan)
    pred_log_vars = np.full(n, np.nan)

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X_feat.iloc[tr_idx].values, y.iloc[tr_idx].values
        X_te, y_te = X_feat.iloc[te_idx].values, y.iloc[te_idx].values

        params = dict(best_params)
        params['loss_function'] = 'RMSEWithUncertainty'
        params['random_seed'] = 42
        params['verbose'] = 0
        params['allow_writing_files'] = False
        n_iter = params.pop('iterations', 800)
        model = cb.CatBoostRegressor(**params, iterations=n_iter)
        model.fit(X_tr, y_tr, eval_set=(X_te, y_te),
                  early_stopping_rounds=50, verbose=0)
        preds = model.predict(X_te)

        if preds.ndim == 2 and preds.shape[1] >= 2:
            pred_means[te_idx] = preds[:, 0]
            pred_log_vars[te_idx] = preds[:, 1]
            rmse = np.sqrt(mean_squared_error(y_te, preds[:, 0]))
        else:
            pred_means[te_idx] = preds
            pred_log_vars[te_idx] = 0.0
            rmse = np.sqrt(mean_squared_error(y_te, preds))

        print(f"      Fold {fold_i+1}: RMSE={rmse:.6f}")

    # Compute signal-to-noise ratio
    valid = ~np.isnan(pred_means)
    snr = np.full(n, np.nan)
    # Clip log_vars to prevent overflow in exp()
    clipped_log_vars = np.clip(pred_log_vars[valid], -10, 10)
    sigma = np.exp(clipped_log_vars / 2)
    snr[valid] = np.abs(pred_means[valid]) / (sigma + 1e-10)

    print(f"   Uncertainty stats:")
    print(f"      Mean |prediction|: {np.nanmean(np.abs(pred_means)):.4f}")
    print(f"      Mean sigma:        {np.nanmean(sigma):.4f}")
    print(f"      Mean SNR:          {np.nanmean(snr[valid]):.4f}")
    print(f"      SNR > 1.0:         {(snr[valid] > 1.0).mean():.1%}")
    print(f"      SNR > 2.0:         {(snr[valid] > 2.0).mean():.1%}")

    return pred_means, pred_log_vars, snr, best_params


# ======================================================================
# 6. V8 IMPROVEMENT #3: MULTI-QUANTILE REGRESSION
# ======================================================================

def train_multiquantile_catboost(X, y, feature_cols, splits, n_trials=40):
    """
    CatBoost with MultiQuantile loss.
    Predicts 5 quantiles: 0.1, 0.25, 0.5, 0.75, 0.9
    For SHORT: only enter when q75 < 0 (even bullish tail is bearish).
    """
    print(f"\n   === MULTI-QUANTILE MODEL (CatBoost MultiQuantile) ===")
    X_feat = X[feature_cols]
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    q_str = ','.join(str(q) for q in quantiles)

    # Tune hyperparams using median quantile RMSE
    tune_splits = purged_wf_splits(len(X), n_splits=3, test_pct=0.15, purge=10)

    def objective(trial):
        params = {
            'loss_function': f'MultiQuantile:alpha={q_str}',
            'random_seed': 42,
            'verbose': 0,
            'allow_writing_files': False,
            'depth': trial.suggest_int('depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
            'iterations': trial.suggest_int('iterations', 400, 1500),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 0.8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 20.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 60),
            'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
        }
        fold_rmses = []
        for tr_idx, te_idx in tune_splits:
            X_tr, y_tr = X_feat.iloc[tr_idx].values, y.iloc[tr_idx].values
            X_te, y_te = X_feat.iloc[te_idx].values, y.iloc[te_idx].values
            n_iter = params.pop('iterations')
            model = cb.CatBoostRegressor(**params, iterations=n_iter)
            params['iterations'] = n_iter
            model.fit(X_tr, y_tr, eval_set=(X_te, y_te),
                      early_stopping_rounds=50, verbose=0)
            preds = model.predict(X_te)
            if preds.ndim == 2:
                # Use median quantile (index 2) for RMSE evaluation
                median_pred = preds[:, 2]
            else:
                median_pred = preds
            rmse = np.sqrt(mean_squared_error(y_te, median_pred))
            fold_rmses.append(rmse)
        return np.mean(fold_rmses)

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   catch=(BaseException,))
    try:
        best_params = study.best_params
        print(f"   [MultiQuantile] Best median RMSE: {study.best_value:.6f}")
    except ValueError:
        print("   [MultiQuantile] ALL trials failed, using defaults")
        best_params = {'depth': 6, 'learning_rate': 0.03, 'iterations': 800,
                       'subsample': 0.7, 'colsample_bylevel': 0.6,
                       'l2_leaf_reg': 1.0, 'min_data_in_leaf': 20,
                       'random_strength': 1.0, 'bagging_temperature': 1.0}

    # Walk-forward predictions
    n = len(X)
    n_q = len(quantiles)
    all_quantile_preds = np.full((n, n_q), np.nan)

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X_feat.iloc[tr_idx].values, y.iloc[tr_idx].values
        X_te, y_te = X_feat.iloc[te_idx].values, y.iloc[te_idx].values

        params = dict(best_params)
        params['loss_function'] = f'MultiQuantile:alpha={q_str}'
        params['random_seed'] = 42
        params['verbose'] = 0
        params['allow_writing_files'] = False
        n_iter = params.pop('iterations', 800)
        model = cb.CatBoostRegressor(**params, iterations=n_iter)
        model.fit(X_tr, y_tr, eval_set=(X_te, y_te),
                  early_stopping_rounds=50, verbose=0)
        preds = model.predict(X_te)

        if preds.ndim == 2 and preds.shape[1] == n_q:
            all_quantile_preds[te_idx] = preds
            rmse = np.sqrt(mean_squared_error(y_te, preds[:, 2]))
        else:
            # Fallback: single output
            all_quantile_preds[te_idx, 2] = preds if preds.ndim == 1 else preds.ravel()
            rmse = np.sqrt(mean_squared_error(y_te, preds if preds.ndim == 1 else preds.ravel()))

        print(f"      Fold {fold_i+1}: median RMSE={rmse:.6f}")

    # Analyze quantile spread
    valid = ~np.isnan(all_quantile_preds[:, 2])
    if valid.sum() > 0:
        q10, q50, q90 = (all_quantile_preds[valid, 0],
                          all_quantile_preds[valid, 2],
                          all_quantile_preds[valid, 4])
        spread = q90 - q10
        print(f"\n   Quantile stats:")
        print(f"      Mean q10:    {np.mean(q10):.4f}")
        print(f"      Mean q50:    {np.mean(q50):.4f}")
        print(f"      Mean q90:    {np.mean(q90):.4f}")
        print(f"      Mean spread: {np.mean(spread):.4f}")
        print(f"      q75 < 0:     {(all_quantile_preds[valid, 3] < 0).mean():.1%} "
              f"(confident SHORT)")
        print(f"      q25 > 0:     {(all_quantile_preds[valid, 1] > 0).mean():.1%} "
              f"(confident LONG)")

    return all_quantile_preds, quantiles, best_params


# ======================================================================
# 7. V8 IMPROVEMENT #4: EXPONENTIAL DECAY SAMPLE WEIGHTING
# ======================================================================

def build_exponential_decay_weights(n_samples, half_life_hours=4000):
    """
    Build exponential decay weights.
    w_i = exp(-λ * (n - i))  where λ = ln(2) / half_life

    More recent data gets higher weight.
    half_life_hours: after this many hours, weight drops to 50%.
    """
    lam = np.log(2) / half_life_hours
    indices = np.arange(n_samples)
    weights = np.exp(-lam * (n_samples - 1 - indices))
    # Normalize so mean weight = 1.0 (preserves effective sample size)
    weights = weights / weights.mean()
    return weights


# ======================================================================
# 8. V8 IMPROVEMENT #5: ADVERSARIAL VALIDATION WEIGHTING
# ======================================================================

def build_adversarial_weights(X, feature_cols, recent_frac=0.20):
    """
    Adversarial validation-based sample weighting.

    1. Label most recent 20% of data as "recent" (y=1), rest as "old" (y=0)
    2. Train LightGBM classifier to distinguish them
    3. Get P(recent | x_i) for all training samples
    4. Upweight samples that "look like" recent data

    The idea: if a historical sample has features similar to recent data,
    it's more relevant for predicting the current market.
    """
    print(f"\n   === ADVERSARIAL VALIDATION WEIGHTING ===")

    n = len(X)
    cutoff = int(n * (1 - recent_frac))

    # Labels: 0 = old, 1 = recent
    adv_labels = np.zeros(n)
    adv_labels[cutoff:] = 1.0

    X_feat = X[feature_cols].values

    # Train adversarial classifier
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_samples=30,
        random_state=42,
        verbose=-1,
    )

    # Use 5-fold CV to get out-of-fold probabilities
    from sklearn.model_selection import StratifiedKFold
    oof_probs = np.zeros(n)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for tr_idx, te_idx in skf.split(X_feat, adv_labels):
        model.fit(X_feat[tr_idx], adv_labels[tr_idx])
        oof_probs[te_idx] = model.predict_proba(X_feat[te_idx])[:, 1]

    # AUC tells us how much drift exists
    adv_auc = roc_auc_score(adv_labels, oof_probs)
    print(f"   Adversarial AUC: {adv_auc:.4f} "
          f"({'significant drift' if adv_auc > 0.6 else 'moderate drift' if adv_auc > 0.55 else 'minimal drift'})")

    # Convert probabilities to weights
    # Higher P(recent) → higher weight
    # Normalize so mean = 1.0
    weights = oof_probs.copy()
    weights = np.clip(weights, 0.05, 0.95)  # Avoid extreme weights
    weights = weights / weights.mean()

    print(f"   Weight stats: min={weights.min():.3f}, "
          f"max={weights.max():.3f}, "
          f"mean={weights.mean():.3f}")
    print(f"   Weight ratio (recent/old): "
          f"{weights[cutoff:].mean() / weights[:cutoff].mean():.2f}x")

    return weights, adv_auc


# ======================================================================
# 9. WALK-FORWARD REGRESSION (V8: Huber loss, sample weights)
# ======================================================================

def walk_forward_regression_v8(X, y, feature_cols, splits,
                                xgb_params, lgb_params, cb_params,
                                sample_weights=None,
                                label="V8"):
    """
    Walk-forward regression with Huber loss, optional sample weighting.
    2-model ensemble (XGB + LGB) — V7 showed CatBoost didn't add value
    as a 3rd ensemble member. We use CB separately for uncertainty/quantile.
    """
    print(f"\n   === WALK-FORWARD REGRESSION ({label}) ===")

    n = len(X)
    xgb_preds = np.full(n, np.nan)
    lgb_preds = np.full(n, np.nan)

    xgb_rmses, lgb_rmses = [], []
    X_feat = X[feature_cols]

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X_feat.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X_feat.iloc[te_idx], y.iloc[te_idx]
        sw = sample_weights[tr_idx] if sample_weights is not None else None

        # --- XGBoost with Huber ---
        xgb_p = dict(xgb_params)
        huber_slope = xgb_p.pop('huber_slope', 1.35)
        xgb_p['objective'] = 'reg:pseudohubererror'
        xgb_p['huber_slope'] = huber_slope
        xgb_p['tree_method'] = 'hist'
        xgb_p['random_state'] = 42
        n_est = xgb_p.pop('n_estimators', 800)
        m_xgb = xgb.XGBRegressor(**xgb_p, n_estimators=n_est,
                                   early_stopping_rounds=50)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False,
                  sample_weight=sw)
        p_xgb = m_xgb.predict(X_te)
        rmse_xgb = np.sqrt(mean_squared_error(y_te, p_xgb))
        xgb_rmses.append(rmse_xgb)
        xgb_preds[te_idx] = p_xgb

        # --- LightGBM with Huber ---
        lgb_p = dict(lgb_params)
        alpha = lgb_p.pop('alpha', 1.35)
        lgb_p['objective'] = 'huber'
        lgb_p['alpha'] = alpha
        lgb_p['metric'] = 'rmse'
        lgb_p['random_state'] = 42
        lgb_p['verbose'] = -1
        n_est_l = lgb_p.pop('n_estimators', 800)
        m_lgb = lgb.LGBMRegressor(**lgb_p, n_estimators=n_est_l)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False)],
                  sample_weight=sw)
        p_lgb = m_lgb.predict(X_te)
        rmse_lgb = np.sqrt(mean_squared_error(y_te, p_lgb))
        lgb_rmses.append(rmse_lgb)
        lgb_preds[te_idx] = p_lgb

        print(f"      Fold {fold_i+1}: XGB={rmse_xgb:.6f} | LGB={rmse_lgb:.6f}")

    mean_xgb = np.mean(xgb_rmses)
    mean_lgb = np.mean(lgb_rmses)
    print(f"\n   Mean RMSE: XGB={mean_xgb:.6f} | LGB={mean_lgb:.6f}")

    # Inverse-RMSE 2-model ensemble
    inv_rmses = np.array([1.0/mean_xgb, 1.0/mean_lgb])
    weights = inv_rmses / inv_rmses.sum()
    print(f"   Ensemble weights: XGB={weights[0]:.3f} | LGB={weights[1]:.3f}")

    ensemble_preds = np.full(n, np.nan)
    for i in range(n):
        vals, wts = [], []
        if not np.isnan(xgb_preds[i]):
            vals.append(xgb_preds[i])
            wts.append(weights[0])
        if not np.isnan(lgb_preds[i]):
            vals.append(lgb_preds[i])
            wts.append(weights[1])
        if vals:
            wts = np.array(wts)
            wts /= wts.sum()
            ensemble_preds[i] = np.dot(vals, wts)

    return {
        'xgb': xgb_preds,
        'lgb': lgb_preds,
        'ensemble': ensemble_preds,
        'weights': weights,
        'rmses': {'xgb': mean_xgb, 'lgb': mean_lgb},
    }


# ======================================================================
# 10. V6/V7 BASELINE (RMSE loss, no weighting)
# ======================================================================

def walk_forward_baseline(X, y, feature_cols, splits, label="baseline"):
    """V6/V7-style regression with default params for fair comparison."""
    print(f"\n   === BASELINE REGRESSION ({label}) ===")
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
        print(f"      Fold {fold_i+1}: RMSE={rmse:.6f}")

    print(f"      Mean RMSE: {np.mean(fold_rmses):.6f}")
    return preds


# ======================================================================
# 11. SIGNAL CONVERSION (enhanced with confidence filters)
# ======================================================================

def regression_to_signals(pred_returns, threshold_pct=0.005):
    """Convert return predictions to long/short probability-like signals."""
    long_probs = np.full_like(pred_returns, np.nan, dtype=np.float64)
    short_probs = np.full_like(pred_returns, np.nan, dtype=np.float64)

    valid = ~np.isnan(pred_returns)
    v = pred_returns[valid]
    long_probs[valid] = 1.0 / (1.0 + np.exp(-v * 200))
    short_probs[valid] = 1.0 / (1.0 + np.exp(v * 200))

    return long_probs, short_probs


def signals_with_uncertainty_filter(pred_returns, snr, snr_threshold=1.0):
    """
    Convert predictions to signals, but ZERO OUT low-confidence predictions.
    Only generate signals where |prediction| / sigma > snr_threshold.
    """
    long_probs = np.full_like(pred_returns, np.nan, dtype=np.float64)
    short_probs = np.full_like(pred_returns, np.nan, dtype=np.float64)

    valid = ~np.isnan(pred_returns) & ~np.isnan(snr)
    confident = valid & (snr >= snr_threshold)

    v = pred_returns[confident]
    long_probs[confident] = 1.0 / (1.0 + np.exp(-v * 200))
    short_probs[confident] = 1.0 / (1.0 + np.exp(v * 200))

    n_valid = valid.sum()
    n_confident = confident.sum()
    pct = n_confident / n_valid * 100 if n_valid > 0 else 0
    print(f"   Uncertainty filter (SNR>={snr_threshold}): "
          f"{n_confident}/{n_valid} bars pass ({pct:.1f}%)")

    return long_probs, short_probs


def signals_with_quantile_filter(quantile_preds, quantiles):
    """
    Convert quantile predictions to signals with confidence filtering.

    SHORT: only when q75 < 0 (even bullish tail is bearish)
    LONG:  only when q25 > 0 (even bearish tail is bullish)

    Uses median (q50) as the signal strength.
    """
    n = len(quantile_preds)
    long_probs = np.full(n, np.nan, dtype=np.float64)
    short_probs = np.full(n, np.nan, dtype=np.float64)

    q_idx = {q: i for i, q in enumerate(quantiles)}
    q25_idx = q_idx.get(0.25, 1)
    q50_idx = q_idx.get(0.5, 2)
    q75_idx = q_idx.get(0.75, 3)

    valid = ~np.isnan(quantile_preds[:, q50_idx])

    for i in range(n):
        if not valid[i]:
            continue
        q25 = quantile_preds[i, q25_idx]
        q50 = quantile_preds[i, q50_idx]
        q75 = quantile_preds[i, q75_idx]

        # LONG: confident when even q25 (bearish tail) is positive
        if q25 > 0:
            long_probs[i] = 1.0 / (1.0 + np.exp(-q50 * 200))

        # SHORT: confident when even q75 (bullish tail) is negative
        if q75 < 0:
            short_probs[i] = 1.0 / (1.0 + np.exp(q50 * 200))

    n_long = np.sum(~np.isnan(long_probs))
    n_short = np.sum(~np.isnan(short_probs))
    print(f"   Quantile filter: {n_long} confident LONGs, "
          f"{n_short} confident SHORTs "
          f"(from {valid.sum()} valid bars)")

    return long_probs, short_probs


def signals_quantile_relaxed(quantile_preds, quantiles, strictness='medium'):
    """
    Relaxed quantile filter with configurable strictness.

    strict:  q75 < 0 for SHORT (original)
    medium:  q50 < 0 AND q75 < median(q75) for SHORT
    relaxed: q50 < 0 for SHORT (just use median, spread as confidence)
    """
    n = len(quantile_preds)
    long_probs = np.full(n, np.nan, dtype=np.float64)
    short_probs = np.full(n, np.nan, dtype=np.float64)

    q_idx = {q: i for i, q in enumerate(quantiles)}
    q10_idx = q_idx.get(0.1, 0)
    q25_idx = q_idx.get(0.25, 1)
    q50_idx = q_idx.get(0.5, 2)
    q75_idx = q_idx.get(0.75, 3)
    q90_idx = q_idx.get(0.9, 4)

    valid = ~np.isnan(quantile_preds[:, q50_idx])

    if strictness == 'strict':
        # Original: q75 < 0 for SHORT
        for i in range(n):
            if not valid[i]:
                continue
            q50 = quantile_preds[i, q50_idx]
            if quantile_preds[i, q25_idx] > 0:
                long_probs[i] = 1.0 / (1.0 + np.exp(-q50 * 200))
            if quantile_preds[i, q75_idx] < 0:
                short_probs[i] = 1.0 / (1.0 + np.exp(q50 * 200))

    elif strictness == 'medium':
        # Medium: q50 < 0 AND spread is narrow
        valid_spreads = quantile_preds[valid, q90_idx] - quantile_preds[valid, q10_idx]
        median_spread = np.median(valid_spreads) if len(valid_spreads) > 0 else 999

        for i in range(n):
            if not valid[i]:
                continue
            q50 = quantile_preds[i, q50_idx]
            spread = quantile_preds[i, q90_idx] - quantile_preds[i, q10_idx]

            if q50 > 0 and spread < median_spread:
                long_probs[i] = 1.0 / (1.0 + np.exp(-q50 * 200))
            if q50 < 0 and spread < median_spread:
                short_probs[i] = 1.0 / (1.0 + np.exp(q50 * 200))

    elif strictness == 'relaxed':
        # Relaxed: just use q50 sign, scale by inverse spread
        for i in range(n):
            if not valid[i]:
                continue
            q50 = quantile_preds[i, q50_idx]
            spread = quantile_preds[i, q90_idx] - quantile_preds[i, q10_idx]
            inv_spread = 1.0 / (spread + 1e-6)

            if q50 > 0:
                # Stronger signal when spread is tight
                long_probs[i] = 1.0 / (1.0 + np.exp(-q50 * inv_spread * 5))
            if q50 < 0:
                short_probs[i] = 1.0 / (1.0 + np.exp(q50 * inv_spread * 5))

    n_long = np.sum(~np.isnan(long_probs))
    n_short = np.sum(~np.isnan(short_probs))
    label = strictness
    print(f"   Quantile filter ({label}): {n_long} LONGs, {n_short} SHORTs")

    return long_probs, short_probs


# ======================================================================
# 12. PORTFOLIO SIMULATOR (identical to V7)
# ======================================================================

def simulate_portfolio(close, high, low, atr, hours,
                       long_probs, short_probs, cfg):
    """Bar-by-bar portfolio simulator. Identical to V7."""
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
    direction = cfg.get('direction', 'both')
    funding = cfg.get('funding_rates', None)
    cooldown = cfg.get('cooldown', 0)
    max_dd_pct = cfg.get('max_drawdown_pct', 0.10)

    signal_mode = cfg.get('signal_mode', 'percentile')
    signal_pct = cfg.get('signal_pct', 0.05)
    regime = cfg.get('regime_mask', np.ones(n, dtype=bool))

    if signal_mode == 'percentile':
        lp_valid = long_probs[~np.isnan(long_probs)]
        sp_valid = short_probs[~np.isnan(short_probs)]
        long_thr = np.percentile(lp_valid, 100 * (1 - signal_pct)) if len(lp_valid) > 0 else 1.0
        short_thr = np.percentile(sp_valid, 100 * (1 - signal_pct)) if len(sp_valid) > 0 else 1.0
    else:
        long_thr = cfg.get('signal_thr', 0.15)
        short_thr = cfg.get('signal_thr', 0.15)

    long_sig = (~np.isnan(long_probs)) & (long_probs >= long_thr) & regime
    short_sig = (~np.isnan(short_probs)) & (short_probs >= short_thr) & regime

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
                    if exit_mode == 'breakeven':
                        if not be_active and high[i] >= entry_p + be_trigger * atr_entry:
                            current_sl = max(current_sl, entry_p)
                            be_active = True

                if exit_price is not None:
                    gross_ret = (exit_price - entry_p) / entry_p
                    net_ret_main = gross_ret * remaining * pos_size
                    total_pnl = net_ret_main + fund_pnl - total_cost * pos_size
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
                    if exit_mode == 'breakeven':
                        if not be_active and low[i] <= entry_p - be_trigger * atr_entry:
                            current_sl = min(current_sl, entry_p)
                            be_active = True

                if exit_price is not None:
                    gross_ret = (entry_p - exit_price) / entry_p
                    net_ret_main = gross_ret * remaining * pos_size
                    total_pnl = net_ret_main + fund_pnl - total_cost * pos_size
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
                    fund_pnl = 0.0
                    remaining = 1.0

    return trades


def calc_stats(trades, label="", initial_capital=25000):
    """Calculate portfolio statistics from trade list."""
    if not trades or len(trades) < 3:
        return None
    df_t = pd.DataFrame(trades)
    n = len(df_t)

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
    sharpe = (np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(365 * 2)
              if len(rets) > 1 else 0)

    return {
        'label': label, 'n_trades': n, 'wr': wr, 'pf': pf,
        'ret': tr, 'max_dd': max_dd, 'sharpe': sharpe,
        'final_cap': final,
        'avg_pnl': df_t['pnl'].mean(),
        'avg_win': df_t.loc[wins, 'pnl'].mean() if wins.sum() > 0 else 0,
        'avg_loss': df_t.loc[~wins, 'pnl'].mean() if (~wins).sum() > 0 else 0,
        'n_long': (df_t['side'] == 'LONG').sum(),
        'n_short': (df_t['side'] == 'SHORT').sum(),
        'tp': (df_t['reason'] == 'TP').sum(),
        'sl': (df_t['reason'] == 'SL').sum(),
        'timeout': n - (df_t['reason'] == 'TP').sum() - (df_t['reason'] == 'SL').sum(),
    }


# ======================================================================
# 13. STRATEGY SWEEP
# ======================================================================

def sweep_strategies(close, high, low, atr, hours,
                     long_probs, short_probs,
                     regime_mask, funding_rates,
                     model_label="model"):
    """Sweep strategy parameters. Same grid as V7 for fair comparison."""
    results = []

    barrier_configs = [
        (1.5, 1.0, "TP1.5_SL1.0"),
        (2.0, 1.0, "TP2.0_SL1.0"),
        (2.0, 1.5, "TP2.0_SL1.5"),
        (2.5, 1.5, "TP2.5_SL1.5"),
        (3.0, 1.5, "TP3.0_SL1.5"),
    ]
    signal_configs = [(0.05, 'top5%'), (0.10, 'top10%'), (0.20, 'top20%')]
    fee_configs = [(0.02, 0.01, 'maker'), (0.01, 0.005, 'vip')]
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
    print(f"      Done: {done} configs ({elapsed:.0f}s), "
          f"{len(results)} valid, {n_prof} profitable")
    return results


def print_top_results(results, title="RESULTS", top_n=20):
    """Print ranked results table."""
    profitable = sorted([r for r in results if r and r['pf'] > 1.0],
                        key=lambda x: x['pf'], reverse=True)

    print(f"\n{'='*130}")
    print(f"  {title}")
    print(f"  Total: {len(results):,} | Profitable: {len(profitable)}")
    print(f"{'='*130}")

    show = profitable[:top_n] if profitable else sorted(
        [r for r in results if r], key=lambda x: x['pf'], reverse=True)[:top_n]

    if not show:
        print("  No results.")
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

    return profitable


# ======================================================================
# MAIN EXPERIMENT
# ======================================================================

def main():
    t_start = time.time()
    print("=" * 100)
    print("  V8 ADVANCED REGRESSION EXPERIMENT")
    print(f"  Started: {datetime.now()}")
    print(f"  Improvements: Huber Loss | Uncertainty | Multi-Quantile | "
          f"Decay Weighting | Adversarial Weighting")
    print(f"  Baseline: V7 best PF=1.58 (vol-norm, SHORT-only)")
    print(f"  Account: $25,000 prop firm | Max DD: 10%")
    print("=" * 100)

    # ---- 1. Load data ----
    print("\n[1/10] Loading and preparing data...")
    df = load_and_prepare_data()
    all_features = get_feature_cols(df)
    print(f"    Using ALL {len(all_features)} features (V7 showed selection hurts)")

    # ---- 2. Build targets ----
    print("\n[2/10] Building vol-normalized target...")
    y_vol_norm, y_raw = build_vol_normalized_target(df, horizon=8)
    df['y_vol_norm'] = y_vol_norm
    df['y_raw'] = y_raw
    df['fwd_ret_8h'] = np.log(df['close'].shift(-8) / df['close'])

    mask = (df['y_vol_norm'].notna() & df['fwd_ret_8h'].notna() &
            df[all_features].notna().all(axis=1))
    df_clean = df[mask].copy().reset_index(drop=True)
    print(f"    Clean samples: {len(df_clean):,}")

    # Arrays for simulation
    close_v = df_clean['close'].values
    high_v = df_clean['high'].values
    low_v = df_clean['low'].values
    atr_v = df_clean['atr_14'].values if 'atr_14' in df_clean.columns else np.ones(len(df_clean))
    hours_v = df_clean['_hour'].values
    regime_v = df_clean['regime_ok'].values
    fund_v = df_clean['funding_rate'].values if 'funding_rate' in df_clean.columns else None

    splits = purged_wf_splits(len(df_clean), n_splits=5, test_pct=0.12, purge=10)
    tune_splits = purged_wf_splits(len(df_clean), n_splits=3, test_pct=0.15, purge=10)

    all_results = []
    summary = {}

    # ================================================================
    # EXPERIMENT A: V7 BASELINE (vol-norm target, RMSE loss, all features)
    # ================================================================
    print("\n" + "=" * 100)
    print("  EXPERIMENT A: V7 BASELINE (vol-norm, RMSE, all features, no weighting)")
    print("=" * 100, flush=True)

    v7_preds = walk_forward_baseline(
        df_clean, df_clean['y_vol_norm'], all_features, splits, "v7_baseline")
    v7_long, v7_short = regression_to_signals(v7_preds)

    v7_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        v7_long, v7_short, regime_v, fund_v, "v7_baseline")
    v7_prof = print_top_results(v7_results, "V7 BASELINE RESULTS")
    all_results.extend(v7_results)
    summary['v7_baseline'] = {
        'n_total': len(v7_results),
        'n_profitable': len(v7_prof),
        'best_pf': max((r['pf'] for r in v7_results), default=0),
    }

    # ================================================================
    # EXPERIMENT B: HUBER LOSS (no weighting)
    # ================================================================
    print("\n" + "=" * 100)
    print("  EXPERIMENT B: HUBER LOSS (vol-norm target, all features)")
    print("=" * 100, flush=True)

    huber_xgb, huber_lgb, huber_cb = optuna_tune_huber(
        df_clean, df_clean['y_vol_norm'], all_features, tune_splits,
        n_trials=50)

    v8b_result = walk_forward_regression_v8(
        df_clean, df_clean['y_vol_norm'], all_features, splits,
        huber_xgb, huber_lgb, huber_cb, label="v8_huber")

    v8b_long, v8b_short = regression_to_signals(v8b_result['ensemble'])
    v8b_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        v8b_long, v8b_short, regime_v, fund_v, "v8_huber")
    v8b_prof = print_top_results(v8b_results, "V8 HUBER LOSS RESULTS")
    all_results.extend(v8b_results)
    summary['v8_huber'] = {
        'n_total': len(v8b_results),
        'n_profitable': len(v8b_prof),
        'best_pf': max((r['pf'] for r in v8b_results), default=0),
        'huber_params': {
            'xgb_slope': huber_xgb.get('huber_slope', 'N/A'),
            'lgb_alpha': huber_lgb.get('alpha', 'N/A'),
            'cb_delta': huber_cb.get('delta', 'N/A'),
        },
    }

    # ================================================================
    # EXPERIMENT C: UNCERTAINTY FILTER (CatBoost RMSEWithUncertainty)
    # ================================================================
    print("\n" + "=" * 100)
    print("  EXPERIMENT C: UNCERTAINTY-FILTERED SIGNALS")
    print("=" * 100, flush=True)

    unc_means, unc_log_vars, unc_snr, unc_params = train_uncertainty_catboost(
        df_clean, df_clean['y_vol_norm'], all_features, splits,
        n_trials=40)

    # Test multiple SNR thresholds
    for snr_thr in [0.5, 1.0, 1.5, 2.0]:
        unc_long, unc_short = signals_with_uncertainty_filter(
            unc_means, unc_snr, snr_threshold=snr_thr)
        label = f"v8_unc_snr{snr_thr}"
        unc_results = sweep_strategies(
            close_v, high_v, low_v, atr_v, hours_v,
            unc_long, unc_short, regime_v, fund_v, label)
        unc_prof = print_top_results(unc_results,
                                     f"UNCERTAINTY FILTER (SNR>={snr_thr})")
        all_results.extend(unc_results)
        summary[label] = {
            'n_total': len(unc_results),
            'n_profitable': len(unc_prof),
            'best_pf': max((r['pf'] for r in unc_results), default=0),
        }

    # Also test raw uncertainty model without filter (just predictions)
    unc_raw_long, unc_raw_short = regression_to_signals(unc_means)
    unc_raw_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        unc_raw_long, unc_raw_short, regime_v, fund_v, "v8_unc_raw")
    unc_raw_prof = print_top_results(unc_raw_results,
                                     "UNCERTAINTY MODEL (no filter)")
    all_results.extend(unc_raw_results)
    summary['v8_unc_raw'] = {
        'n_total': len(unc_raw_results),
        'n_profitable': len(unc_raw_prof),
        'best_pf': max((r['pf'] for r in unc_raw_results), default=0),
    }

    # ================================================================
    # EXPERIMENT D: MULTI-QUANTILE FILTER
    # ================================================================
    print("\n" + "=" * 100)
    print("  EXPERIMENT D: MULTI-QUANTILE REGRESSION FILTER")
    print("=" * 100, flush=True)

    mq_preds, mq_quantiles, mq_params = train_multiquantile_catboost(
        df_clean, df_clean['y_vol_norm'], all_features, splits,
        n_trials=40)

    # Strict filter: q75 < 0 for SHORT
    mq_strict_long, mq_strict_short = signals_with_quantile_filter(
        mq_preds, mq_quantiles)
    mq_strict_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        mq_strict_long, mq_strict_short, regime_v, fund_v, "v8_mq_strict")
    mq_strict_prof = print_top_results(mq_strict_results,
                                        "MULTI-QUANTILE (strict: q75<0)")
    all_results.extend(mq_strict_results)
    summary['v8_mq_strict'] = {
        'n_total': len(mq_strict_results),
        'n_profitable': len(mq_strict_prof),
        'best_pf': max((r['pf'] for r in mq_strict_results), default=0),
    }

    # Medium filter
    mq_med_long, mq_med_short = signals_quantile_relaxed(
        mq_preds, mq_quantiles, strictness='medium')
    mq_med_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        mq_med_long, mq_med_short, regime_v, fund_v, "v8_mq_medium")
    mq_med_prof = print_top_results(mq_med_results,
                                     "MULTI-QUANTILE (medium)")
    all_results.extend(mq_med_results)
    summary['v8_mq_medium'] = {
        'n_total': len(mq_med_results),
        'n_profitable': len(mq_med_prof),
        'best_pf': max((r['pf'] for r in mq_med_results), default=0),
    }

    # Relaxed filter
    mq_rel_long, mq_rel_short = signals_quantile_relaxed(
        mq_preds, mq_quantiles, strictness='relaxed')
    mq_rel_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        mq_rel_long, mq_rel_short, regime_v, fund_v, "v8_mq_relaxed")
    mq_rel_prof = print_top_results(mq_rel_results,
                                     "MULTI-QUANTILE (relaxed)")
    all_results.extend(mq_rel_results)
    summary['v8_mq_relaxed'] = {
        'n_total': len(mq_rel_results),
        'n_profitable': len(mq_rel_prof),
        'best_pf': max((r['pf'] for r in mq_rel_results), default=0),
    }

    # ================================================================
    # EXPERIMENT E: EXPONENTIAL DECAY WEIGHTING + HUBER
    # ================================================================
    print("\n" + "=" * 100)
    print("  EXPERIMENT E: EXPONENTIAL DECAY WEIGHTING + HUBER LOSS")
    print("=" * 100, flush=True)

    # Test multiple half-lives
    for hl in [2000, 4000, 8000]:
        decay_weights = build_exponential_decay_weights(len(df_clean),
                                                        half_life_hours=hl)
        print(f"\n   --- Half-life = {hl}h ---")
        print(f"   Oldest weight: {decay_weights[0]:.4f}, "
              f"Newest: {decay_weights[-1]:.4f}, "
              f"Ratio: {decay_weights[-1]/decay_weights[0]:.1f}x")

        # Tune with weights
        hw_xgb, hw_lgb, hw_cb = optuna_tune_huber(
            df_clean, df_clean['y_vol_norm'], all_features, tune_splits,
            n_trials=30, sample_weights=decay_weights)

        hw_result = walk_forward_regression_v8(
            df_clean, df_clean['y_vol_norm'], all_features, splits,
            hw_xgb, hw_lgb, hw_cb, sample_weights=decay_weights,
            label=f"v8_decay_hl{hl}")

        hw_long, hw_short = regression_to_signals(hw_result['ensemble'])
        label = f"v8_decay{hl}"
        hw_results = sweep_strategies(
            close_v, high_v, low_v, atr_v, hours_v,
            hw_long, hw_short, regime_v, fund_v, label)
        hw_prof = print_top_results(hw_results,
                                    f"DECAY WEIGHTING (hl={hl}h) + HUBER")
        all_results.extend(hw_results)
        summary[label] = {
            'n_total': len(hw_results),
            'n_profitable': len(hw_prof),
            'best_pf': max((r['pf'] for r in hw_results), default=0),
            'half_life': hl,
        }

    # ================================================================
    # EXPERIMENT F: ADVERSARIAL WEIGHTING + HUBER
    # ================================================================
    print("\n" + "=" * 100)
    print("  EXPERIMENT F: ADVERSARIAL VALIDATION WEIGHTING + HUBER LOSS")
    print("=" * 100, flush=True)

    adv_weights, adv_auc = build_adversarial_weights(
        df_clean, all_features, recent_frac=0.20)

    # Tune with adversarial weights
    aw_xgb, aw_lgb, aw_cb = optuna_tune_huber(
        df_clean, df_clean['y_vol_norm'], all_features, tune_splits,
        n_trials=40, sample_weights=adv_weights)

    aw_result = walk_forward_regression_v8(
        df_clean, df_clean['y_vol_norm'], all_features, splits,
        aw_xgb, aw_lgb, aw_cb, sample_weights=adv_weights,
        label="v8_adversarial")

    aw_long, aw_short = regression_to_signals(aw_result['ensemble'])
    aw_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        aw_long, aw_short, regime_v, fund_v, "v8_adversarial")
    aw_prof = print_top_results(aw_results,
                                "ADVERSARIAL WEIGHTING + HUBER")
    all_results.extend(aw_results)
    summary['v8_adversarial'] = {
        'n_total': len(aw_results),
        'n_profitable': len(aw_prof),
        'best_pf': max((r['pf'] for r in aw_results), default=0),
        'adversarial_auc': adv_auc,
    }

    # ================================================================
    # EXPERIMENT G: COMBINED BEST (Huber + Decay + Uncertainty filter)
    # ================================================================
    print("\n" + "=" * 100)
    print("  EXPERIMENT G: COMBINED BEST "
          "(Huber + Decay + Uncertainty filter)")
    print("=" * 100, flush=True)

    # Use best half-life from experiment E
    best_hl_key = max(
        [k for k in summary if k.startswith('v8_decay')],
        key=lambda k: summary[k]['best_pf'],
        default=None
    )
    best_hl = summary[best_hl_key]['half_life'] if best_hl_key else 4000
    print(f"   Using best half-life: {best_hl}h")

    combo_weights = build_exponential_decay_weights(len(df_clean),
                                                     half_life_hours=best_hl)

    # Train Huber ensemble with decay
    combo_xgb, combo_lgb, combo_cb = optuna_tune_huber(
        df_clean, df_clean['y_vol_norm'], all_features, tune_splits,
        n_trials=40, sample_weights=combo_weights)

    combo_result = walk_forward_regression_v8(
        df_clean, df_clean['y_vol_norm'], all_features, splits,
        combo_xgb, combo_lgb, combo_cb, sample_weights=combo_weights,
        label="v8_combo")

    # Apply uncertainty filter on the Huber ensemble predictions
    # We reuse the SNR from the uncertainty model as a meta-filter
    for snr_thr in [0.5, 1.0]:
        # Use uncertainty SNR to filter Huber ensemble signals
        combo_long, combo_short = signals_with_uncertainty_filter(
            combo_result['ensemble'], unc_snr, snr_threshold=snr_thr)
        label = f"v8_combo_snr{snr_thr}"
        combo_results = sweep_strategies(
            close_v, high_v, low_v, atr_v, hours_v,
            combo_long, combo_short, regime_v, fund_v, label)
        combo_prof = print_top_results(combo_results,
                                       f"COMBINED (Huber+Decay+SNR>={snr_thr})")
        all_results.extend(combo_results)
        summary[label] = {
            'n_total': len(combo_results),
            'n_profitable': len(combo_prof),
            'best_pf': max((r['pf'] for r in combo_results), default=0),
        }

    # Also without filter
    combo_nf_long, combo_nf_short = regression_to_signals(
        combo_result['ensemble'])
    combo_nf_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        combo_nf_long, combo_nf_short, regime_v, fund_v, "v8_combo_nofilt")
    combo_nf_prof = print_top_results(combo_nf_results,
                                      "COMBINED (Huber+Decay, no filter)")
    all_results.extend(combo_nf_results)
    summary['v8_combo_nofilt'] = {
        'n_total': len(combo_nf_results),
        'n_profitable': len(combo_nf_prof),
        'best_pf': max((r['pf'] for r in combo_nf_results), default=0),
    }

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print("\n" + "=" * 130)
    print("  V8 GRAND SUMMARY")
    print("=" * 130)

    all_profitable = print_top_results(all_results,
                                       "ALL V8 STRATEGIES RANKED", top_n=50)

    # Head-to-head comparison
    print(f"\n{'='*100}")
    print(f"  MODEL-BY-MODEL COMPARISON")
    print(f"{'='*100}")
    print(f"\n  {'Model':<30s} {'Exps':>6s} {'Prof':>6s} {'Prof%':>7s} "
          f"{'Best PF':>8s} {'Best Sharpe':>11s}")
    print(f"  {'-'*75}")

    model_tags = sorted(summary.keys())
    for tag in model_tags:
        s = summary[tag]
        matching = [r for r in all_results if r and r['label'].startswith(tag)]
        best_sharpe = max((r['sharpe'] for r in matching), default=0)
        prof_pct = s['n_profitable'] / s['n_total'] * 100 if s['n_total'] > 0 else 0
        marker = " ★" if s['best_pf'] > summary.get('v7_baseline', {}).get('best_pf', 0) else ""
        print(f"  {tag:<30s} {s['n_total']:>6d} {s['n_profitable']:>6d} "
              f"{prof_pct:>6.1f}% {s['best_pf']:>8.2f} {best_sharpe:>11.2f}{marker}")

    # Improvement over V7
    v7_best = summary.get('v7_baseline', {}).get('best_pf', 1.0)
    print(f"\n  V7 Baseline best PF: {v7_best:.2f}")
    best_v8 = max(s['best_pf'] for s in summary.values())
    best_v8_model = max(summary.keys(), key=lambda k: summary[k]['best_pf'])
    print(f"  V8 Best PF:          {best_v8:.2f} ({best_v8_model})")
    delta_pf = best_v8 - v7_best
    print(f"  Improvement:         {delta_pf:+.2f} "
          f"({delta_pf/v7_best*100:+.1f}%)")

    # Top strategies for prop firm
    if all_profitable:
        print(f"\n{'='*100}")
        print(f"  TOP 10 STRATEGIES FOR $25K PROP FIRM")
        print(f"{'='*100}")
        for i, s in enumerate(all_profitable[:10]):
            dollar_pnl = s['final_cap'] - 25000
            print(f"\n  {i+1}. {s['label']}")
            print(f"     Trades: {s['n_trades']} | WR: {s['wr']:.1%} | PF: {s['pf']:.2f}")
            print(f"     Total: {s['ret']:+.1f}% (${dollar_pnl:+,.0f}) | "
                  f"DD: {s['max_dd']:.1f}% | Sharpe: {s['sharpe']:.2f}")
            print(f"     Avg: ${s['avg_pnl']:+.2f} | "
                  f"Win: ${s['avg_win']:+.2f} | Loss: ${s['avg_loss']:+.2f}")
            print(f"     L/S: {s['n_long']}/{s['n_short']}")

    # Save results
    duration = time.time() - t_start
    save_data = {
        'experiment': 'V8 Advanced Regression',
        'improvements': [
            'Huber loss (down-weights tail outliers)',
            'CatBoost RMSEWithUncertainty (confidence filter)',
            'Multi-Quantile regression (distributional prediction)',
            'Exponential decay sample weighting',
            'Adversarial validation sample weighting',
        ],
        'v7_baseline_pf': v7_best,
        'v8_best_pf': best_v8,
        'v8_best_model': best_v8_model,
        'pf_improvement': delta_pf,
        'total_experiments': len(all_results),
        'profitable_count': len(all_profitable) if all_profitable else 0,
        'top_50': (all_profitable[:50] if all_profitable else []),
        'model_summary': summary,
        'huber_params': {
            'xgb': huber_xgb,
            'lgb': huber_lgb,
            'cb': huber_cb,
        },
        'uncertainty_params': unc_params,
        'multiquantile_params': mq_params,
        'adversarial_auc': adv_auc,
        'best_decay_half_life': best_hl,
        'n_features': len(all_features),
        'timestamp': str(datetime.now()),
        'duration_seconds': duration,
        'account_size': 25000,
    }

    out_path = OUTPUT_DIR / "v8_experiment_results.json"
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n   Results saved to {out_path}")

    print(f"\n   Total duration: {duration/60:.1f} minutes")
    print(f"   Experiments run: {len(all_results):,}")
    print("\n   V8 experiment complete!")


if __name__ == "__main__":
    main()
