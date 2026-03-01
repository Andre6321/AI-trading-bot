"""
V9 Maximum Edge Experiment
=============================
Building on V8's finding that exponential decay (λ=8000) was the key win (PF 1.64),
this experiment applies EVERY profitable idea to squeeze maximum edge:

  PROVEN FROM V8 (keep):
    1. Exponential decay sample weighting (half-life=8000)
    2. Huber loss (down-weights tails)
    3. Vol-normalized 8h target (proven in V7+V8)
    4. SHORT-only direction (strongest edge)
    5. Trailing stop 0.75×ATR

  NEW V9 IMPROVEMENTS:
    6. LONGER DATA: 2020-01 → 2026-02 (53K bars vs 43K, +24%)
       - Includes COVID crash, full 2020 bull run, 2021 peak, 2022 bear,
         2023 recovery, 2024-2025 cycle → robust across ALL regimes
    7. ZERO-FEE MODELING: model exchanges with 0% maker fees
       (Hyperliquid, dYdX, MEXC) for realistic PF boost
    8. HOUR-OF-DAY FILTER: avoid low-liquidity dead zones
       (06:00-10:00 UTC historically weakest for BTC shorts)
    9. MULTI-TIMEFRAME CONFIRMATION: 4H structure alignment
       (only short when 4H trend is also bearish)
    10. KELLY CRITERION POSITION SIZING: optimal risk allocation
        based on WR and win/loss ratio from walk-forward folds
    11. FUNDING RATE BOOST: boost short signals when funding is
        positive (longs paying shorts = edge amplification)
    12. VOLATILITY-SCALED POSITION SIZING: larger positions in
        moderate vol, smaller in extreme vol (both ways)
    13. MORE WALK-FORWARD FOLDS: 7 folds instead of 5 for better
        out-of-sample estimation with the larger dataset
    14. EXPANDED DECAY SEARCH: test half-lives 6000, 8000, 10000, 12000
    15. MOMENTUM ACCELERATION FILTER: only enter when short-term
        momentum is accelerating in our direction

  Strategy:
    - V8 decay + Huber = proven foundation (don't change what works)
    - Layer on ALL new edges multiplicatively
    - Compare: V8-baseline vs V9-full vs individual improvements
    - More data = more walk-forward folds = more robust estimates

Usage:
    python scripts/v9_maximum_edge.py 2>&1 | Tee-Object -FilePath logs/v9_run.log
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
signal.signal(signal.SIGINT, signal.SIG_IGN)

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features.alpha_features import build_alpha_features

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Tee stdout to a log file so output is always captured
class TeeWriter:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_log_file = open(LOG_DIR / "v9_live.log", "w", encoding="utf-8")
sys.stdout = TeeWriter(sys.__stdout__, _log_file)
sys.stderr = TeeWriter(sys.__stderr__, _log_file)


# ======================================================================
# 1. DATA LOADING (extended to 2020)
# ======================================================================

def load_and_prepare_data():
    """Load data, build features, return clean DataFrame."""
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
    print(f"   Date range: {df.index.min()} to {df.index.max()}")

    # Build alpha features
    df = build_alpha_features(df)
    # Only drop rows where NON-funding columns are inf/NaN
    # funding_rate can be NaN (it's sparse — every 8h)
    non_fund_cols = [c for c in df.columns if c != 'funding_rate']
    df[non_fund_cols] = df[non_fund_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=non_fund_cols)
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

    # Load funding rates and merge (only if not already present)
    if 'funding_rate' not in df.columns or df['funding_rate'].isna().all():
        funding_path = ROOT / "data" / "raw" / "btcusdt_funding_rates.parquet"
        if funding_path.exists():
            fund_df = pd.read_parquet(funding_path)
            if 'timestamp' in fund_df.columns:
                fund_df['timestamp'] = pd.to_datetime(fund_df['timestamp'])
                fund_df.set_index('timestamp', inplace=True)
            fund_df = fund_df[~fund_df.index.duplicated(keep='first')]
            if 'funding_rate' in fund_df.columns:
                # Drop existing funding_rate if it exists but is all NaN
                if 'funding_rate' in df.columns:
                    df = df.drop(columns=['funding_rate'])
                df = df.join(fund_df[['funding_rate']], how='left')
                df['funding_rate'] = df['funding_rate'].ffill()
                print(f"   Funding rates merged: {df['funding_rate'].notna().sum():,} bars")
            else:
                df['funding_rate'] = np.nan
        else:
            df['funding_rate'] = np.nan
            print("   Warning: No funding rate data found")
    else:
        # Forward-fill existing funding_rate
        df['funding_rate'] = df['funding_rate'].ffill()
        print(f"   Funding rates already present: {df['funding_rate'].notna().sum():,} bars")

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
# 2. TARGET ENGINEERING (from V7/V8 — proven winner)
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
# 3. WALK-FORWARD SPLITS (more folds for more data)
# ======================================================================

def purged_wf_splits(n, n_splits=7, test_pct=0.10, purge=12):
    """Walk-forward splits with purge gap. More folds for larger dataset."""
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
# 4. EXPONENTIAL DECAY WEIGHTS (V8 proven winner)
# ======================================================================

def build_exponential_decay_weights(n_samples, half_life_hours=8000):
    """
    Build exponential decay weights.
    w_i = exp(-λ * (n - i))  where λ = ln(2) / half_life
    """
    lam = np.log(2) / half_life_hours
    indices = np.arange(n_samples)
    weights = np.exp(-lam * (n_samples - 1 - indices))
    weights = weights / weights.mean()
    return weights


# ======================================================================
# 5. OPTUNA TUNING WITH HUBER + DECAY (from V8)
# ======================================================================

def optuna_tune_huber(X, y, feature_cols, splits, n_trials=50,
                      sample_weights=None):
    """Optuna-tune XGB + LGB with HUBER loss."""
    print(f"\n   [Optuna-Huber] Tuning 2 models ({n_trials} trials each)...")
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
        print(f"   [Optuna-Huber] XGB best RMSE: {study_xgb.best_value:.6f}")
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
        print(f"   [Optuna-Huber] LGB best RMSE: {study_lgb.best_value:.6f}")
    except ValueError:
        print("   [Optuna-Huber] LGB: ALL trials failed, using defaults")
        lgb_params = {'alpha': 2.0, 'num_leaves': 63, 'learning_rate': 0.03,
                      'n_estimators': 800, 'subsample': 0.7, 'colsample_bytree': 0.6,
                      'reg_alpha': 0.5, 'reg_lambda': 1.0, 'min_child_samples': 20}

    return xgb_params, lgb_params


# ======================================================================
# 6. WALK-FORWARD REGRESSION (XGB + LGB ensemble with Huber)
# ======================================================================

def walk_forward_regression(X, y, feature_cols, splits,
                            xgb_params, lgb_params,
                            sample_weights=None, label="model"):
    """Walk-forward regression with 2-model ensemble."""
    print(f"\n   === WALK-FORWARD REGRESSION ({label}) ===")
    n = len(X)
    X_feat = X[feature_cols]
    xgb_preds = np.full(n, np.nan)
    lgb_preds = np.full(n, np.nan)
    xgb_rmses, lgb_rmses = [], []

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

    # Inverse-RMSE ensemble
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
        'ensemble': ensemble_preds,
        'weights': weights,
        'rmses': {'xgb': mean_xgb, 'lgb': mean_lgb},
    }


# ======================================================================
# 7. BASELINE (RMSE loss, no weighting, no filters)
# ======================================================================

def walk_forward_baseline(X, y, feature_cols, splits, label="baseline"):
    """V7-style regression with default params for fair comparison."""
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
# 8. SIGNAL CONVERSION
# ======================================================================

def regression_to_signals(pred_returns):
    """Convert return predictions to long/short probability-like signals."""
    long_probs = np.full_like(pred_returns, np.nan, dtype=np.float64)
    short_probs = np.full_like(pred_returns, np.nan, dtype=np.float64)

    valid = ~np.isnan(pred_returns)
    v = pred_returns[valid]
    long_probs[valid] = 1.0 / (1.0 + np.exp(-v * 200))
    short_probs[valid] = 1.0 / (1.0 + np.exp(v * 200))

    return long_probs, short_probs


# ======================================================================
# 9. V9 ENHANCEMENT: FUNDING RATE SIGNAL BOOST
# ======================================================================

def apply_funding_boost(short_probs, funding_rates, boost_factor=1.15):
    """
    Boost short signal probability when funding rate is positive.
    Positive funding = longs paying shorts → structural edge for shorts.
    
    boost_factor: multiply short_prob by this when funding > 0
    """
    boosted = short_probs.copy()
    for i in range(len(boosted)):
        if np.isnan(boosted[i]) or np.isnan(funding_rates[i]):
            continue
        if funding_rates[i] > 0:
            # Positive funding → shorts get paid → boost signal
            boosted[i] = min(boosted[i] * boost_factor, 1.0)
        elif funding_rates[i] < -0.0003:
            # Very negative funding → shorts paying longs → dampen signal
            boosted[i] = boosted[i] * 0.9
    return boosted


# ======================================================================
# 10. V9 ENHANCEMENT: MOMENTUM ACCELERATION FILTER
# ======================================================================

def apply_momentum_filter(short_probs, close, atr, lookback_short=6,
                          lookback_long=24):
    """
    Only keep short signals where short-term momentum is accelerating
    downward relative to longer-term momentum.
    
    This filters out shorts during strong uptrend bounces.
    """
    filtered = short_probs.copy()
    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.concatenate([[0], log_ret])
    
    n = len(close)
    for i in range(lookback_long, n):
        if np.isnan(filtered[i]):
            continue
        
        # Short-term momentum (last 6h)
        mom_short = np.sum(log_ret[i-lookback_short+1:i+1])
        # Long-term momentum (last 24h)
        mom_long = np.sum(log_ret[i-lookback_long+1:i+1])
        
        # Expected short-term momentum from long-term trend
        expected_short = mom_long * (lookback_short / lookback_long)
        
        # If short-term momentum is MORE POSITIVE than expected
        # (i.e., bouncing UP during our short setup), kill the signal
        if mom_short > expected_short + 0.003:  # 0.3% buffer
            filtered[i] = np.nan
    
    n_orig = np.sum(~np.isnan(short_probs))
    n_filt = np.sum(~np.isnan(filtered))
    print(f"   Momentum filter: {n_orig} → {n_filt} signals "
          f"({n_orig - n_filt} filtered out, {(n_orig-n_filt)/max(n_orig,1)*100:.1f}%)")
    return filtered


# ======================================================================
# 11. PORTFOLIO SIMULATOR (V9 Enhanced)
# ======================================================================

def simulate_portfolio(close, high, low, atr, hours,
                       long_probs, short_probs, cfg):
    """
    Bar-by-bar portfolio simulator.
    V9 enhancements:
      - Hour-of-day filter
      - Kelly criterion position sizing
      - Volatility-scaled position sizing
      - Funding rate P&L tracking
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
    direction = cfg.get('direction', 'both')
    funding = cfg.get('funding_rates', None)
    cooldown = cfg.get('cooldown', 0)
    max_dd_pct = cfg.get('max_drawdown_pct', 0.10)

    # V9: Hour-of-day filter
    hour_filter = cfg.get('hour_filter', None)  # e.g., (6, 10) = skip 06-10 UTC
    
    # V9: Kelly criterion
    use_kelly = cfg.get('use_kelly', False)
    kelly_wr = cfg.get('kelly_wr', 0.50)
    kelly_wl_ratio = cfg.get('kelly_wl_ratio', 1.65)
    
    # V9: Vol-scaled sizing
    vol_scale = cfg.get('vol_scale', False)

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

    # V9: Hour filter
    if hour_filter is not None:
        skip_start, skip_end = hour_filter
        for i in range(n):
            h = hours[i] if i < len(hours) else 0
            if skip_start <= h < skip_end:
                long_sig[i] = False
                short_sig[i] = False

    in_pos = False
    trades = []
    last_exit_bar = -cooldown - 1
    dd_breached = False

    # V9: Kelly fraction
    if use_kelly:
        kelly_f = (kelly_wr * kelly_wl_ratio - (1 - kelly_wr)) / kelly_wl_ratio
        kelly_f = max(0.005, min(kelly_f, 0.25))  # Clamp to 0.5%-25%
        risk_pct = kelly_f * 0.5  # Use half-Kelly for safety
    
    # V9: Compute rolling ATR percentile for vol-scaling
    atr_pct_arr = atr / (close + 1e-10) * 100
    atr_median = np.nanmedian(atr_pct_arr)

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
                
                # V9: Vol-scaled position sizing
                effective_risk = risk_pct
                if vol_scale:
                    vol_ratio = atr_pct_arr[i] / (atr_median + 1e-10)
                    # Scale inversely: low vol = larger size, high vol = smaller
                    # But cap at 0.5x to 2.0x
                    vol_multiplier = max(0.5, min(2.0, 1.0 / vol_ratio))
                    effective_risk = risk_pct * vol_multiplier
                
                pos_size = min(cap * effective_risk / max(sl_dist_pct, 1e-6),
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
# 12. STRATEGY SWEEP (V9 Enhanced)
# ======================================================================

def sweep_strategies(close, high, low, atr, hours,
                     long_probs, short_probs,
                     regime_mask, funding_rates,
                     model_label="model",
                     kelly_stats=None):
    """Sweep strategy parameters with V9 enhancements."""
    results = []

    barrier_configs = [
        (1.5, 1.0, "TP1.5_SL1.0"),
        (2.0, 1.0, "TP2.0_SL1.0"),
        (2.0, 1.5, "TP2.0_SL1.5"),
        (2.5, 1.5, "TP2.5_SL1.5"),
        (3.0, 1.5, "TP3.0_SL1.5"),
    ]
    signal_configs = [(0.05, 'top5%'), (0.10, 'top10%'), (0.20, 'top20%')]
    
    # V9: Added zero-fee and rebate tiers
    fee_configs = [
        (0.02, 0.01, 'maker'),       # Standard maker
        (0.01, 0.005, 'vip'),         # VIP tier
        (0.00, 0.003, 'zero_fee'),    # dYdX/MEXC 0% maker
        (-0.002, 0.002, 'rebate'),    # Hyperliquid maker rebate
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
    
    # V9: Hour filter configs
    hour_filter_configs = [
        (None, 'allhours'),
        ((6, 10), 'skip0610'),    # Skip low-liquidity UTC morning
    ]
    
    # V9: Position sizing configs
    sizing_configs = [
        (False, False, 'std'),          # Standard 1% risk
        (True, False, 'kelly'),         # Kelly criterion
        (False, True, 'volscale'),      # Vol-scaled
        (True, True, 'kelly_volscale'), # Both
    ]

    total = (len(barrier_configs) * len(signal_configs) * len(dir_configs) *
             len(fee_configs) * len(exit_configs) * len(hold_configs) *
             len(cooldown_configs) * len(hour_filter_configs) *
             len(sizing_configs))
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
                                for (hf, hf_lbl) in hour_filter_configs:
                                    for (use_k, use_vs, sz_lbl) in sizing_configs:
                                        cfg = {
                                            'tp_mult': tp, 'sl_mult': sl,
                                            'max_hold': hold,
                                            'exit_mode': exit_mode,
                                            'trail_mult': trail_m,
                                            'be_trigger': 1.0,
                                            'fee_pct': fee_p,
                                            'slippage_pct': slip_p,
                                            'signal_mode': 'percentile',
                                            'signal_pct': sig_pct,
                                            'direction': direction,
                                            'initial_capital': 25000,
                                            'risk_per_trade': 0.01,
                                            'max_pos_frac': 0.25,
                                            'cooldown': cd,
                                            'max_drawdown_pct': 0.10,
                                            'regime_mask': regime_mask,
                                            'funding_rates': funding_rates,
                                            'hour_filter': hf,
                                            'use_kelly': use_k,
                                            'vol_scale': use_vs,
                                        }
                                        # V9: Set Kelly params from walk-forward stats
                                        if use_k and kelly_stats:
                                            cfg['kelly_wr'] = kelly_stats.get('wr', 0.50)
                                            cfg['kelly_wl_ratio'] = kelly_stats.get('wl_ratio', 1.65)
                                        
                                        label = (f"{model_label}|{b_lbl}|{sig_lbl}"
                                                 f"|{direction}|{fee_lbl}|{exit_lbl}"
                                                 f"|H{hold}|{hf_lbl}|{sz_lbl}")
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
    """Print top results sorted by PF."""
    if not results:
        print(f"\n   {title}: No valid results")
        return []
    sorted_r = sorted(results, key=lambda x: x['pf'], reverse=True)
    profitable = [r for r in sorted_r if r['pf'] > 1.0]

    print(f"\n   {'='*120}")
    print(f"   {title}")
    print(f"   {'='*120}")
    print(f"   Total: {len(results)}, Profitable: {len(profitable)} "
          f"({len(profitable)/len(results)*100:.0f}%)")
    print(f"\n   {'#':<4s} {'Label':<65s} {'Trades':>6s} {'WR':>6s} "
          f"{'PF':>6s} {'Ret%':>7s} {'DD%':>6s} {'Sharpe':>7s}")
    print(f"   {'-'*110}")
    for i, r in enumerate(sorted_r[:top_n]):
        print(f"   {i+1:<4d} {r['label']:<65s} {r['n_trades']:>6d} "
              f"{r['wr']:>5.1%} {r['pf']:>6.2f} {r['ret']:>+6.1f}% "
              f"{r['max_dd']:>5.1f}% {r['sharpe']:>7.2f}")
    return profitable


# ======================================================================
# 13. MAIN EXPERIMENT
# ======================================================================

def main():
    t_start = time.time()
    print("=" * 130)
    print("  V9 MAXIMUM EDGE EXPERIMENT")
    print("  Extended data (2020→2026) + ALL profitability improvements")
    print("=" * 130, flush=True)

    # --- Load data ---
    print("\n[1] LOADING DATA...")
    df = load_and_prepare_data()
    all_features = get_feature_cols(df)
    print(f"   Features: {len(all_features)}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    print(f"   Total bars: {len(df):,}")

    # BTC price change over test period
    btc_start = df['close'].iloc[0]
    btc_end = df['close'].iloc[-1]
    btc_ret = (btc_end - btc_start) / btc_start * 100
    print(f"   BTC: ${btc_start:,.0f} → ${btc_end:,.0f} ({btc_ret:+.1f}%)")

    # --- Build target ---
    print("\n[2] BUILDING TARGET...")
    df['y_vol_norm'], df['y_raw_fwd'] = build_vol_normalized_target(df)
    
    # Drop NaN target rows
    df_clean = df.dropna(subset=['y_vol_norm'])
    print(f"   Clean samples: {len(df_clean):,}")

    # --- Walk-forward splits (V9: 7 folds for larger dataset) ---
    print("\n[3] WALK-FORWARD SPLITS...")
    splits = purged_wf_splits(len(df_clean), n_splits=7, test_pct=0.10, purge=12)
    tune_splits = purged_wf_splits(len(df_clean), n_splits=3, test_pct=0.15, purge=10)
    print(f"   Main splits: {len(splits)} folds")
    print(f"   Tuning splits: {len(tune_splits)} folds")
    for i, (tr, te) in enumerate(splits):
        print(f"      Fold {i+1}: train={len(tr):,} | test={len(te):,} | "
              f"test dates: {df_clean.index[te[0]].date()} → "
              f"{df_clean.index[te[-1]].date()}")

    # Vectors for simulation
    close_v = df_clean['close'].values
    high_v = df_clean['high'].values
    low_v = df_clean['low'].values
    atr_v = df_clean['atr_14'].values if 'atr_14' in df_clean.columns else \
            np.full(len(df_clean), 1.0)
    hours_v = df_clean['_hour'].values
    regime_v = df_clean['regime_ok'].values
    
    # Funding rates
    if 'funding_rate' in df_clean.columns:
        fund_v = df_clean['funding_rate'].values
    else:
        fund_v = np.zeros(len(df_clean))

    # ================================================================
    # Track all results
    # ================================================================
    all_results = []
    summary = {}

    # ================================================================
    # EXPERIMENT A: V7 BASELINE (RMSE, no weighting, no filters)
    # ================================================================
    print("\n" + "=" * 130)
    print("  EXPERIMENT A: V7 BASELINE (RMSE, no weighting)")
    print("=" * 130, flush=True)

    baseline_preds = walk_forward_baseline(
        df_clean, df_clean['y_vol_norm'], all_features, splits,
        label="v7_baseline_6yr")

    bl_long, bl_short = regression_to_signals(baseline_preds)
    bl_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        bl_long, bl_short, regime_v, fund_v, "v7_baseline")
    bl_prof = print_top_results(bl_results, "V7 BASELINE (6-year data)")
    all_results.extend(bl_results)
    summary['v7_baseline'] = {
        'n_total': len(bl_results),
        'n_profitable': len(bl_prof),
        'best_pf': max((r['pf'] for r in bl_results), default=0),
    }

    # ================================================================
    # EXPERIMENT B: V8 DECAY (proven winner, now with 6yr data)
    # ================================================================
    print("\n" + "=" * 130)
    print("  EXPERIMENT B: V8 DECAY WEIGHTING + HUBER (6-year data)")
    print("=" * 130, flush=True)

    for hl in [6000, 8000, 10000, 12000]:
        decay_weights = build_exponential_decay_weights(len(df_clean),
                                                        half_life_hours=hl)
        print(f"\n   --- Half-life = {hl}h ---")
        print(f"   Oldest weight: {decay_weights[0]:.4f}, "
              f"Newest: {decay_weights[-1]:.4f}, "
              f"Ratio: {decay_weights[-1]/decay_weights[0]:.1f}x")

        hw_xgb, hw_lgb = optuna_tune_huber(
            df_clean, df_clean['y_vol_norm'], all_features, tune_splits,
            n_trials=40, sample_weights=decay_weights)

        hw_result = walk_forward_regression(
            df_clean, df_clean['y_vol_norm'], all_features, splits,
            hw_xgb, hw_lgb, sample_weights=decay_weights,
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

    # Find best half-life
    best_hl_key = max(
        [k for k in summary if k.startswith('v8_decay')],
        key=lambda k: summary[k]['best_pf'],
        default='v8_decay8000'
    )
    best_hl = summary[best_hl_key].get('half_life', 8000)
    print(f"\n   >>> Best half-life: {best_hl}h ({best_hl_key})")

    # ================================================================
    # EXPERIMENT C: V9 FULL EDGE (best decay + ALL enhancements)
    # ================================================================
    print("\n" + "=" * 130)
    print("  EXPERIMENT C: V9 FULL EDGE")
    print(f"  (decay hl={best_hl} + funding boost + momentum filter)")
    print("=" * 130, flush=True)

    # Retrain with best half-life
    best_weights = build_exponential_decay_weights(len(df_clean),
                                                    half_life_hours=best_hl)
    v9_xgb, v9_lgb = optuna_tune_huber(
        df_clean, df_clean['y_vol_norm'], all_features, tune_splits,
        n_trials=50, sample_weights=best_weights)

    v9_result = walk_forward_regression(
        df_clean, df_clean['y_vol_norm'], all_features, splits,
        v9_xgb, v9_lgb, sample_weights=best_weights,
        label="v9_full")

    v9_long, v9_short = regression_to_signals(v9_result['ensemble'])

    # --- V9 enhancement: funding boost ---
    print("\n   Applying funding rate boost to short signals...")
    for boost in [1.0, 1.10, 1.15, 1.20]:
        if boost == 1.0:
            v9_short_boosted = v9_short.copy()
            boost_lbl = "noboost"
        else:
            v9_short_boosted = apply_funding_boost(v9_short, fund_v,
                                                    boost_factor=boost)
            boost_lbl = f"fboost{int(boost*100)}"

        # --- V9 enhancement: momentum filter ---
        for use_mom_filter in [False, True]:
            if use_mom_filter:
                v9_short_final = apply_momentum_filter(
                    v9_short_boosted, close_v, atr_v)
                mom_lbl = "momfilt"
            else:
                v9_short_final = v9_short_boosted
                mom_lbl = "nomom"

            label = f"v9_{boost_lbl}_{mom_lbl}"
            
            # Compute Kelly stats from walk-forward trades
            # First do a quick pre-sweep to get WR and W/L ratio
            quick_cfg = {
                'tp_mult': 2.0, 'sl_mult': 1.0, 'max_hold': 12,
                'exit_mode': 'trailing', 'trail_mult': 0.75,
                'fee_pct': 0.01, 'slippage_pct': 0.005,
                'signal_mode': 'percentile', 'signal_pct': 0.05,
                'direction': 'short', 'initial_capital': 25000,
                'risk_per_trade': 0.01, 'max_pos_frac': 0.25,
                'cooldown': 0, 'max_drawdown_pct': 0.10,
                'regime_mask': regime_v, 'funding_rates': fund_v,
            }
            quick_trades = simulate_portfolio(
                close_v, high_v, low_v, atr_v, hours_v,
                v9_long, v9_short_final, quick_cfg)
            kelly_stats = None
            if quick_trades and len(quick_trades) > 10:
                qdf = pd.DataFrame(quick_trades)
                q_wins = qdf['pnl'] > 0
                q_wr = q_wins.mean()
                q_avg_win = qdf.loc[q_wins, 'pnl'].mean() if q_wins.sum() > 0 else 1
                q_avg_loss = abs(qdf.loc[~q_wins, 'pnl'].mean()) if (~q_wins).sum() > 0 else 1
                q_wl = q_avg_win / q_avg_loss if q_avg_loss > 0 else 1.5
                kelly_stats = {'wr': q_wr, 'wl_ratio': q_wl}
                print(f"   Kelly stats ({label}): WR={q_wr:.1%}, "
                      f"W/L={q_wl:.2f}, "
                      f"Kelly_f={max(0,(q_wr*q_wl-(1-q_wr))/q_wl):.3f}")

            v9_results = sweep_strategies(
                close_v, high_v, low_v, atr_v, hours_v,
                v9_long, v9_short_final, regime_v, fund_v, label,
                kelly_stats=kelly_stats)
            v9_prof = print_top_results(v9_results, f"V9 FULL EDGE ({label})")
            all_results.extend(v9_results)
            summary[label] = {
                'n_total': len(v9_results),
                'n_profitable': len(v9_prof),
                'best_pf': max((r['pf'] for r in v9_results), default=0),
                'kelly_stats': kelly_stats,
            }

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print("\n" + "=" * 130)
    print("  V9 GRAND SUMMARY")
    print("=" * 130)

    all_profitable = print_top_results(all_results,
                                       "ALL V9 STRATEGIES RANKED", top_n=50)

    # Head-to-head comparison
    print(f"\n{'='*120}")
    print(f"  MODEL-BY-MODEL COMPARISON")
    print(f"{'='*120}")
    print(f"\n  {'Model':<40s} {'Exps':>6s} {'Prof':>6s} {'Prof%':>7s} "
          f"{'Best PF':>8s} {'Best Sharpe':>11s}")
    print(f"  {'-'*85}")

    model_tags = sorted(summary.keys())
    for tag in model_tags:
        s = summary[tag]
        matching = [r for r in all_results if r and r['label'].startswith(tag)]
        best_sharpe = max((r['sharpe'] for r in matching), default=0)
        prof_pct = s['n_profitable'] / s['n_total'] * 100 if s['n_total'] > 0 else 0
        v7_pf = summary.get('v7_baseline', {}).get('best_pf', 0)
        marker = " ★" if s['best_pf'] > v7_pf else ""
        print(f"  {tag:<40s} {s['n_total']:>6d} {s['n_profitable']:>6d} "
              f"{prof_pct:>6.1f}% {s['best_pf']:>8.2f} {best_sharpe:>11.2f}{marker}")

    # Improvement over baselines
    v7_best = summary.get('v7_baseline', {}).get('best_pf', 1.0)
    print(f"\n  V7 Baseline best PF (6yr data): {v7_best:.2f}")
    best_v9 = max(s['best_pf'] for s in summary.values())
    best_v9_model = max(summary.keys(), key=lambda k: summary[k]['best_pf'])
    print(f"  V9 Best PF:                     {best_v9:.2f} ({best_v9_model})")
    delta_pf = best_v9 - v7_best
    print(f"  Improvement:                    {delta_pf:+.2f} "
          f"({delta_pf/v7_best*100:+.1f}%)")

    # V8 reference (from previous run)
    v8_ref_pf = 1.64
    delta_v8 = best_v9 - v8_ref_pf
    print(f"\n  V8 Reference PF (5yr data):     {v8_ref_pf:.2f}")
    print(f"  V9 vs V8:                       {delta_v8:+.2f} "
          f"({delta_v8/v8_ref_pf*100:+.1f}%)")

    # BTC comparison
    print(f"\n  BTC Buy & Hold return:          {btc_ret:+.1f}%")
    if all_profitable:
        best_ret = all_profitable[0]['ret']
        print(f"  V9 Best strategy return:        {best_ret:+.1f}%")
        print(f"  Outperformance:                 {best_ret - btc_ret:+.1f}%")

    # Top strategies for prop firm
    if all_profitable:
        print(f"\n{'='*120}")
        print(f"  TOP 15 STRATEGIES FOR $25K PROP FIRM")
        print(f"{'='*120}")
        for i, s in enumerate(all_profitable[:15]):
            dollar_pnl = s['final_cap'] - 25000
            print(f"\n  {i+1}. {s['label']}")
            print(f"     Trades: {s['n_trades']} | WR: {s['wr']:.1%} | PF: {s['pf']:.2f}")
            print(f"     Total: {s['ret']:+.1f}% (${dollar_pnl:+,.0f}) | "
                  f"DD: {s['max_dd']:.1f}% | Sharpe: {s['sharpe']:.2f}")
            print(f"     Avg: ${s['avg_pnl']:+.2f} | "
                  f"Win: ${s['avg_win']:+.2f} | Loss: ${s['avg_loss']:+.2f}")
            print(f"     L/S: {s['n_long']}/{s['n_short']} | "
                  f"TP: {s['tp']} | SL: {s['sl']} | Timeout: {s['timeout']}")

    # Save results
    duration = time.time() - t_start
    save_data = {
        'experiment': 'V9 Maximum Edge',
        'improvements': [
            'Extended data: 2020-01 → 2026-02 (53K bars, 6+ years)',
            'Exponential decay sample weighting (best half-life tuned)',
            'Huber loss (down-weights tail outliers)',
            'Zero-fee exchange modeling (Hyperliquid/dYdX)',
            'Hour-of-day filter (skip low-liquidity 0600-1000 UTC)',
            'Kelly criterion position sizing',
            'Volatility-scaled position sizing',
            'Funding rate signal boost',
            'Momentum acceleration filter',
            '7 walk-forward folds (vs 5 in V8)',
            'Expanded decay half-life search (6K/8K/10K/12K)',
        ],
        'data_range': f"{df.index.min()} to {df.index.max()}",
        'total_bars': len(df_clean),
        'btc_return_pct': btc_ret,
        'v7_baseline_pf': v7_best,
        'v8_reference_pf': v8_ref_pf,
        'v9_best_pf': best_v9,
        'v9_best_model': best_v9_model,
        'pf_improvement_vs_v7': best_v9 - v7_best,
        'pf_improvement_vs_v8': delta_v8,
        'best_half_life': best_hl,
        'total_experiments': len(all_results),
        'profitable_count': len(all_profitable) if all_profitable else 0,
        'top_50': (all_profitable[:50] if all_profitable else []),
        'model_summary': {k: {kk: vv for kk, vv in v.items()
                              if kk != 'kelly_stats'}
                          for k, v in summary.items()},
        'n_features': len(all_features),
        'timestamp': str(datetime.now()),
        'duration_seconds': duration,
        'account_size': 25000,
    }

    out_path = OUTPUT_DIR / "v9_experiment_results.json"
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n   Results saved to {out_path}")

    print(f"\n   Total duration: {duration/60:.1f} minutes")
    print(f"   Experiments run: {len(all_results):,}")
    print("\n   V9 experiment complete!")


if __name__ == "__main__":
    main()
