"""
Train the final 8h regression model on ALL data for live trading.

The V6 experiment used walk-forward (train/test splits) for honest evaluation.
For live deployment, we retrain on the FULL dataset to maximize signal quality.

Outputs:
  - models/reg_8h_lgb_final.txt        (LightGBM model)
  - models/reg_8h_xgb_final.json       (XGBoost model)
  - models/reg_8h_metadata.json        (feature list, thresholds, etc.)
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features.alpha_features import build_alpha_features

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


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
            and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]


def main():
    print("=" * 70)
    print("  TRAINING FINAL 8H REGRESSION MODEL FOR LIVE DEPLOYMENT")
    print("=" * 70)

    # 1. Load data
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

    print(f"  Loaded {len(df):,} rows")

    # 2. Build features
    df = build_alpha_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  After cleanup: {len(df):,} rows")

    feature_cols = get_feature_cols(df)
    print(f"  Features: {len(feature_cols)}")

    # 3. Build target: 8h forward log-return
    df['fwd_ret_8h'] = np.log(df['close'].shift(-8) / df['close'])
    mask = ~np.isnan(df['fwd_ret_8h'])
    X = df.loc[mask, feature_cols]
    y = df.loc[mask, 'fwd_ret_8h']
    print(f"  Training samples: {len(X):,}")

    # 4. Train LightGBM
    print("\n  Training LightGBM...")
    m_lgb = lgb.LGBMRegressor(
        objective='regression', metric='rmse', random_state=42, verbose=-1,
        max_depth=5, learning_rate=0.03, n_estimators=800,
        subsample=0.7, colsample_bytree=0.6, min_child_samples=30,
        reg_alpha=1.0, reg_lambda=5.0, num_leaves=31,
    )
    m_lgb.fit(X, y)
    lgb_path = MODEL_DIR / "reg_8h_lgb_final.txt"
    m_lgb.booster_.save_model(str(lgb_path))
    print(f"  Saved: {lgb_path}")

    # 5. Train XGBoost
    print("  Training XGBoost...")
    m_xgb = xgb.XGBRegressor(
        objective='reg:squarederror', eval_metric='rmse',
        tree_method='hist', random_state=42,
        max_depth=5, learning_rate=0.03, n_estimators=800,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=20,
        gamma=1.0, reg_alpha=1.0, reg_lambda=5.0,
    )
    m_xgb.fit(X, y)
    xgb_path = MODEL_DIR / "reg_8h_xgb_final.json"
    m_xgb.save_model(str(xgb_path))
    print(f"  Saved: {xgb_path}")

    # 6. Compute signal thresholds from training predictions
    preds = (m_lgb.predict(X) + m_xgb.predict(X)) / 2
    long_probs = 1.0 / (1.0 + np.exp(-preds * 200))  # sigmoid scaling
    short_probs = 1.0 / (1.0 + np.exp(preds * 200))

    # Top 5% thresholds (matching best config from V6)
    long_thr_5 = float(np.percentile(long_probs, 95))
    short_thr_5 = float(np.percentile(short_probs, 95))
    long_thr_10 = float(np.percentile(long_probs, 90))
    short_thr_10 = float(np.percentile(short_probs, 90))
    long_thr_20 = float(np.percentile(long_probs, 80))
    short_thr_20 = float(np.percentile(short_probs, 80))

    # ATR baseline for stop-loss calculation
    if 'atr_14' in df.columns:
        atr_pct_25 = float((df['atr_14'] / df['close'] * 100).quantile(0.25))
    else:
        atr_pct_25 = 0.5

    # Feature importances
    lgb_imp = dict(zip(feature_cols, m_lgb.feature_importances_.tolist()))
    xgb_imp = dict(zip(feature_cols, m_xgb.feature_importances_.tolist()))
    combined_imp = {k: lgb_imp.get(k, 0) + xgb_imp.get(k, 0) for k in feature_cols}
    top_features = sorted(combined_imp.items(), key=lambda x: x[1], reverse=True)[:20]

    # 7. Save metadata
    metadata = {
        'model_type': 'regression_8h',
        'horizon_hours': 8,
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'n_training_samples': len(X),
        'data_range': f"{df.index[0]} to {df.index[-1]}",
        'trained_at': str(datetime.now()),
        'lgb_model_path': str(lgb_path),
        'xgb_model_path': str(xgb_path),
        'signal_thresholds': {
            'top_5pct': {'long': long_thr_5, 'short': short_thr_5},
            'top_10pct': {'long': long_thr_10, 'short': short_thr_10},
            'top_20pct': {'long': long_thr_20, 'short': short_thr_20},
        },
        'atr_pct_25_threshold': atr_pct_25,
        'top_20_features': top_features,
        'best_strategy_config': {
            'signal': 'top_5pct',
            'direction': 'both',
            'fee_tier': 'vip',
            'exit_mode': 'trailing',
            'trail_mult_atr': 0.75,
            'tp_mult_atr': 3.0,
            'sl_mult_atr': 1.5,
            'max_hold_bars': 12,
            'cooldown_bars': 0,
        },
        'backtest_results': {
            'profit_factor': 1.54,
            'win_rate': 0.470,
            'total_return_pct': 19.6,
            'max_drawdown_pct': 2.0,
            'sharpe_ratio': 4.06,
            'n_trades_5yr': 592,
            'avg_trade_pnl_usd': 8.27,
        },
        'prediction_stats': {
            'mean_pred': float(np.mean(preds)),
            'std_pred': float(np.std(preds)),
            'min_pred': float(np.min(preds)),
            'max_pred': float(np.max(preds)),
            'pct_positive': float((preds > 0).mean()),
        },
    }

    meta_path = MODEL_DIR / "reg_8h_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved: {meta_path}")

    print(f"\n  Top 10 features by importance:")
    for i, (feat, imp) in enumerate(top_features[:10]):
        print(f"    {i+1:2d}. {feat:<30s} {imp:.1f}")

    print(f"\n  Signal thresholds (top 5%): LONG >= {long_thr_5:.4f}, SHORT >= {short_thr_5:.4f}")
    print(f"  Prediction stats: mean={np.mean(preds):.6f}, std={np.std(preds):.6f}")
    print(f"  Positive predictions: {(preds > 0).mean():.1%}")

    print(f"\n  DONE. Models ready for live trading.")
    print(f"  Use scripts/generate_signals.py to generate live signals.")


if __name__ == "__main__":
    main()
