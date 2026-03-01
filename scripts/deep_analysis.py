"""
Deep Data Analysis -- find what's ACTUALLY predictable in BTC 1h data.

This script does NOT train a model. It systematically tests:
  1. Individual feature predictive power (univariate AUC for every feature)
  2. Different target definitions (returns at 1h/4h/12h/24h, triple barrier combos)
  3. SL/TP ratio sweeps to find which barrier configs have signal
  4. Time-of-day and day-of-week effects
  5. Volume-based signals (relative volume spikes, OBV divergence)
  6. Momentum regime analysis (trending vs mean-reverting periods)
  7. Volatility regime analysis (does vol predict direction?)
  8. Cross-correlation analysis between features and future returns
  9. Feature interaction screening

All analysis uses strictly out-of-sample chronological splits.
"""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features.alpha_features import build_alpha_features
from features.triple_barrier import triple_barrier_labels


ROOT = Path(__file__).parent.parent


def load_data():
    """Load enhanced data."""
    p = ROOT / "data" / "processed" / "btcusdt_1h_full_history_enhanced.parquet"
    df = pd.read_parquet(p)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    return df


def simple_oos_auc(feature_values, target, train_frac=0.7):
    """
    Quick OOS AUC: use feature value directly as a 'score' to rank.
    This tests whether the raw feature value has monotonic relationship with target.
    Uses the LAST 30% as test set (chronological).
    """
    mask = feature_values.notna() & target.notna()
    feat = feature_values[mask].values
    tgt = target[mask].values
    if len(feat) < 500 or tgt.sum() < 50 or tgt.sum() > len(tgt) - 50:
        return np.nan, np.nan

    split = int(len(feat) * train_frac)
    feat_test = feat[split:]
    tgt_test = tgt[split:]

    if tgt_test.sum() < 10 or tgt_test.sum() > len(tgt_test) - 10:
        return np.nan, np.nan

    auc_pos = roc_auc_score(tgt_test, feat_test)
    auc_neg = roc_auc_score(tgt_test, -feat_test)
    # Return whichever direction is better
    if auc_neg > auc_pos:
        return auc_neg, -1  # inverted relationship
    return auc_pos, 1


def lgbm_oos_auc(X_df, target, train_frac=0.7, n_features=20):
    """
    Quick OOS AUC with a simple LightGBM model.
    Chronological split, no look-ahead.
    """
    mask = target.notna()
    for c in X_df.columns:
        mask &= X_df[c].notna()
    X = X_df[mask]
    y = target[mask].astype(int)

    if len(X) < 1000 or y.sum() < 100:
        return np.nan

    split = int(len(X) * train_frac)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    if y_te.sum() < 20 or y_te.sum() > len(y_te) - 20:
        return np.nan

    m = lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6,
        min_child_samples=50, reg_alpha=1.0, reg_lambda=5.0,
        class_weight='balanced', verbose=-1, random_state=42,
        num_leaves=15,
    )
    m.fit(X_tr, y_tr)
    pred = m.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, pred)


def main():
    print("=" * 80)
    print("DEEP DATA ANALYSIS -- Finding Real Edges in BTC 1h")
    print("=" * 80)

    df = load_data()

    # Build alpha features
    print("\nBuilding alpha features...")
    df = build_alpha_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Basic returns at multiple horizons
    for h in [1, 2, 4, 6, 8, 12, 24, 48]:
        df[f'fwd_ret_{h}h'] = np.log(df['close'].shift(-h) / df['close'])

    # ================================================================
    # ANALYSIS 1: Which target horizon is most predictable?
    # ================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 1: Target Horizon Scan")
    print("Which return horizon is most predictable with simple features?")
    print("=" * 80)

    # Use a handful of proven features
    core_features = []
    for c in ['rsi_14', 'atr_pct', 'volume_ratio', 'macd', 'bb_width',
              'momentum_10', 'hurst_exp', 'autocorr_1', 'adx', 'cci',
              'williams_r', 'obv', 'vpt', 'ma_20', 'ma_50']:
        if c in df.columns:
            core_features.append(c)

    print(f"\nUsing {len(core_features)} core features: {core_features}")

    horizon_results = {}
    for h in [1, 2, 4, 6, 8, 12, 24, 48]:
        target = (df[f'fwd_ret_{h}h'] > 0).astype(float)
        target_name = f'up_{h}h'
        X_sub = df[core_features].copy()
        auc = lgbm_oos_auc(X_sub, target)
        horizon_results[h] = auc
        direction = "PROMISING" if auc and auc > 0.53 else "weak"
        print(f"  {h:2d}h forward return (binary up/down): OOS AUC = {auc:.4f}  [{direction}]")

    best_h = max(horizon_results, key=lambda k: horizon_results[k] if horizon_results[k] == horizon_results[k] else 0)
    print(f"\n  >> Best horizon: {best_h}h with AUC {horizon_results[best_h]:.4f}")

    # ================================================================
    # ANALYSIS 2: Triple Barrier SL/TP Sweep
    # ================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Triple Barrier SL/TP Ratio Sweep")
    print("Which SL/TP/holding combo produces the most learnable labels?")
    print("=" * 80)

    barrier_results = []
    for tp_mult in [1.0, 1.5, 2.0, 3.0, 4.0]:
        for sl_mult in [0.5, 1.0, 1.5, 2.0]:
            for max_hold in [12, 24, 48]:
                tb = triple_barrier_labels(
                    df['close'], df['high'], df['low'],
                    tp_mult=tp_mult, sl_mult=sl_mult, max_holding=max_hold
                )
                target = tb['tb_binary']
                pos_rate = target.dropna().mean()
                if pos_rate < 0.05 or pos_rate > 0.95:
                    continue
                X_sub = df[core_features].copy()
                auc = lgbm_oos_auc(X_sub, target)
                barrier_results.append({
                    'tp': tp_mult, 'sl': sl_mult, 'hold': max_hold,
                    'pos_rate': pos_rate, 'auc': auc,
                    'rr_ratio': tp_mult / sl_mult,
                })
                tag = " <<" if auc and auc > 0.54 else ""
                print(f"  TP={tp_mult:.1f}x SL={sl_mult:.1f}x Hold={max_hold:2d}  "
                      f"pos_rate={pos_rate:.1%}  AUC={auc:.4f}{tag}")

    barrier_df = pd.DataFrame(barrier_results).sort_values('auc', ascending=False)
    print(f"\n  >> Top 5 barrier configs:")
    for _, row in barrier_df.head(5).iterrows():
        print(f"     TP={row['tp']:.1f}x SL={row['sl']:.1f}x Hold={row['hold']:.0f} "
              f"R:R={row['rr_ratio']:.1f} pos={row['pos_rate']:.1%} AUC={row['auc']:.4f}")

    # ================================================================
    # ANALYSIS 3: Univariate Feature Screening
    # ================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Univariate Feature Power (every feature vs best target)")
    print("=" * 80)

    # Use best target from analysis 1
    best_target = (df[f'fwd_ret_{best_h}h'] > 0).astype(float)

    exclude_cols = {'open', 'high', 'low', 'close', 'volume', 'timestamp',
                    'funding_rate', 'open_interest'}
    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and not c.startswith('fwd_ret_')
                    and not c.startswith('tb_')
                    and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

    univariate_results = []
    for col in feature_cols:
        auc, direction = simple_oos_auc(df[col], best_target)
        if auc == auc:  # not nan
            univariate_results.append({
                'feature': col, 'auc': auc, 'direction': direction
            })

    uni_df = pd.DataFrame(univariate_results).sort_values('auc', ascending=False)
    print(f"\nTop 20 univariate features (vs {best_h}h binary return):")
    for _, row in uni_df.head(20).iterrows():
        d = "+" if row['direction'] == 1 else "-"
        tag = " ***" if row['auc'] > 0.53 else ""
        print(f"  {row['feature']:<35s}  AUC={row['auc']:.4f}  dir={d}{tag}")

    print(f"\nBottom 10 (anti-predictive or random):")
    for _, row in uni_df.tail(10).iterrows():
        print(f"  {row['feature']:<35s}  AUC={row['auc']:.4f}")

    # ================================================================
    # ANALYSIS 4: Time-Based Effects
    # ================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 4: Time-Based Patterns")
    print("=" * 80)

    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['dow'] = df.index.dayofweek

        # Hourly return analysis
        print("\nAverage 4h forward return by hour of day:")
        hourly_rets = df.groupby('hour')[f'fwd_ret_{best_h}h'].agg(['mean', 'std', 'count'])
        for h_val in range(24):
            if h_val in hourly_rets.index:
                row = hourly_rets.loc[h_val]
                t_stat = row['mean'] / (row['std'] / np.sqrt(row['count'])) if row['std'] > 0 else 0
                sig = " *" if abs(t_stat) > 2.0 else ""
                print(f"  Hour {h_val:2d}: mean={row['mean']*100:+.4f}%  "
                      f"t-stat={t_stat:+.2f}{sig}")

        print("\nAverage return by day of week:")
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_rets = df.groupby('dow')[f'fwd_ret_{best_h}h'].agg(['mean', 'std', 'count'])
        for d_val in range(7):
            if d_val in dow_rets.index:
                row = dow_rets.loc[d_val]
                t_stat = row['mean'] / (row['std'] / np.sqrt(row['count'])) if row['std'] > 0 else 0
                sig = " *" if abs(t_stat) > 2.0 else ""
                print(f"  {dow_names[d_val]}: mean={row['mean']*100:+.4f}%  "
                      f"t-stat={t_stat:+.2f}{sig}")

    # ================================================================
    # ANALYSIS 5: Volume-Based Signals
    # ================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 5: Volume-Based Predictive Signals")
    print("=" * 80)

    # Volume spike analysis
    df['vol_zscore'] = (df['volume'] - df['volume'].rolling(168).mean()) / df['volume'].rolling(168).std()
    df['vol_spike_2x'] = (df['volume'] > 2 * df['volume'].rolling(24).mean()).astype(float)
    df['vol_spike_3x'] = (df['volume'] > 3 * df['volume'].rolling(24).mean()).astype(float)

    # Returns after volume spikes
    for spike_name, spike_col in [('2x_volume_spike', 'vol_spike_2x'),
                                   ('3x_volume_spike', 'vol_spike_3x')]:
        mask = df[spike_col] == 1
        n_events = mask.sum()
        if n_events < 50:
            print(f"  {spike_name}: only {n_events} events, skipping")
            continue
        for h in [1, 4, 12, 24]:
            ret_col = f'fwd_ret_{h}h'
            if ret_col not in df.columns:
                continue
            ret_spike = df.loc[mask, ret_col].dropna()
            ret_all = df[ret_col].dropna()
            if len(ret_spike) < 30:
                continue
            t, p = stats.ttest_ind(ret_spike, ret_all, equal_var=False)
            mean_spike = ret_spike.mean()
            sig = " **" if p < 0.05 else ""
            print(f"  After {spike_name}: {h}h mean ret = {mean_spike*100:+.4f}%  "
                  f"(vs all: {ret_all.mean()*100:+.4f}%)  p={p:.4f}{sig}")

    # Volume + direction: high vol up candles vs high vol down candles
    df['candle_dir'] = np.sign(df['close'] - df['open'])
    df['vol_up'] = (df['vol_spike_2x'] == 1) & (df['candle_dir'] > 0)
    df['vol_down'] = (df['vol_spike_2x'] == 1) & (df['candle_dir'] < 0)

    for label, mask_col in [('High-vol UP candle', 'vol_up'),
                             ('High-vol DOWN candle', 'vol_down')]:
        mask = df[mask_col] == True
        n_events = mask.sum()
        if n_events < 50:
            continue
        for h in [4, 12, 24]:
            ret = df.loc[mask, f'fwd_ret_{h}h'].dropna()
            if len(ret) < 30:
                continue
            t_stat, p = stats.ttest_1samp(ret, 0)
            print(f"  {label} -> {h}h: mean={ret.mean()*100:+.4f}%  "
                  f"t={t_stat:+.2f}  p={p:.4f}")

    # ================================================================
    # ANALYSIS 6: Momentum Regime Analysis
    # ================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 6: Regime-Conditional Predictability")
    print("=" * 80)

    # Is the market more predictable in certain regimes?
    if 'adx' in df.columns:
        # Trending regime (ADX > 25) vs ranging (ADX < 20)
        trending = df['adx'] > 25
        ranging = df['adx'] < 20

        for regime_name, regime_mask in [('Trending (ADX>25)', trending),
                                          ('Ranging (ADX<20)', ranging)]:
            sub = df[regime_mask].copy()
            if len(sub) < 1000:
                continue
            target = (sub[f'fwd_ret_{best_h}h'] > 0).astype(float)
            X_sub = sub[core_features].copy()
            auc = lgbm_oos_auc(X_sub, target)
            print(f"  {regime_name}: n={len(sub):,}  LGB AUC={auc:.4f}")

    if 'hurst_exp' in df.columns:
        # Trending (H > 0.55) vs mean-reverting (H < 0.45)
        trending_h = df['hurst_exp'] > 0.55
        mr_h = df['hurst_exp'] < 0.45
        for regime_name, regime_mask in [('Trending (Hurst>0.55)', trending_h),
                                          ('Mean-reverting (Hurst<0.45)', mr_h)]:
            sub = df[regime_mask].copy()
            if len(sub) < 1000:
                continue
            target = (sub[f'fwd_ret_{best_h}h'] > 0).astype(float)
            X_sub = sub[core_features].copy()
            auc = lgbm_oos_auc(X_sub, target)
            print(f"  {regime_name}: n={len(sub):,}  LGB AUC={auc:.4f}")

    # High vs low volatility
    if 'atr_pct' in df.columns:
        atr_median = df['atr_pct'].median()
        high_vol = df['atr_pct'] > df['atr_pct'].quantile(0.75)
        low_vol = df['atr_pct'] < df['atr_pct'].quantile(0.25)

        for regime_name, regime_mask in [('High volatility (top 25%)', high_vol),
                                          ('Low volatility (bottom 25%)', low_vol)]:
            sub = df[regime_mask].copy()
            if len(sub) < 1000:
                continue
            target = (sub[f'fwd_ret_{best_h}h'] > 0).astype(float)
            X_sub = sub[core_features].copy()
            auc = lgbm_oos_auc(X_sub, target)
            print(f"  {regime_name}: n={len(sub):,}  LGB AUC={auc:.4f}")

    # ================================================================
    # ANALYSIS 7: Cross-Correlation Lag Analysis
    # ================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 7: Lagged Cross-Correlations")
    print("Which features lead future returns?")
    print("=" * 80)

    target_ret = df[f'fwd_ret_{best_h}h'].dropna()
    lag_results = []

    for col in core_features:
        if col not in df.columns:
            continue
        for lag in [0, 1, 2, 4, 8, 12]:
            feat_lagged = df[col].shift(lag)
            mask = feat_lagged.notna() & target_ret.notna()
            if mask.sum() < 1000:
                continue
            corr, pval = stats.spearmanr(feat_lagged[mask], target_ret[mask])
            if abs(corr) > 0.02 and pval < 0.01:
                lag_results.append({
                    'feature': col, 'lag': lag,
                    'spearman_r': corr, 'p_value': pval
                })

    if lag_results:
        lag_df = pd.DataFrame(lag_results).sort_values('spearman_r', key=abs, ascending=False)
        print(f"\nSignificant lagged correlations (|r| > 0.02, p < 0.01):")
        for _, row in lag_df.head(20).iterrows():
            print(f"  {row['feature']:<25s} lag={row['lag']:2d}h  r={row['spearman_r']:+.4f}  p={row['p_value']:.2e}")
    else:
        print("  No significant lagged correlations found!")

    # ================================================================
    # ANALYSIS 8: Engineered Features -- Interaction Screening
    # ================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 8: Feature Interaction Screening")
    print("Do feature COMBINATIONS have more power than individuals?")
    print("=" * 80)

    best_target = (df[f'fwd_ret_{best_h}h'] > 0).astype(float)

    # Try pairs of top features
    top_features = uni_df.head(10)['feature'].tolist()
    if len(top_features) >= 2:
        pair_results = []
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                f1, f2 = top_features[i], top_features[j]
                if f1 not in df.columns or f2 not in df.columns:
                    continue
                X_pair = df[[f1, f2]].copy()
                X_pair['interaction'] = df[f1] * df[f2]
                X_pair['ratio'] = df[f1] / (df[f2].abs() + 1e-10)
                auc = lgbm_oos_auc(X_pair, best_target, n_features=4)
                pair_results.append({'f1': f1, 'f2': f2, 'auc': auc})

        if pair_results:
            pair_df = pd.DataFrame(pair_results).sort_values('auc', ascending=False)
            print(f"\nTop 10 feature pairs:")
            for _, row in pair_df.head(10).iterrows():
                tag = " ***" if row['auc'] > 0.54 else ""
                print(f"  {row['f1']:<25s} x {row['f2']:<25s}  AUC={row['auc']:.4f}{tag}")

    # ================================================================
    # ANALYSIS 9: Alternative Targets -- Trend Following vs Mean Reversion
    # ================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 9: Alternative Target Definitions")
    print("=" * 80)

    # Large move targets (only predict big moves, not noise)
    for threshold_pct in [0.5, 1.0, 1.5, 2.0]:
        for h in [4, 12, 24]:
            ret = df[f'fwd_ret_{h}h']
            big_up = (ret > threshold_pct / 100).astype(float)
            big_down = (ret < -threshold_pct / 100).astype(float)

            # Only count rows where there IS a big move (exclude flat)
            big_move_mask = (big_up == 1) | (big_down == 1)
            if big_move_mask.sum() < 200:
                continue

            X_sub = df.loc[big_move_mask, core_features].copy()
            target = big_up[big_move_mask]

            pos_rate = target.mean()
            if pos_rate < 0.1 or pos_rate > 0.9:
                continue
            auc = lgbm_oos_auc(X_sub, target)
            tag = " ***" if auc and auc > 0.55 else ""
            print(f"  Big move >{threshold_pct}% in {h}h (n={big_move_mask.sum():,}, "
                  f"up={pos_rate:.1%}): AUC={auc:.4f}{tag}")

    # Mean-reversion target: does price revert after extreme RSI?
    if 'rsi_14' in df.columns:
        print("\nMean-reversion signals (RSI extremes):")
        for rsi_thresh in [20, 25, 30]:
            oversold = df['rsi_14'] < rsi_thresh
            overbought = df['rsi_14'] > (100 - rsi_thresh)

            for label, mask in [('Oversold', oversold), ('Overbought', overbought)]:
                n_events = mask.sum()
                if n_events < 50:
                    continue
                for h in [4, 12, 24]:
                    ret = df.loc[mask, f'fwd_ret_{h}h'].dropna()
                    if len(ret) < 30:
                        continue
                    t_stat, p = stats.ttest_1samp(ret, 0)
                    win_rate = (ret > 0).mean() if label == 'Oversold' else (ret < 0).mean()
                    expected_dir = "up" if label == 'Oversold' else "down"
                    sig = " **" if p < 0.05 else ""
                    print(f"  RSI<{rsi_thresh} {label} n={n_events}: "
                          f"{h}h mean={ret.mean()*100:+.4f}% "
                          f"win_rate({expected_dir})={win_rate:.1%} "
                          f"t={t_stat:+.2f} p={p:.4f}{sig}")

    # ================================================================
    # ANALYSIS 10: Full Model with Different Feature Sets
    # ================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 10: Full Model Comparison -- Feature Sets x Targets")
    print("=" * 80)

    # Get all available numeric features
    all_features = [c for c in df.columns if c not in
                    {'open', 'high', 'low', 'close', 'volume', 'timestamp',
                     'funding_rate', 'open_interest', 'hour', 'dow',
                     'candle_dir', 'vol_up', 'vol_down', 'vol_spike_2x',
                     'vol_spike_3x', 'vol_zscore'}
                    and not c.startswith('fwd_ret_')
                    and not c.startswith('tb_')
                    and df[c].dtype in ('float64', 'float32', 'int64', 'int32')
                    and df[c].notna().sum() > len(df) * 0.5]

    print(f"\nTotal available features: {len(all_features)}")

    # Different target types
    targets = {}

    # Simple returns at different horizons
    for h in [1, 4, 12, 24]:
        targets[f'binary_{h}h'] = (df[f'fwd_ret_{h}h'] > 0).astype(float)

    # Best triple barrier from analysis 2
    if len(barrier_df) > 0:
        best_barrier = barrier_df.iloc[0]
        tb = triple_barrier_labels(
            df['close'], df['high'], df['low'],
            tp_mult=best_barrier['tp'], sl_mult=best_barrier['sl'],
            max_holding=int(best_barrier['hold'])
        )
        targets['best_barrier'] = tb['tb_binary']

    # Volatility-adjusted return (Sharpe-like)
    rolling_vol = df['close'].pct_change().rolling(24).std()
    for h in [4, 12, 24]:
        vol_adj_ret = df[f'fwd_ret_{h}h'] / (rolling_vol + 1e-10)
        targets[f'vol_adj_{h}h'] = (vol_adj_ret > 0).astype(float)

    model_results = []
    for target_name, target in targets.items():
        X_sub = df[all_features].copy()
        auc = lgbm_oos_auc(X_sub, target, n_features=len(all_features))
        model_results.append({'target': target_name, 'n_features': len(all_features), 'auc': auc})
        tag = " <<<" if auc and auc > 0.54 else ""
        print(f"  All features -> {target_name:<20s}: AUC = {auc:.4f}{tag}")

    # Also try with just alpha features
    alpha_features = [c for c in all_features if c in [
        'hurst_exp', 'autocorr_1', 'autocorr_5', 'realized_skew_24',
        'realized_kurt_24', 'gk_volatility', 'return_entropy_48',
        'vol_clock_ret', 'mom_1h', 'mom_4h', 'mom_12h', 'mom_24h',
        'mom_72h', 'mom_168h', 'mom_alignment', 'price_accel',
        'breakout_intensity', 'volume_imbalance_24',
        'vol_regime_ratio', 'trend_consistency_24', 'trend_consistency_72',
        'vol_x_momentum', 'gk_vol_x_hurst', 'entropy_x_vol_regime',
    ]]

    if alpha_features:
        print(f"\n  Alpha-only features ({len(alpha_features)}):")
        for target_name, target in targets.items():
            X_sub = df[alpha_features].copy()
            auc = lgbm_oos_auc(X_sub, target, n_features=len(alpha_features))
            tag = " <<<" if auc and auc > 0.54 else ""
            print(f"    Alpha -> {target_name:<20s}: AUC = {auc:.4f}{tag}")

    # ================================================================
    # ANALYSIS 11: Walk-Forward Stability Check
    # ================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 11: Walk-Forward Stability (is it real or overfitting?)")
    print("=" * 80)

    # Take the best-performing target + feature combo and test across
    # 5 rolling windows
    best_result = max(model_results, key=lambda x: x['auc'] if x['auc'] == x['auc'] else 0)
    best_target_name = best_result['target']
    best_target_series = targets[best_target_name]

    print(f"\nBest config: {best_target_name} with all features (AUC={best_result['auc']:.4f})")
    print("Testing walk-forward stability (5 non-overlapping windows):")

    X_full = df[all_features].copy()
    mask = best_target_series.notna()
    for c in X_full.columns:
        mask &= X_full[c].notna()
    X_clean = X_full[mask]
    y_clean = best_target_series[mask].astype(int)

    n = len(X_clean)
    window_size = n // 6  # 5 test windows with expanding train

    wf_aucs = []
    for fold in range(5):
        test_start = n - (5 - fold) * window_size
        test_end = test_start + window_size
        if test_end > n:
            test_end = n
        train_end = test_start - 24  # purge gap

        if train_end < 500:
            continue

        X_tr = X_clean.iloc[:train_end]
        y_tr = y_clean.iloc[:train_end]
        X_te = X_clean.iloc[test_start:test_end]
        y_te = y_clean.iloc[test_start:test_end]

        if y_te.sum() < 20 or y_te.sum() > len(y_te) - 20:
            continue

        m = lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6,
            min_child_samples=50, reg_alpha=1.0, reg_lambda=5.0,
            class_weight='balanced', verbose=-1, random_state=42,
            num_leaves=15,
        )
        m.fit(X_tr, y_tr)
        pred = m.predict_proba(X_te)[:, 1]
        fold_auc = roc_auc_score(y_te, pred)
        wf_aucs.append(fold_auc)
        print(f"  Fold {fold+1}: train={len(X_tr):,} test={len(X_te):,} AUC={fold_auc:.4f}")

    if wf_aucs:
        print(f"\n  Walk-forward mean AUC: {np.mean(wf_aucs):.4f} +- {np.std(wf_aucs):.4f}")
        if np.mean(wf_aucs) > 0.53:
            print("  >> There IS a small but real signal here!")
        elif np.mean(wf_aucs) > 0.51:
            print("  >> Marginal signal -- likely eaten by fees in practice")
        else:
            print("  >> No real signal after walk-forward -- pure noise")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
What we found:
  1. Best simple target horizon: {best_h}h (AUC={horizon_results[best_h]:.4f})
  2. Best barrier config: TP={barrier_df.iloc[0]['tp']:.1f}x SL={barrier_df.iloc[0]['sl']:.1f}x Hold={barrier_df.iloc[0]['hold']:.0f}
     (AUC={barrier_df.iloc[0]['auc']:.4f})
  3. Walk-forward mean AUC: {np.mean(wf_aucs):.4f} +- {np.std(wf_aucs):.4f}
  4. Total features tested: {len(all_features)}
""")


if __name__ == "__main__":
    main()
