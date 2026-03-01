"""
Alpha Feature Engine -- statistically-grounded features designed for BTC perpetual futures.

These features capture microstructure, regime transitions, and statistical properties
that standard TA indicators miss. Each feature is strictly causal (no look-ahead).
"""
import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Statistical / Distributional features
# ---------------------------------------------------------------------------

def hurst_exponent(series: pd.Series, max_lag: int = 20) -> pd.Series:
    """
    Rolling Hurst exponent via rescaled-range analysis.

    H < 0.5  -- mean-reverting
    H ~ 0.5  -- random walk
    H > 0.5  -- trending

    Uses a 100-bar rolling window. Computed every 4th bar and forward-filled
    for speed (~10x faster than per-bar).
    """
    window = 100
    result = pd.Series(np.nan, index=series.index)
    values = series.values.astype(np.float64)
    lags = np.arange(2, max_lag + 1)
    log_lags = np.log(lags)

    for i in range(window, len(values), 4):  # every 4th bar
        x = values[i - window:i]
        if np.std(x) == 0:
            result.iloc[i] = np.nan
            continue
        tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
        if np.any(tau == 0):
            result.iloc[i] = np.nan
            continue
        log_tau = np.log(tau)
        poly = np.polyfit(log_lags, log_tau, 1)
        result.iloc[i] = poly[0]

    return result.ffill()


def rolling_autocorrelation(returns: pd.Series, lag: int = 1, window: int = 48) -> pd.Series:
    """
    Rolling autocorrelation of returns at a given lag.
    High positive autocorrelation = trending; negative = mean-reverting.
    Vectorized via rolling cov / rolling var.
    """
    r = returns
    r_lag = returns.shift(lag)
    cov = r.rolling(window, min_periods=window).cov(r_lag)
    var = r.rolling(window, min_periods=window).var()
    return cov / (var + 1e-10)


def realized_skewness(returns: pd.Series, window: int = 24) -> pd.Series:
    """Rolling skewness of returns -- tail asymmetry."""
    return returns.rolling(window, min_periods=window).skew()


def realized_kurtosis(returns: pd.Series, window: int = 24) -> pd.Series:
    """Rolling excess kurtosis -- fat tails indicator."""
    return returns.rolling(window, min_periods=window).kurt()


def garman_klass_volatility(high: pd.Series, low: pd.Series,
                            open_: pd.Series, close: pd.Series,
                            window: int = 24) -> pd.Series:
    """
    Garman-Klass volatility estimator -- more efficient than close-to-close vol.
    Uses the full OHLC range.
    """
    log_hl = (np.log(high / low)) ** 2
    log_co = (np.log(close / open_)) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return gk.rolling(window, min_periods=window).mean().apply(np.sqrt)


def return_entropy(returns: pd.Series, window: int = 48, n_bins: int = 10) -> pd.Series:
    """
    Shannon entropy of discretised returns.
    Low entropy = predictable (trending or stable)
    High entropy = random

    Computed every 4th bar and forward-filled for speed.
    """
    result = pd.Series(np.nan, index=returns.index)
    values = returns.values.astype(np.float64)

    for i in range(window, len(values), 4):
        x = values[i - window:i]
        if np.all(np.isnan(x)):
            continue
        x = x[~np.isnan(x)]
        if len(x) < 5:
            continue
        counts, _ = np.histogram(x, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        result.iloc[i] = -np.sum(probs * np.log2(probs))

    return result.ffill()


def volume_clock_returns(close: pd.Series, volume: pd.Series,
                         window: int = 24) -> pd.Series:
    """
    Volume-weighted return -- heavier weight to bars with more volume.
    Captures institutional activity better than equal-weight returns.
    """
    ret = np.log(close / close.shift(1))
    vol_weight = volume / volume.rolling(window, min_periods=1).sum()
    return (ret * vol_weight).rolling(window, min_periods=window).sum()


# ---------------------------------------------------------------------------
# 2. Funding-rate alpha (perpetual-futures specific)
# ---------------------------------------------------------------------------

def funding_cumulative(funding_rate: pd.Series, window: int = 24) -> pd.Series:
    """Cumulative funding over *window* hours -- the actual cost of holding."""
    return funding_rate.rolling(window, min_periods=1).sum()


def funding_oi_divergence(funding_rate: pd.Series,
                          open_interest: pd.Series,
                          window: int = 24) -> pd.Series:
    """
    Funding x OI divergence.
    When OI rises but funding drops = new shorts entering (bearish pressure).
    When OI rises and funding rises = new longs entering (bullish crowding).
    """
    fr_z = (funding_rate - funding_rate.rolling(window).mean()) / \
           (funding_rate.rolling(window).std() + 1e-10)
    oi_z = (open_interest - open_interest.rolling(window).mean()) / \
           (open_interest.rolling(window).std() + 1e-10)
    return fr_z * oi_z


def funding_mean_reversion_signal(funding_rate: pd.Series,
                                   window: int = 168) -> pd.Series:
    """
    Z-score of funding rate over *window* hours.
    Extreme positive funding = crowded longs, mean-revert down.
    Extreme negative funding = crowded shorts, mean-revert up.
    Signal is NEGATED so +1 = expected UP move (contrarian).
    """
    z = (funding_rate - funding_rate.rolling(window).mean()) / \
        (funding_rate.rolling(window).std() + 1e-10)
    return -z


# ---------------------------------------------------------------------------
# 3. Cross-timeframe / momentum features
# ---------------------------------------------------------------------------

def multi_horizon_momentum(close: pd.Series) -> pd.DataFrame:
    """
    Returns momentum at 1h, 4h, 12h, 24h, 72h, 168h (1w) and a
    momentum_alignment score (how many horizons agree on direction).
    """
    horizons = [1, 4, 12, 24, 72, 168]
    mom = pd.DataFrame(index=close.index)
    for h in horizons:
        mom[f'mom_{h}h'] = np.log(close / close.shift(h))

    # Alignment: fraction of horizons with same sign as 1h momentum
    signs = mom.apply(np.sign)
    mom['momentum_alignment'] = signs.apply(
        lambda row: (row == row.iloc[0]).sum() / len(row) if not np.isnan(row.iloc[0]) else np.nan,
        axis=1
    )
    return mom


def price_acceleration(close: pd.Series, window: int = 12) -> pd.Series:
    """
    Second derivative of log-price -- are we accelerating or decelerating?
    Positive -> momentum increasing, Negative -> momentum fading.
    """
    mom = np.log(close / close.shift(1))
    return mom.diff().rolling(window, min_periods=window).mean()


def breakout_intensity(close: pd.Series, high: pd.Series, low: pd.Series,
                       window: int = 48) -> pd.Series:
    """
    How far above/below the recent range are we?
    > 0 = breaking above range (bullish)
    < 0 = breaking below range (bearish)
    Normalised by ATR-like range.
    """
    range_high = high.rolling(window, min_periods=window).max()
    range_low = low.rolling(window, min_periods=window).min()
    range_width = range_high - range_low
    mid = (range_high + range_low) / 2
    return (close - mid) / (range_width + 1e-10)


# ---------------------------------------------------------------------------
# 4. Volume microstructure
# ---------------------------------------------------------------------------

def volume_imbalance(close: pd.Series, volume: pd.Series,
                     window: int = 24) -> pd.Series:
    """
    Buy-volume vs sell-volume imbalance based on close vs open proxy.
    """
    direction = np.sign(close.diff())
    buy_vol = (volume * (direction == 1)).rolling(window, min_periods=1).sum()
    sell_vol = (volume * (direction == -1)).rolling(window, min_periods=1).sum()
    total = buy_vol + sell_vol + 1e-10
    return (buy_vol - sell_vol) / total


def relative_volume_profile(volume: pd.Series, windows: list = None) -> pd.DataFrame:
    """
    Volume ratios across timeframes — spike detection at multiple scales.
    """
    if windows is None:
        windows = [4, 12, 24, 72]
    result = pd.DataFrame(index=volume.index)
    for w in windows:
        ma = volume.rolling(w, min_periods=1).mean()
        result[f'vol_ratio_{w}h'] = volume / (ma + 1e-10)
    return result


# ---------------------------------------------------------------------------
# 5. Regime transition features
# ---------------------------------------------------------------------------

def volatility_regime_score(returns: pd.Series,
                            short_window: int = 12,
                            long_window: int = 72) -> pd.Series:
    """
    Ratio of short-term to long-term volatility.
    > 1 = volatility expanding (regime change likely)
    < 1 = volatility contracting (consolidation)
    """
    short_vol = returns.rolling(short_window, min_periods=short_window).std()
    long_vol = returns.rolling(long_window, min_periods=long_window).std()
    return short_vol / (long_vol + 1e-10)


def trend_consistency(close: pd.Series, window: int = 24) -> pd.Series:
    """
    Fraction of bars in *window* that moved in the same direction as net move.
    High consistency = clean trend, Low = choppy.

    Vectorized: count positive diffs and compare to net direction.
    """
    rets = close.diff()
    pos_count = (rets > 0).rolling(window, min_periods=window).sum()
    neg_count = (rets < 0).rolling(window, min_periods=window).sum()
    total = pos_count + neg_count
    # Fraction of bars aligned with majority direction
    majority = pd.concat([pos_count, neg_count], axis=1).max(axis=1)
    return majority / (total + 1e-10)


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all alpha features to the DataFrame. Expects columns:
      open, high, low, close, volume
      Optionally: funding_rate, open_interest

    Returns a new DataFrame with additional columns. Does NOT drop NaN rows
    (caller decides warm-up policy).
    """
    out = df.copy()
    returns = np.log(out['close'] / out['close'].shift(1))

    print("🧪 Building alpha features...")

    # --- Statistical features ---
    print("   [1/7] Statistical features (Hurst, autocorrelation, skew, kurtosis, entropy)...")
    out['hurst_exp'] = hurst_exponent(out['close'])
    out['autocorr_1'] = rolling_autocorrelation(returns, lag=1, window=48)
    out['autocorr_5'] = rolling_autocorrelation(returns, lag=5, window=48)
    out['realized_skew_24'] = realized_skewness(returns, window=24)
    out['realized_kurt_24'] = realized_kurtosis(returns, window=24)
    out['gk_volatility'] = garman_klass_volatility(
        out['high'], out['low'], out['open'], out['close'], window=24
    )
    out['return_entropy_48'] = return_entropy(returns, window=48)
    out['vol_clock_ret'] = volume_clock_returns(out['close'], out['volume'])

    # --- Funding-rate alpha ---
    if 'funding_rate' in out.columns:
        print("   [2/7] Funding-rate alpha features...")
        out['funding_cum_8h'] = funding_cumulative(out['funding_rate'], 8)
        out['funding_cum_24h'] = funding_cumulative(out['funding_rate'], 24)
        out['funding_cum_72h'] = funding_cumulative(out['funding_rate'], 72)
        out['funding_mr_signal'] = funding_mean_reversion_signal(out['funding_rate'], 168)
        if 'open_interest' in out.columns:
            out['funding_oi_div'] = funding_oi_divergence(
                out['funding_rate'], out['open_interest'], 24
            )
    else:
        print("   [2/7] Skipping funding-rate features (column not present)")

    # --- Multi-horizon momentum ---
    print("   [3/7] Multi-horizon momentum & acceleration...")
    mom_df = multi_horizon_momentum(out['close'])
    for col in mom_df.columns:
        out[col] = mom_df[col]
    out['price_accel'] = price_acceleration(out['close'], window=12)
    out['breakout_intensity'] = breakout_intensity(
        out['close'], out['high'], out['low'], window=48
    )

    # --- Volume microstructure ---
    print("   [4/7] Volume microstructure...")
    out['volume_imbalance_24'] = volume_imbalance(out['close'], out['volume'], 24)
    vol_profile = relative_volume_profile(out['volume'])
    for col in vol_profile.columns:
        out[col] = vol_profile[col]

    # --- Regime transition ---
    print("   [5/7] Regime-transition features...")
    out['vol_regime_ratio'] = volatility_regime_score(returns)
    out['trend_consistency_24'] = trend_consistency(out['close'], 24)
    out['trend_consistency_72'] = trend_consistency(out['close'], 72)

    # --- Interaction features ---
    print("   [6/7] Interaction features (volume x momentum, volatility x trend)...")
    out['vol_x_momentum'] = out.get('volume_imbalance_24', 0) * returns.rolling(12).sum()
    out['gk_vol_x_hurst'] = out.get('gk_volatility', 0) * out.get('hurst_exp', 0.5)
    out['entropy_x_vol_regime'] = out.get('return_entropy_48', 0) * out.get('vol_regime_ratio', 1)

    # --- Lagged return features (avoid giving model raw close) ---
    print("   [7/7] Lagged return features...")
    for lag in [2, 3, 6, 12]:
        out[f'ret_lag_{lag}h'] = returns.shift(lag)

    # Replace inf with NaN (caller handles NaN cleanup)
    out = out.replace([np.inf, -np.inf], np.nan)

    n_new = len(out.columns) - len(df.columns)
    print(f"   [OK] Added {n_new} alpha features (total columns: {len(out.columns)})")
    return out
