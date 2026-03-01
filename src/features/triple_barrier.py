"""
Triple-Barrier Labeling for BTC perpetual futures.

Instead of binary "price goes up/down", labels are:
  1  = take-profit hit first    (high-conviction bullish)
 -1  = stop-loss hit first      (high-conviction bearish)
  0  = neither hit within max_holding_period (uncertain / low-conviction)

This produces cleaner labels because it captures the *path* of price, not
just the endpoint.  Combined with a binary wrapper (target = label > 0),
it gives the model a better signal to learn from.

Supports both LONG and SHORT directions for bidirectional trading.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def triple_barrier_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 24,
    atr_window: int = 14,
    volatility_col: Optional[pd.Series] = None,
    direction: str = "long",
) -> pd.DataFrame:
    """
    Compute triple-barrier labels for every bar.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    tp_mult : float
        Take-profit distance in multiples of ATR (default 2.0).
    sl_mult : float
        Stop-loss distance in multiples of ATR (default 1.0).
    max_holding : int
        Maximum bars to hold before timeout (default 24 = 1 day on 1h).
    atr_window : int
        ATR look-back for dynamic barrier sizing.
    volatility_col : pd.Series, optional
        If provided, use this as the volatility measure instead of ATR.
    direction : str
        "long" (default) or "short".  For shorts, TP is below entry and SL above.

    Returns
    -------
    pd.DataFrame with columns (prefixed by direction for short):
        tb_label      :  1 (TP hit), -1 (SL hit), 0 (timeout)
        tb_binary     :  1 if tb_label == 1, else 0  (for binary classifiers)
        tb_holding    :  number of bars until barrier hit
        tb_barrier_tp :  TP price level
        tb_barrier_sl :  SL price level
    """
    assert direction in ("long", "short"), f"direction must be 'long' or 'short', got '{direction}'"

    n = len(close)
    close_arr = close.values.astype(np.float64)
    high_arr = high.values.astype(np.float64)
    low_arr = low.values.astype(np.float64)

    # Compute ATR for barrier sizing
    if volatility_col is not None:
        atr = volatility_col.values.astype(np.float64)
    else:
        hl = high - low
        hc = np.abs(high - close.shift(1))
        lc = np.abs(low - close.shift(1))
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(atr_window, min_periods=atr_window).mean().values.astype(np.float64)

    labels = np.full(n, np.nan)
    holdings = np.full(n, np.nan)
    tp_levels = np.full(n, np.nan)
    sl_levels = np.full(n, np.nan)

    is_long = (direction == "long")

    for i in range(n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        entry = close_arr[i]

        if is_long:
            tp_price = entry + tp_mult * atr[i]
            sl_price = entry - sl_mult * atr[i]
        else:
            # SHORT: TP below entry, SL above entry
            tp_price = entry - tp_mult * atr[i]
            sl_price = entry + sl_mult * atr[i]

        tp_levels[i] = tp_price
        sl_levels[i] = sl_price

        label = 0  # default: timeout
        hold = max_holding

        end_idx = min(i + max_holding, n)
        for j in range(i + 1, end_idx):
            if is_long:
                # LONG: SL if low drops to sl_price, TP if high reaches tp_price
                if low_arr[j] <= sl_price:
                    label = -1
                    hold = j - i
                    break
                if high_arr[j] >= tp_price:
                    label = 1
                    hold = j - i
                    break
            else:
                # SHORT: SL if high rises to sl_price, TP if low drops to tp_price
                if high_arr[j] >= sl_price:
                    label = -1
                    hold = j - i
                    break
                if low_arr[j] <= tp_price:
                    label = 1
                    hold = j - i
                    break

        # If we're too close to end of series, mark as NaN
        if i + max_holding > n:
            label = np.nan
            hold = np.nan

        labels[i] = label
        holdings[i] = hold

    # Column prefix for short-side labels
    pfx = "" if is_long else "short_"

    result = pd.DataFrame(index=close.index)
    result[f'{pfx}tb_label'] = labels
    result[f'{pfx}tb_binary'] = (labels == 1).astype(float)
    result[f'{pfx}tb_binary'] = result[f'{pfx}tb_binary'].where(~np.isnan(labels), np.nan)
    result[f'{pfx}tb_holding'] = holdings
    result[f'{pfx}tb_barrier_tp'] = tp_levels
    result[f'{pfx}tb_barrier_sl'] = sl_levels

    # Stats
    valid = result[f'{pfx}tb_label'].dropna()
    dir_label = "LONG" if is_long else "SHORT"
    if len(valid) > 0:
        counts = valid.value_counts().sort_index()
        total = len(valid)
        print(f"[TARGET] Triple-barrier labeling ({dir_label}):")
        print(f"   SL hit (-1): {counts.get(-1, 0):>6,} ({counts.get(-1, 0)/total:.1%})")
        print(f"   Timeout (0): {counts.get(0, 0):>6,} ({counts.get(0, 0)/total:.1%})")
        print(f"   TP hit  (1): {counts.get(1, 0):>6,} ({counts.get(1, 0)/total:.1%})")
        print(f"   Avg holding : {result[f'{pfx}tb_holding'].mean():.1f} bars")

    return result


def build_dataset_with_triple_barrier(
    df: pd.DataFrame,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 24,
    feature_cols: Optional[list] = None,
    direction: str = "long",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Convenience function: add triple-barrier labels to a feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume + feature columns.
    tp_mult, sl_mult, max_holding : barrier parameters.
    feature_cols : list of column names to use as features.
                   If None, auto-detects numeric columns excluding OHLCV/target.
    direction : str
        "long" or "short" — determines barrier orientation.

    Returns
    -------
    X : pd.DataFrame of features
    y : pd.Series of binary labels (tb_binary)
    meta : pd.DataFrame with all triple-barrier columns for analysis
    """
    tb = triple_barrier_labels(
        df['close'], df['high'], df['low'],
        tp_mult=tp_mult, sl_mult=sl_mult,
        max_holding=max_holding,
        direction=direction,
    )

    pfx = "" if direction == "long" else "short_"

    # Merge
    combined = pd.concat([df, tb], axis=1)

    # Auto-select feature columns
    if feature_cols is None:
        exclude = {'open', 'high', 'low', 'close', 'volume', 'funding_rate',
                    'open_interest', 'timestamp',
                    'tb_label', 'tb_binary', 'tb_holding',
                    'tb_barrier_tp', 'tb_barrier_sl',
                    'short_tb_label', 'short_tb_binary', 'short_tb_holding',
                    'short_tb_barrier_tp', 'short_tb_barrier_sl'}
        feature_cols = [c for c in combined.columns
                        if c not in exclude
                        and combined[c].dtype in ('float64', 'int64', 'float32', 'int32')
                        and 'target' not in c.lower()
                        and 'future' not in c.lower()
                        and 'label' not in c.lower()]

    # Drop NaN
    target_col = f'{pfx}tb_binary'
    mask = combined[feature_cols + [target_col]].notna().all(axis=1)
    combined = combined[mask]

    X = combined[feature_cols]
    y = combined[target_col].astype(int)
    meta = combined[[f'{pfx}tb_label', f'{pfx}tb_binary', f'{pfx}tb_holding',
                     f'{pfx}tb_barrier_tp', f'{pfx}tb_barrier_sl']]

    dir_label = "LONG" if direction == "long" else "SHORT"
    print(f"   Dataset ({dir_label}): X={X.shape}, y={y.shape}, positive_rate={y.mean():.1%}")
    return X, y, meta
