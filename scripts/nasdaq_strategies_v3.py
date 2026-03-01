"""
NASDAQ Strategy V3 — Extra Profitability Improvements
=======================================================
Building on V2's findings, this tests NEW ideas not yet explored:

NEW IMPROVEMENTS:
  N1.  FOMC Day Effect — Markets tend to rally into/after Fed meetings
  N2.  Month-Start Effect — First 3 trading days of month (fund flows)
  N3.  Month-End Effect — Last 3 trading days of month (window dressing)
  N4.  Earnings Season Filter — Skip volatile periods (Jan/Apr/Jul/Oct)
  N5.  Consecutive Down + Gap Down — Double panic = stronger bounce
  N6.  Volume Confirmation — Only trade when yesterday's volume was above average
  N7.  Close-to-Close vs Open-to-Close — Hold overnight if bounce day is green
  N8.  ATR Percentile Sizing — Low ATR days → more size, high ATR → less
  N9.  Regime Stacking — Mean reversion + day-of-week on same day = extra size
  N10. Walk-Forward Validation — Train on first 7 years, test on last 3
  N11. Thursday Pre-Bounce — Buy Thursday close if 2+ down, sell Friday close
  N12. VIX Mean Reversion Combo — High VIX (20-25) + down streak = panic selling overdone
  N13. SMA200 Distance — Further below SMA200 = bigger bounce potential
  N14. Weekly Close Effect — If week closes red, Monday bounces harder
  N15. Asymmetric SL/TP — Tight stop (0.5%) with wide target (2%) for runners

Capital: $25,000 | Instrument: QQQ
"""

import sys, os, warnings, json
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats

LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'nasdaq_v3_results.txt')

class TeeWriter:
    def __init__(self, filepath):
        self._stdout = sys.stdout
        self._file = open(filepath, 'w', encoding='utf-8')
    def write(self, text):
        self._stdout.write(text)
        self._file.write(text)
        self._file.flush()
    def flush(self):
        self._stdout.flush()
        self._file.flush()

sys.stdout = TeeWriter(LOG_FILE)

def header(title):
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}\n")

# ═════════════════════════════════════════════════════════════════════════════
# DATA
# ═════════════════════════════════════════════════════════════════════════════
header("NASDAQ V3 — EXTRA PROFITABILITY IMPROVEMENTS")
print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

print("\n[1] QQQ daily (10yr)...")
qqq_daily = yf.download("QQQ", interval="1d", period="10y", progress=False)
if isinstance(qqq_daily.columns, pd.MultiIndex):
    qqq_daily.columns = qqq_daily.columns.get_level_values(0)
print(f"    {len(qqq_daily)} bars")

print("[2] VIX daily (10yr)...")
vix_daily = yf.download("^VIX", interval="1d", period="10y", progress=False)
if isinstance(vix_daily.columns, pd.MultiIndex):
    vix_daily.columns = vix_daily.columns.get_level_values(0)
print(f"    {len(vix_daily)} bars")
print("  Done.\n")

# ═════════════════════════════════════════════════════════════════════════════
# PREPARE DATA
# ═════════════════════════════════════════════════════════════════════════════
daily = qqq_daily.copy()
daily['prev_close'] = daily['Close'].shift(1)
daily['gap_pct'] = (daily['Open'] - daily['prev_close']) / daily['prev_close'] * 100
daily['intraday_ret_pct'] = (daily['Close'] - daily['Open']) / daily['Open'] * 100
daily['full_ret_pct'] = daily['Close'].pct_change() * 100
daily['dow'] = daily.index.dayofweek
daily['month'] = daily.index.month
daily['day_of_month'] = daily.index.day
daily = daily.dropna()

# VIX
vix_close = vix_daily['Close'].copy()
vix_close.index = pd.to_datetime(vix_close.index)
if hasattr(vix_close.index, 'tz') and vix_close.index.tz is not None:
    vix_close.index = vix_close.index.tz_localize(None)
daily_idx = pd.to_datetime(daily.index)
if hasattr(daily_idx, 'tz') and daily_idx.tz is not None:
    daily_idx = daily_idx.tz_localize(None)
daily.index = daily_idx
daily['vix'] = vix_close.reindex(daily.index, method='ffill')

# Technical
daily['sma50'] = daily['Close'].rolling(50).mean()
daily['sma200'] = daily['Close'].rolling(200).mean()
daily['atr'] = pd.concat([
    daily['High'] - daily['Low'],
    (daily['High'] - daily['Close'].shift(1)).abs(),
    (daily['Low'] - daily['Close'].shift(1)).abs()
], axis=1).max(axis=1).rolling(14).mean()
daily['atr_pct'] = daily['atr'] / daily['Close'] * 100
daily['atr_percentile'] = daily['atr_pct'].rolling(252).rank(pct=True)
daily['vol_20d'] = daily['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
daily['volume_ma20'] = daily['Volume'].rolling(20).mean()
daily['volume_ratio'] = daily['Volume'] / daily['volume_ma20']

# Streak
streak = 0
streaks = []
cum_drop = 0.0
drops = []
for i in range(len(daily)):
    ret = daily.iloc[i]['full_ret_pct']
    if ret < 0:
        streak = streak - 1 if streak < 0 else -1
        cum_drop += ret
    else:
        streak = streak + 1 if streak > 0 else 1
        cum_drop = 0.0
    streaks.append(streak)
    drops.append(cum_drop)
daily['streak'] = streaks
daily['prev_streak'] = daily['streak'].shift(1).fillna(0)
daily['streak_drop'] = drops
daily['prev_streak_drop'] = daily['streak_drop'].shift(1).fillna(0)

# Week tracking (for weekly close effect)
daily['week'] = daily.index.isocalendar().week.values
daily['year_week'] = daily.index.year * 100 + daily['week']

# Trading day of month (rank within month)
daily['trading_day'] = daily.groupby([daily.index.year, daily.index.month]).cumcount() + 1
# Reverse trading day (days until month end)
daily['trading_days_left'] = daily.groupby([daily.index.year, daily.index.month]).cumcount(ascending=False) + 1

# SMA200 distance
daily['sma200_dist'] = (daily['Close'] - daily['sma200']) / daily['sma200'] * 100

print(f"  Data: {len(daily)} days, {daily.index[0].strftime('%Y-%m-%d')} → {daily.index[-1].strftime('%Y-%m-%d')}")

# ═════════════════════════════════════════════════════════════════════════════
# BUY AND HOLD
# ═════════════════════════════════════════════════════════════════════════════
bh_start = daily.iloc[0]['Close']
bh_end = daily.iloc[-1]['Close']
bh_years = (daily.index[-1] - daily.index[0]).days / 365.25
bh_total = (bh_end - bh_start) / bh_start * 100
bh_cagr = ((bh_end / bh_start) ** (1 / bh_years) - 1) * 100
bh_sharpe = (daily['full_ret_pct'].mean() / daily['full_ret_pct'].std()) * np.sqrt(252)
bh_dd = ((daily['Close'].cummax() - daily['Close']) / daily['Close'].cummax() * 100).max()

print(f"  B&H: {bh_total:+.0f}% ({bh_cagr:+.1f}% CAGR), Sharpe={bh_sharpe:.2f}, DD={bh_dd:.0f}%\n")


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════════════
CAPITAL = 25000.0
SLIPPAGE = 0.01

def backtest(trades_df, name, capital=CAPITAL):
    if len(trades_df) == 0:
        return None
    results = []
    equity = capital
    peak = capital
    max_dd_pct = 0
    for _, t in trades_df.iterrows():
        entry, exit_p = t['entry_price'], t['exit_price']
        d = t.get('direction', 'long')
        size = t.get('size_pct', 5.0)
        if d == 'long':
            pnl_pct = (exit_p * (1 - SLIPPAGE/100) - entry * (1 + SLIPPAGE/100)) / (entry * (1 + SLIPPAGE/100)) * 100
        else:
            pnl_pct = (entry * (1 - SLIPPAGE/100) - exit_p * (1 + SLIPPAGE/100)) / (entry * (1 - SLIPPAGE/100)) * 100
        pnl_dollar = equity * size / 100 * pnl_pct / 100
        equity += pnl_dollar
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd_pct = max(max_dd_pct, dd)
        results.append({'date': t['date'], 'pnl_pct': pnl_pct, 'pnl_dollar': pnl_dollar, 'equity': equity})
    
    res = pd.DataFrame(results)
    wins = res[res['pnl_pct'] > 0]
    losses = res[res['pnl_pct'] <= 0]
    wr = len(wins) / len(res) * 100
    gp = wins['pnl_dollar'].sum() if len(wins) > 0 else 0
    gl = abs(losses['pnl_dollar'].sum()) if len(losses) > 0 else 0
    pf = gp / gl if gl > 0 else float('inf')
    sharpe = (res['pnl_pct'].mean() / res['pnl_pct'].std()) * np.sqrt(252) if res['pnl_pct'].std() > 0 else 0
    t_stat, p_val = stats.ttest_1samp(res['pnl_pct'], 0) if len(res) > 2 else (0, 1)
    first = pd.to_datetime(res['date'].iloc[0])
    last = pd.to_datetime(res['date'].iloc[-1])
    years = max((last - first).days / 365.25, 0.1)
    
    return {
        'name': name, 'trades': len(res), 'trades_per_year': len(res)/years, 'years': years,
        'win_rate': wr,
        'avg_win': wins['pnl_pct'].mean() if len(wins) > 0 else 0,
        'avg_loss': abs(losses['pnl_pct'].mean()) if len(losses) > 0 else 0,
        'profit_factor': pf, 'total_pnl': res['pnl_dollar'].sum(),
        'total_pnl_pct': (equity - capital) / capital * 100,
        'max_dd_pct': max_dd_pct, 'sharpe': sharpe,
        'calmar': ((equity-capital)/capital*100/years) / max_dd_pct if max_dd_pct > 0 else float('inf'),
        'expectancy': res['pnl_pct'].mean(),
        't_stat': t_stat, 'p_value': p_val, 'final_equity': equity,
    }

def pr(r):
    if r is None:
        print("  NO TRADES\n"); return
    sig = "***" if r['p_value'] < 0.01 else "** " if r['p_value'] < 0.05 else "*  " if r['p_value'] < 0.10 else "   "
    print(f"  Trades: {r['trades']:,d} ({r['trades_per_year']:.0f}/yr, {r['years']:.1f}yr)  WR: {r['win_rate']:.1f}%")
    print(f"  PF: {r['profit_factor']:.2f}  Sharpe: {r['sharpe']:+.2f}  Calmar: {r['calmar']:.2f}")
    print(f"  P&L: ${r['total_pnl']:+,.0f} ({r['total_pnl_pct']:+.2f}%)  DD: {r['max_dd_pct']:.2f}%")
    print(f"  Avg W: +{r['avg_win']:.3f}%  Avg L: -{r['avg_loss']:.3f}%  Exp: {r['expectancy']:+.4f}%")
    print(f"  t={r['t_stat']:+.2f}  p={r['p_value']:.4f} {sig}\n")

all_results = []

# ═════════════════════════════════════════════════════════════════════════════
# BASELINE
# ═════════════════════════════════════════════════════════════════════════════
header("BASELINE: 3+DOWN + VIX<25 (Depth-weighted)")
trades = []
for idx, row in daily.iterrows():
    if row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25:
        drop = abs(row['prev_streak_drop'])
        size = 10.0 if drop >= 5 else 7.0 if drop >= 3 else 4.0
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': size})
r = backtest(pd.DataFrame(trades), "BASE: 3+down+VIX<25 depth")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N1. FOMC DAY EFFECT
# ═════════════════════════════════════════════════════════════════════════════
header("N1. FOMC DAY EFFECT — Buy on Fed meeting days")
print("  FOMC typically meets 8 times/year. Markets often rally.\n")

# FOMC dates are predetermined — use Wednesdays with typical schedule
# Approximate: 3rd Wednesday of Jan, Mar, May, Jun, Jul, Sep, Nov, Dec
# For simplicity, we test the broader pattern: "3rd Wednesday of even months"
fomc_months = {1, 3, 5, 6, 7, 9, 11, 12}  # months with FOMC meetings
trades = []
for idx, row in daily.iterrows():
    if row['dow'] == 2 and idx.month in fomc_months:  # Wednesday
        if 14 <= idx.day <= 21:  # 3rd week
            trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': 5.0})
r = backtest(pd.DataFrame(trades), "N1: FOMC Wed buy")
pr(r); all_results.append(r) if r else None

# Day AFTER potential FOMC
trades = []
fomc_dates = set()
for idx, row in daily.iterrows():
    if row['dow'] == 2 and idx.month in fomc_months and 14 <= idx.day <= 21:
        fomc_dates.add(idx)
for i in range(1, len(daily)):
    if daily.index[i-1] in fomc_dates:
        row = daily.iloc[i]
        trades.append({'date': daily.index[i], 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': 5.0})
r = backtest(pd.DataFrame(trades), "N1b: Day after FOMC")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N2. MONTH-START EFFECT (First 3 trading days)
# ═════════════════════════════════════════════════════════════════════════════
header("N2. MONTH-START EFFECT — First 3 Trading Days")
print("  Fund flows, 401k contributions, pension rebalancing\n")

trades = []
for idx, row in daily.iterrows():
    if row['trading_day'] <= 3:
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': 4.0})
r = backtest(pd.DataFrame(trades), "N2: Month start (1-3)")
pr(r); all_results.append(r) if r else None

# VIX-filtered version
trades = []
for idx, row in daily.iterrows():
    if row['trading_day'] <= 3 and pd.notna(row['vix']) and row['vix'] < 20:
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': 4.0})
r = backtest(pd.DataFrame(trades), "N2b: Month start+VIX<20")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N3. MONTH-END EFFECT (Last 3 trading days)
# ═════════════════════════════════════════════════════════════════════════════
header("N3. MONTH-END EFFECT — Last 3 Trading Days")
print("  Window dressing, rebalancing\n")

trades = []
for idx, row in daily.iterrows():
    if row['trading_days_left'] <= 3:
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': 4.0})
r = backtest(pd.DataFrame(trades), "N3: Month end (last 3)")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N4. SKIP EARNINGS SEASON
# ═════════════════════════════════════════════════════════════════════════════
header("N4. 3+DOWN + VIX<25 + SKIP EARNINGS SEASON")
print("  Skip mid-Jan, mid-Apr, mid-Jul, mid-Oct (earnings volatility)\n")

earnings_danger = set()
for y in range(daily.index[0].year, daily.index[-1].year + 1):
    for m in [1, 4, 7, 10]:
        for d in range(10, 28):
            try:
                dt = pd.Timestamp(year=y, month=m, day=d)
                earnings_danger.add(dt.date())
            except:
                pass

trades = []
for idx, row in daily.iterrows():
    if row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25:
        if idx.date() not in earnings_danger:
            drop = abs(row['prev_streak_drop'])
            size = 10.0 if drop >= 5 else 7.0 if drop >= 3 else 4.0
            trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': size})
r = backtest(pd.DataFrame(trades), "N4: 3+down+VIX+NoEarnings")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N5. DOWN STREAK + GAP DOWN (Double panic)
# ═════════════════════════════════════════════════════════════════════════════
header("N5. 3+DOWN + GAP DOWN > 0.3% — Double Panic Bounce")
print("  If market gaps down AFTER 3+ red days, panic is extreme → bounce harder\n")

trades = []
for idx, row in daily.iterrows():
    if row['prev_streak'] <= -3 and row['gap_pct'] < -0.3:
        drop = abs(row['prev_streak_drop'])
        size = 10.0 if drop >= 5 else 7.0 if drop >= 3 else 5.0
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': size})
r = backtest(pd.DataFrame(trades), "N5: 3+down+gap down")
pr(r); all_results.append(r) if r else None

# With VIX filter
trades = []
for idx, row in daily.iterrows():
    if row['prev_streak'] <= -3 and row['gap_pct'] < -0.3 and pd.notna(row['vix']) and row['vix'] < 30:
        drop = abs(row['prev_streak_drop'])
        size = 10.0 if drop >= 5 else 7.0 if drop >= 3 else 5.0
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': size})
r = backtest(pd.DataFrame(trades), "N5b: 3+down+gap+VIX<30")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N6. VOLUME CONFIRMATION
# ═════════════════════════════════════════════════════════════════════════════
header("N6. 3+DOWN + VIX<25 + HIGH VOLUME (selling climax)")
print("  High volume on down days = capitulation = stronger bounce\n")

trades = []
for idx, row in daily.iterrows():
    if (row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25
            and pd.notna(row['volume_ratio']) and row['volume_ratio'] > 1.2):
        drop = abs(row['prev_streak_drop'])
        size = 10.0 if drop >= 5 else 7.0 if drop >= 3 else 5.0
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': size})
r = backtest(pd.DataFrame(trades), "N6: 3+down+VIX+HiVol")
pr(r); all_results.append(r) if r else None

# Low volume version (exhaustion selling)
trades = []
for idx, row in daily.iterrows():
    if (row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25
            and pd.notna(row['volume_ratio']) and row['volume_ratio'] < 0.8):
        drop = abs(row['prev_streak_drop'])
        size = 10.0 if drop >= 5 else 7.0 if drop >= 3 else 5.0
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': size})
r = backtest(pd.DataFrame(trades), "N6b: 3+down+VIX+LoVol")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N7. CLOSE-TO-CLOSE (hold overnight if bounce works)
# ═════════════════════════════════════════════════════════════════════════════
header("N7. 3+DOWN CLOSE-TO-CLOSE (buy prev close, sell today close)")
print("  Captures the overnight gap as well as intraday bounce\n")

trades = []
for i in range(1, len(daily)):
    prev_row = daily.iloc[i-1]
    row = daily.iloc[i]
    ps = prev_row['prev_streak']
    pv = prev_row['vix']
    if pd.notna(ps) and ps <= -3 and pd.notna(pv) and pv < 25:
        trades.append({'date': daily.index[i], 'entry_price': prev_row['Close'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': 5.0})
r = backtest(pd.DataFrame(trades), "N7: 3+down close-to-close")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N8. ATR PERCENTILE SIZING
# ═════════════════════════════════════════════════════════════════════════════
header("N8. 3+DOWN + VIX<25 + ATR-PERCENTILE SIZING")
print("  Low ATR (calm market) → bigger size | High ATR → smaller\n")

trades = []
for idx, row in daily.iterrows():
    if row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25:
        if pd.notna(row['atr_percentile']):
            # Inverse ATR sizing: lower percentile = bigger trade
            if row['atr_percentile'] < 0.3:
                size = 8.0  # Low vol, safe
            elif row['atr_percentile'] < 0.6:
                size = 5.0
            else:
                size = 3.0  # High vol, cautious
        else:
            size = 5.0
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': size})
r = backtest(pd.DataFrame(trades), "N8: 3+down+VIX+ATR sizing")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N9. REGIME STACKING (Mean Reversion + Day-of-Week combo day)
# ═════════════════════════════════════════════════════════════════════════════
header("N9. REGIME STACKING — Double Edge Days")
print("  When BOTH mean reversion AND day-of-week fire = double conviction\n")

trades = []
for idx, row in daily.iterrows():
    mr_signal = row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25
    dow_signal = row['dow'] in [0, 2] and pd.notna(row['vix']) and row['vix'] < 20
    
    if mr_signal and dow_signal:
        # Both edges fire — double conviction
        drop = abs(row['prev_streak_drop'])
        size = 12.0 if drop >= 5 else 10.0 if drop >= 3 else 8.0
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': size})
    elif mr_signal:
        drop = abs(row['prev_streak_drop'])
        size = 7.0 if drop >= 5 else 5.0 if drop >= 3 else 4.0
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': size})
    elif dow_signal:
        size = 3.0
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': size})

trades_sorted = sorted(trades, key=lambda x: x['date'])
r = backtest(pd.DataFrame(trades_sorted), "N9: Regime stacking")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N10. WALK-FORWARD VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
header("N10. WALK-FORWARD — Train 7yr, Test 3yr")
print("  Split data into training (first 7yr) and test (last 3yr)")
print("  If test PF > 1.2, the edge is real and not overfit\n")

split_date = daily.index[0] + pd.DateOffset(years=7)
train = daily[daily.index < split_date]
test = daily[daily.index >= split_date]

print(f"  Train: {train.index[0].strftime('%Y-%m-%d')} → {train.index[-1].strftime('%Y-%m-%d')} ({len(train)} days)")
print(f"  Test:  {test.index[0].strftime('%Y-%m-%d')} → {test.index[-1].strftime('%Y-%m-%d')} ({len(test)} days)\n")

# Test the best strategy on out-of-sample data
for label, df in [("N10-TRAIN", train), ("N10-TEST", test)]:
    trades = []
    for idx, row in df.iterrows():
        if row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25:
            drop = abs(row['prev_streak_drop'])
            size = 10.0 if drop >= 5 else 7.0 if drop >= 3 else 4.0
            trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': size})
    r = backtest(pd.DataFrame(trades), f"{label}: 3+down+VIX depth")
    pr(r); all_results.append(r) if r else None

# Mon+Wed+VIX<20
for label, df in [("N10b-TRAIN", train), ("N10b-TEST", test)]:
    trades = []
    for idx, row in df.iterrows():
        if row['dow'] in [0, 2] and pd.notna(row['vix']) and row['vix'] < 20:
            trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': 3.0})
    r = backtest(pd.DataFrame(trades), f"{label}: Mon+Wed+VIX<20")
    pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N11. THURSDAY PRE-BOUNCE
# ═════════════════════════════════════════════════════════════════════════════
header("N11. THURSDAY PRE-BOUNCE — Buy Thu close if 2+ down, sell Fri close")
print("  Hypothesis: If week has been red, Friday recovery is common\n")

trades = []
for i in range(len(daily) - 1):
    row = daily.iloc[i]
    if row['dow'] == 3 and row['prev_streak'] <= -2:  # Thursday, 2+ down
        next_row = daily.iloc[i + 1]
        if next_row['dow'] == 4:  # Friday
            trades.append({'date': daily.index[i], 'entry_price': row['Close'], 'exit_price': next_row['Close'], 'direction': 'long', 'size_pct': 5.0})
r = backtest(pd.DataFrame(trades), "N11: Thu→Fri prebounce")
pr(r); all_results.append(r) if r else None

# VIX-filtered
trades = []
for i in range(len(daily) - 1):
    row = daily.iloc[i]
    if row['dow'] == 3 and row['prev_streak'] <= -2 and pd.notna(row['vix']) and row['vix'] < 25:
        next_row = daily.iloc[i + 1]
        if next_row['dow'] == 4:
            trades.append({'date': daily.index[i], 'entry_price': row['Close'], 'exit_price': next_row['Close'], 'direction': 'long', 'size_pct': 5.0})
r = backtest(pd.DataFrame(trades), "N11b: Thu→Fri+VIX<25")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N12. VIX 20-25 RANGE + DOWN STREAK
# ═════════════════════════════════════════════════════════════════════════════
header("N12. VIX 20-25 + 3+DOWN — Panic Overdone")
print("  VIX 20-25 = elevated fear but not crash. Mean reversion stronger here?\n")

trades = []
for idx, row in daily.iterrows():
    if row['prev_streak'] <= -3 and pd.notna(row['vix']) and 20 <= row['vix'] < 25:
        drop = abs(row['prev_streak_drop'])
        size = 10.0 if drop >= 5 else 7.0 if drop >= 3 else 5.0
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': size})
r = backtest(pd.DataFrame(trades), "N12: VIX 20-25+3+down")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N13. SMA200 DISTANCE SIZING
# ═════════════════════════════════════════════════════════════════════════════
header("N13. 3+DOWN + Size Based on SMA200 Distance")
print("  Further below SMA200 = more oversold = bigger position\n")

trades = []
for idx, row in daily.iterrows():
    if row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25 and pd.notna(row['sma200_dist']):
        # Above SMA200: standard size
        if row['sma200_dist'] > 5:
            size = 4.0  # Well above, smaller bounce expected
        elif row['sma200_dist'] > 0:
            size = 5.0  # Just above
        elif row['sma200_dist'] > -3:
            size = 7.0  # Slightly below (good pullback)
        else:
            size = 3.0  # Far below (danger zone, reduce)
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': size})
r = backtest(pd.DataFrame(trades), "N13: SMA200 dist sizing")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N14. WEEKLY CLOSE EFFECT
# ═════════════════════════════════════════════════════════════════════════════
header("N14. WEEKLY CLOSE RED → MONDAY BOUNCE")
print("  If the week closes red, Monday tends to bounce harder\n")

# Calculate weekly returns
weekly_rets = {}
for yw in daily['year_week'].unique():
    week_data = daily[daily['year_week'] == yw]
    if len(week_data) >= 2:
        weekly_ret = (week_data.iloc[-1]['Close'] - week_data.iloc[0]['Open']) / week_data.iloc[0]['Open'] * 100
        weekly_rets[yw] = weekly_ret

# Buy Monday after red week
trades = []
for i in range(len(daily)):
    row = daily.iloc[i]
    if row['dow'] == 0:  # Monday
        # Find previous week
        prev_yw = row['year_week'] - 1
        if prev_yw in weekly_rets and weekly_rets[prev_yw] < 0:
            trades.append({'date': daily.index[i], 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': 5.0})
r = backtest(pd.DataFrame(trades), "N14: Mon after red week")
pr(r); all_results.append(r) if r else None

# VIX-filtered
trades = []
for i in range(len(daily)):
    row = daily.iloc[i]
    if row['dow'] == 0 and pd.notna(row['vix']) and row['vix'] < 20:
        prev_yw = row['year_week'] - 1
        if prev_yw in weekly_rets and weekly_rets[prev_yw] < 0:
            trades.append({'date': daily.index[i], 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': 5.0})
r = backtest(pd.DataFrame(trades), "N14b: Mon red wk+VIX<20")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N15. ASYMMETRIC SL/TP ON MEAN REVERSION
# ═════════════════════════════════════════════════════════════════════════════
header("N15. 3+DOWN + VIX<25 + TIGHT SL / WIDE TP")
print("  SL=-0.5%, TP=+1.5% — Cuts losses fast, lets winners run\n")

trades = []
for i in range(len(daily)):
    row = daily.iloc[i]
    if not (row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25):
        continue
    
    entry_price = row['Open']
    sl = entry_price * 0.995  # -0.5%
    tp = entry_price * 1.015  # +1.5%
    
    # Simulate intraday with daily bar
    if row['Low'] <= sl:
        exit_price = sl
    elif row['High'] >= tp:
        exit_price = tp
    else:
        exit_price = row['Close']
    
    trades.append({'date': daily.index[i], 'entry_price': entry_price, 'exit_price': exit_price, 'direction': 'long', 'size_pct': 5.0})
r = backtest(pd.DataFrame(trades), "N15: SL-0.5% TP+1.5%")
pr(r); all_results.append(r) if r else None

# Variant: SL=-1%, TP=+2%
trades = []
for i in range(len(daily)):
    row = daily.iloc[i]
    if not (row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25):
        continue
    entry_price = row['Open']
    sl = entry_price * 0.99
    tp = entry_price * 1.02
    if row['Low'] <= sl:
        exit_price = sl
    elif row['High'] >= tp:
        exit_price = tp
    else:
        exit_price = row['Close']
    trades.append({'date': daily.index[i], 'entry_price': entry_price, 'exit_price': exit_price, 'direction': 'long', 'size_pct': 5.0})
r = backtest(pd.DataFrame(trades), "N15b: SL-1% TP+2%")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# N16. ULTIMATE V3 STRATEGY — Best of everything
# ═════════════════════════════════════════════════════════════════════════════
header("N16. ULTIMATE V3 — ALL PROFITABLE IMPROVEMENTS STACKED")
print("  Combines: depth sizing + regime stacking + month-start boost")
print("  + volume confirmation where available + VIX filtering\n")

trades = []
for idx, row in daily.iterrows():
    conviction = 0
    size = 0.0
    reasons = []
    
    # Mean reversion (strongest edge)
    if row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25:
        drop = abs(row['prev_streak_drop'])
        conviction += 3
        size += 10.0 if drop >= 5 else 7.0 if drop >= 3 else 4.0
        reasons.append('MR')
    
    # Day of week
    if row['dow'] in [0, 2] and pd.notna(row['vix']) and row['vix'] < 20:
        conviction += 1
        size += 3.0
        reasons.append('DOW')
    
    # Calm market
    if pd.notna(row['vix']) and row['vix'] < 15:
        conviction += 1
        size += 2.0
        reasons.append('VIX')
    
    # Month start bonus
    if row['trading_day'] <= 3:
        conviction += 1
        size += 1.0
        reasons.append('MoStart')
    
    # Monday after red week bonus
    if row['dow'] == 0:
        prev_yw = row['year_week'] - 1
        if prev_yw in weekly_rets and weekly_rets[prev_yw] < -1.0:
            conviction += 1
            size += 1.0
            reasons.append('RedWk')
    
    # Gap down on streak = extra conviction
    if row['prev_streak'] <= -3 and row['gap_pct'] < -0.3:
        conviction += 1
        size += 2.0
        reasons.append('GapDown')
    
    if conviction >= 2:
        # Cap total size
        size = min(size, 15.0)
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long', 'size_pct': size})

trades_sorted = sorted(trades, key=lambda x: x['date'])
r = backtest(pd.DataFrame(trades_sorted), "N16: Ultimate V3")
pr(r); all_results.append(r) if r else None


# ═════════════════════════════════════════════════════════════════════════════
# FINAL RANKINGS
# ═════════════════════════════════════════════════════════════════════════════
header("FINAL V3 RANKINGS")

print(f"  B&H: {bh_total:+.0f}% ({bh_cagr:+.1f}% CAGR), Sharpe={bh_sharpe:.2f}, DD={bh_dd:.0f}%\n")

sorted_r = sorted([r for r in all_results if r is not None], key=lambda x: x['profit_factor'] if x['profit_factor'] < 100 else 0, reverse=True)

print(f"  {'#':>3} {'Strategy':<38} {'PF':>5} {'WR%':>5} {'Sharpe':>7} {'Calmar':>7} {'Trades':>6} {'P&L$':>9} {'DD%':>6} {'p-val':>7} {'Sig':>3}")
print(f"  {'─'*3} {'─'*38} {'─'*5} {'─'*5} {'─'*7} {'─'*7} {'─'*6} {'─'*9} {'─'*6} {'─'*7} {'─'*3}")

for i, r in enumerate(sorted_r):
    sig = "***" if r['p_value'] < 0.01 else "** " if r['p_value'] < 0.05 else "*  " if r['p_value'] < 0.10 else "   "
    pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "INF"
    cal = f"{r['calmar']:.2f}" if r['calmar'] < 100 else "INF"
    print(f"  {i+1:3d} {r['name']:<38} {pf:>5} {r['win_rate']:5.1f} {r['sharpe']:+7.2f} {cal:>7} {r['trades']:6d} {r['total_pnl']:+9,.0f} {r['max_dd_pct']:5.2f}% {r['p_value']:7.4f} {sig}")

# Walk-forward comparison
print(f"\n  ── WALK-FORWARD VALIDATION ──")
wf_results = {r['name']: r for r in all_results if r and 'N10' in r['name']}
for base in ['3+down+VIX depth', 'Mon+Wed+VIX<20']:
    train_key = [k for k in wf_results if 'TRAIN' in k and base in k]
    test_key = [k for k in wf_results if 'TEST' in k and base in k]
    if train_key and test_key:
        tr = wf_results[train_key[0]]
        te = wf_results[test_key[0]]
        status = "✅ CONFIRMED" if te['profit_factor'] > 1.0 and te['sharpe'] > 0 else "❌ FAILED"
        print(f"  {base}:")
        print(f"    Train PF={tr['profit_factor']:.2f} Sharpe={tr['sharpe']:+.2f}")
        print(f"    Test  PF={te['profit_factor']:.2f} Sharpe={te['sharpe']:+.2f}  {status}")

# Best new improvements
print(f"\n  ── BEST NEW IMPROVEMENTS ──")
new_r = [r for r in sorted_r if r and 'N' in r['name'] and 'BASE' not in r['name'] and 'N10' not in r['name']]
profitable = [r for r in new_r if r['profit_factor'] > 1.0 and r['trades'] > 20]
if profitable:
    for r in profitable[:10]:
        sig = "***" if r['p_value'] < 0.01 else "** " if r['p_value'] < 0.05 else "*  " if r['p_value'] < 0.10 else "   "
        print(f"    {r['name']:<38} PF={r['profit_factor']:.2f} Sharpe={r['sharpe']:+.2f} P&L=${r['total_pnl']:+,.0f} {sig}")

# Save
results_json = {
    'generated': datetime.now().isoformat(),
    'capital': CAPITAL,
    'buy_and_hold': {'total_pct': bh_total, 'cagr': bh_cagr, 'sharpe': bh_sharpe, 'max_dd': bh_dd},
    'strategies': sorted_r
}
json_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'nasdaq_v3_results.json')
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2, default=str)

print(f"\n  Saved: outputs/nasdaq_v3_results.txt + .json")
print(f"  Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
