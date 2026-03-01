"""
NASDAQ Concrete Strategy Backtester
====================================
Tests SPECIFIC tradeable strategies with proper P&L, drawdown, win rates.
Every strategy produces a trade log so we can see if it's actually profitable.

Capital: $25,000 (prop firm context)
Instrument: QQQ / NAS100 / NQ futures

Strategies tested:
  S1.  Buy pre-open (9:28 ET), sell 10:30 ET — ride first hour
  S2.  Buy at open (9:30 ET), sell 10:30 ET
  S3.  Buy at open (9:30 ET), sell at close (16:00 ET)
  S4.  Buy 3+ down days — mean reversion (daily)
  S5.  Monday LONG at open → close
  S6.  Wednesday LONG at open → close
  S7.  Mon+Wed + VIX<20 combined
  S8.  3+ down days + VIX<25
  S9.  Gap fade: short big gap-ups, long big gap-downs
  S10. Pre-market momentum continuation into first hour
  S11. After consecutive down days + buy pre-open → sell 10:30
  S12. VIX regime long-only (<15)
  S13. First hour breakout (buy if first 15-min breaks high)
  S14. Power hour reversal (fade 14:00-15:00 move at 15:00)
  S15. Overnight hold: buy at close, sell at open
"""

import sys, os, warnings, json
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats

LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'nasdaq_strategies_results.txt')

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
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}\n")

# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
header("NASDAQ STRATEGY BACKTESTER")
print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

print("\n[1] QQQ hourly (2yr)...")
qqq_1h = yf.download("QQQ", interval="1h", period="2y", progress=False, prepost=True)
if isinstance(qqq_1h.columns, pd.MultiIndex):
    qqq_1h.columns = qqq_1h.columns.get_level_values(0)
print(f"    {len(qqq_1h)} bars")

print("[2] QQQ 5-min (60d, pre/post)...")
qqq_5m = yf.download("QQQ", interval="5m", period="60d", progress=False, prepost=True)
if isinstance(qqq_5m.columns, pd.MultiIndex):
    qqq_5m.columns = qqq_5m.columns.get_level_values(0)
print(f"    {len(qqq_5m)} bars")

print("[3] QQQ 2-min (60d, pre/post)...")
qqq_2m = yf.download("QQQ", interval="2m", period="60d", progress=False, prepost=True)
if isinstance(qqq_2m.columns, pd.MultiIndex):
    qqq_2m.columns = qqq_2m.columns.get_level_values(0)
print(f"    {len(qqq_2m)} bars")

print("[4] QQQ daily (10yr)...")
qqq_daily = yf.download("QQQ", interval="1d", period="10y", progress=False)
if isinstance(qqq_daily.columns, pd.MultiIndex):
    qqq_daily.columns = qqq_daily.columns.get_level_values(0)
print(f"    {len(qqq_daily)} bars")

print("[5] VIX daily (10yr)...")
vix_daily = yf.download("^VIX", interval="1d", period="10y", progress=False)
if isinstance(vix_daily.columns, pd.MultiIndex):
    vix_daily.columns = vix_daily.columns.get_level_values(0)
print(f"    {len(vix_daily)} bars")

print("  Downloads complete.\n")

# ─────────────────────────────────────────────────────────────────────────────
# PREPARE DAILY DATA
# ─────────────────────────────────────────────────────────────────────────────
daily = qqq_daily.copy()
daily['prev_close'] = daily['Close'].shift(1)
daily['gap_pct'] = (daily['Open'] - daily['prev_close']) / daily['prev_close'] * 100
daily['intraday_ret_pct'] = (daily['Close'] - daily['Open']) / daily['Open'] * 100
daily['full_ret_pct'] = daily['Close'].pct_change() * 100
daily['overnight_ret_pct'] = daily['gap_pct']
daily['dow'] = daily.index.dayofweek
daily = daily.dropna()

# Add VIX
vix_close = vix_daily['Close'].copy()
vix_close.index = pd.to_datetime(vix_close.index)
if hasattr(vix_close.index, 'tz') and vix_close.index.tz is not None:
    vix_close.index = vix_close.index.tz_localize(None)
daily_idx = pd.to_datetime(daily.index)
if hasattr(daily_idx, 'tz') and daily_idx.tz is not None:
    daily_idx = daily_idx.tz_localize(None)
daily.index = daily_idx
daily['vix'] = vix_close.reindex(daily.index, method='ffill')

# Consecutive days streak
daily['streak'] = 0
streak = 0
streaks = []
for i in range(len(daily)):
    ret = daily.iloc[i]['full_ret_pct']
    if ret > 0:
        streak = streak + 1 if streak > 0 else 1
    elif ret < 0:
        streak = streak - 1 if streak < 0 else -1
    else:
        streak = 0
    streaks.append(streak)
daily['streak'] = streaks
daily['prev_streak'] = daily['streak'].shift(1)

# ─────────────────────────────────────────────────────────────────────────────
# PREPARE HOURLY DATA
# ─────────────────────────────────────────────────────────────────────────────
h1 = qqq_1h.copy()
h1.index = h1.index.tz_convert('US/Eastern')
h1['hour'] = h1.index.hour
h1['minute'] = h1.index.minute
h1['date'] = h1.index.date

# ─────────────────────────────────────────────────────────────────────────────
# PREPARE 5-MIN DATA
# ─────────────────────────────────────────────────────────────────────────────
m5 = qqq_5m.copy()
m5.index = m5.index.tz_convert('US/Eastern')
m5['hour'] = m5.index.hour
m5['minute'] = m5.index.minute
m5['date'] = m5.index.date

# ─────────────────────────────────────────────────────────────────────────────
# PREPARE 2-MIN DATA
# ─────────────────────────────────────────────────────────────────────────────
m2 = qqq_2m.copy()
m2.index = m2.index.tz_convert('US/Eastern')
m2['hour'] = m2.index.hour
m2['minute'] = m2.index.minute
m2['date'] = m2.index.date


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
CAPITAL = 25000.0
COMMISSION_PCT = 0.0  # QQQ/NAS100 CFD spread covered in slippage
SLIPPAGE_PCT = 0.01   # 1 bp slippage per side

def backtest_trades(trades_df, name, capital=CAPITAL):
    """
    Given a DataFrame with columns: date, entry_price, exit_price, direction ('long'/'short')
    Returns detailed metrics.
    """
    if len(trades_df) == 0:
        return None
    
    results = []
    equity = capital
    peak = capital
    max_dd = 0
    max_dd_pct = 0
    
    for _, t in trades_df.iterrows():
        entry = t['entry_price']
        exit_p = t['exit_price']
        direction = t.get('direction', 'long')
        
        # Apply slippage
        if direction == 'long':
            adj_entry = entry * (1 + SLIPPAGE_PCT / 100)
            adj_exit = exit_p * (1 - SLIPPAGE_PCT / 100)
            pnl_pct = (adj_exit - adj_entry) / adj_entry * 100
        else:
            adj_entry = entry * (1 - SLIPPAGE_PCT / 100)
            adj_exit = exit_p * (1 + SLIPPAGE_PCT / 100)
            pnl_pct = (adj_entry - adj_exit) / adj_entry * 100
        
        # Position sizing: risk 5% of equity per trade
        position_size = equity * 0.05
        pnl_dollar = position_size * pnl_pct / 100
        
        equity += pnl_dollar
        peak = max(peak, equity)
        dd = peak - equity
        dd_pct = dd / peak * 100
        max_dd = max(max_dd, dd)
        max_dd_pct = max(max_dd_pct, dd_pct)
        
        results.append({
            'date': t['date'],
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'equity': equity,
            'drawdown_pct': dd_pct
        })
    
    res_df = pd.DataFrame(results)
    
    wins = res_df[res_df['pnl_pct'] > 0]
    losses = res_df[res_df['pnl_pct'] <= 0]
    
    total_trades = len(res_df)
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['pnl_pct'].mean()) if len(losses) > 0 else 0
    
    gross_profit = wins['pnl_dollar'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl_dollar'].sum()) if len(losses) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    total_pnl = res_df['pnl_dollar'].sum()
    total_pnl_pct = (equity - capital) / capital * 100
    
    # Sharpe (annualized)
    daily_rets = res_df['pnl_pct']
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() > 0 else 0
    
    # Expectancy per trade
    expectancy = daily_rets.mean()
    
    # t-test
    t_stat, p_val = stats.ttest_1samp(daily_rets, 0) if len(daily_rets) > 2 else (0, 1)
    
    # Trades per year estimate
    if len(res_df) > 1:
        first_date = pd.to_datetime(res_df['date'].iloc[0])
        last_date = pd.to_datetime(res_df['date'].iloc[-1])
        years = max((last_date - first_date).days / 365.25, 0.1)
        trades_per_year = total_trades / years
    else:
        trades_per_year = 0
        years = 0
    
    return {
        'name': name,
        'trades': total_trades,
        'trades_per_year': trades_per_year,
        'years': years,
        'win_rate': win_rate,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'profit_factor': pf,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'max_dd_pct': max_dd_pct,
        'sharpe': sharpe,
        'expectancy_pct': expectancy,
        't_stat': t_stat,
        'p_value': p_val,
        'final_equity': equity,
    }


def print_result(r):
    if r is None:
        print("  NO TRADES\n")
        return
    sig = "***" if r['p_value'] < 0.01 else "** " if r['p_value'] < 0.05 else "*  " if r['p_value'] < 0.10 else "   "
    print(f"  Trades:         {r['trades']:,d} ({r['trades_per_year']:.0f}/yr over {r['years']:.1f} yrs)")
    print(f"  Win Rate:       {r['win_rate']:.1f}%")
    print(f"  Avg Win:        +{r['avg_win_pct']:.3f}%    Avg Loss: -{r['avg_loss_pct']:.3f}%")
    print(f"  Profit Factor:  {r['profit_factor']:.2f}")
    print(f"  Total P&L:      ${r['total_pnl']:+,.2f} ({r['total_pnl_pct']:+.1f}%)")
    print(f"  Max Drawdown:   {r['max_dd_pct']:.2f}%")
    print(f"  Sharpe (ann):   {r['sharpe']:+.2f}")
    print(f"  Expectancy:     {r['expectancy_pct']:+.4f}% per trade")
    print(f"  t-stat:         {r['t_stat']:+.3f}  p={r['p_value']:.4f} {sig}")
    print(f"  Final Equity:   ${r['final_equity']:,.2f} (from $25,000)")
    print()


all_results = []

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S1: Buy pre-open 9:25, sell 10:30 (first hour ride)
# ─────────────────────────────────────────────────────────────────────────────
header("S1. BUY PRE-OPEN (9:25 ET) → SELL 10:30 ET — First Hour Ride")
print("  Logic: Buy at the 9:25 5-min bar close (5 min before open),")
print("  capture the opening volatility burst, exit at 10:30.\n")

trades_s1 = []
for date in m5['date'].unique():
    day = m5[m5['date'] == date]
    # Entry: 9:25 bar
    entry_bar = day[(day['hour'] == 9) & (day['minute'] == 25)]
    if len(entry_bar) == 0:
        continue
    # Exit: 10:30 bar
    exit_bar = day[(day['hour'] == 10) & (day['minute'] == 30)]
    if len(exit_bar) == 0:
        # Try 10:25 or 10:35
        exit_bar = day[(day['hour'] == 10) & (day['minute'] >= 25) & (day['minute'] <= 35)]
    if len(exit_bar) == 0:
        continue
    
    trades_s1.append({
        'date': date,
        'entry_price': entry_bar.iloc[0]['Close'],
        'exit_price': exit_bar.iloc[-1]['Close'],
        'direction': 'long'
    })

r = backtest_trades(pd.DataFrame(trades_s1), "S1: Pre-open→10:30")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S2: Buy at open 9:30, sell 10:30
# ─────────────────────────────────────────────────────────────────────────────
header("S2. BUY AT OPEN (9:30 ET) → SELL 10:30 ET")

trades_s2 = []
for date in m5['date'].unique():
    day = m5[m5['date'] == date]
    entry_bar = day[(day['hour'] == 9) & (day['minute'] == 30)]
    exit_bar = day[(day['hour'] == 10) & (day['minute'] >= 25) & (day['minute'] <= 35)]
    if len(entry_bar) == 0 or len(exit_bar) == 0:
        continue
    trades_s2.append({
        'date': date,
        'entry_price': entry_bar.iloc[0]['Open'],
        'exit_price': exit_bar.iloc[-1]['Close'],
        'direction': 'long'
    })

r = backtest_trades(pd.DataFrame(trades_s2), "S2: Open→10:30")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S3: Buy open, sell close (full day long)
# ─────────────────────────────────────────────────────────────────────────────
header("S3. BUY OPEN → SELL CLOSE (Full Day Long, 10yr)")

trades_s3 = []
for idx, row in daily.iterrows():
    trades_s3.append({
        'date': idx,
        'entry_price': row['Open'],
        'exit_price': row['Close'],
        'direction': 'long'
    })

r = backtest_trades(pd.DataFrame(trades_s3), "S3: Full day long")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S4: Buy after 3+ consecutive down days
# ─────────────────────────────────────────────────────────────────────────────
header("S4. MEAN REVERSION — Buy After 3+ Down Days (10yr)")
print("  Logic: If QQQ closed red for 3+ consecutive days,")
print("  go LONG at next day open, exit at close.\n")

trades_s4 = []
for idx, row in daily.iterrows():
    if row['prev_streak'] <= -3:
        trades_s4.append({
            'date': idx,
            'entry_price': row['Open'],
            'exit_price': row['Close'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s4), "S4: 3+ down days")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S4b: Buy after 2+ consecutive down days
# ─────────────────────────────────────────────────────────────────────────────
header("S4b. MEAN REVERSION — Buy After 2+ Down Days (10yr)")

trades_s4b = []
for idx, row in daily.iterrows():
    if row['prev_streak'] <= -2:
        trades_s4b.append({
            'date': idx,
            'entry_price': row['Open'],
            'exit_price': row['Close'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s4b), "S4b: 2+ down days")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S5: Monday only long
# ─────────────────────────────────────────────────────────────────────────────
header("S5. MONDAY LONG — Open to Close (10yr)")

trades_s5 = []
for idx, row in daily.iterrows():
    if row['dow'] == 0:  # Monday
        trades_s5.append({
            'date': idx,
            'entry_price': row['Open'],
            'exit_price': row['Close'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s5), "S5: Monday long")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S6: Wednesday only long
# ─────────────────────────────────────────────────────────────────────────────
header("S6. WEDNESDAY LONG — Open to Close (10yr)")

trades_s6 = []
for idx, row in daily.iterrows():
    if row['dow'] == 2:  # Wednesday
        trades_s6.append({
            'date': idx,
            'entry_price': row['Open'],
            'exit_price': row['Close'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s6), "S6: Wednesday long")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S7: Mon+Wed + VIX < 20
# ─────────────────────────────────────────────────────────────────────────────
header("S7. MON + WED + VIX < 20 — Combined Filter (10yr)")
print("  Logic: Only trade Mondays and Wednesdays when VIX < 20.\n")

trades_s7 = []
for idx, row in daily.iterrows():
    if row['dow'] in [0, 2] and pd.notna(row['vix']) and row['vix'] < 20:
        trades_s7.append({
            'date': idx,
            'entry_price': row['Open'],
            'exit_price': row['Close'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s7), "S7: Mon+Wed+VIX<20")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S8: 3+ down days + VIX < 25
# ─────────────────────────────────────────────────────────────────────────────
header("S8. 3+ DOWN DAYS + VIX < 25 (10yr)")
print("  Logic: Mean reversion bounce but skip panic/crash periods.\n")

trades_s8 = []
for idx, row in daily.iterrows():
    if row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25:
        trades_s8.append({
            'date': idx,
            'entry_price': row['Open'],
            'exit_price': row['Close'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s8), "S8: 3+down+VIX<25")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S8b: 3+ down days + VIX < 30, hold for 2 days
# ─────────────────────────────────────────────────────────────────────────────
header("S8b. 3+ DOWN DAYS + VIX < 30, HOLD 2 DAYS (10yr)")
print("  Logic: Buy day after 3+ reds, hold through next day.\n")

trades_s8b = []
for i in range(len(daily) - 1):
    row = daily.iloc[i]
    if row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 30:
        next_row = daily.iloc[i + 1]
        trades_s8b.append({
            'date': daily.index[i],
            'entry_price': row['Open'],
            'exit_price': next_row['Close'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s8b), "S8b: 3+down 2-day hold")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S9: Gap Fade (only large gaps >0.5%)
# ─────────────────────────────────────────────────────────────────────────────
header("S9. GAP FADE — Short Gap-Ups, Long Gap-Downs (>0.5%, 10yr)")

trades_s9 = []
for idx, row in daily.iterrows():
    if row['gap_pct'] > 0.5:
        trades_s9.append({
            'date': idx,
            'entry_price': row['Open'],
            'exit_price': row['Close'],
            'direction': 'short'
        })
    elif row['gap_pct'] < -0.5:
        trades_s9.append({
            'date': idx,
            'entry_price': row['Open'],
            'exit_price': row['Close'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s9), "S9: Gap fade >0.5%")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S10: Pre-market momentum continuation into first hour
# ─────────────────────────────────────────────────────────────────────────────
header("S10. PRE-MARKET MOMENTUM → FIRST HOUR CONTINUATION (5m data)")
print("  Logic: If pre-market (9:00-9:25) is UP, go LONG at 9:30.")
print("  If pre-market is DOWN, go SHORT at 9:30. Exit at 10:30.\n")

trades_s10 = []
for date in m5['date'].unique():
    day = m5[m5['date'] == date]
    
    # Pre-market momentum 9:00-9:25
    pre = day[(day['hour'] == 9) & (day['minute'] < 30)]
    if len(pre) < 2:
        continue
    pre_ret = (pre.iloc[-1]['Close'] - pre.iloc[0]['Open']) / pre.iloc[0]['Open']
    
    # Entry at 9:30
    entry_bar = day[(day['hour'] == 9) & (day['minute'] == 30)]
    if len(entry_bar) == 0:
        continue
    
    # Exit at 10:30
    exit_bar = day[(day['hour'] == 10) & (day['minute'] >= 25) & (day['minute'] <= 35)]
    if len(exit_bar) == 0:
        continue
    
    direction = 'long' if pre_ret > 0 else 'short'
    trades_s10.append({
        'date': date,
        'entry_price': entry_bar.iloc[0]['Open'],
        'exit_price': exit_bar.iloc[-1]['Close'],
        'direction': direction,
        'pre_ret': pre_ret
    })

r = backtest_trades(pd.DataFrame(trades_s10), "S10: Pre-mkt momentum→1hr")
print_result(r)
if r: all_results.append(r)

# Also test: fade pre-market
header("S10b. FADE PRE-MARKET INTO FIRST HOUR (5m data)")
print("  Logic: If pre-market UP → SHORT at open. If DOWN → LONG at open.\n")

trades_s10b = []
for t in trades_s10:
    trades_s10b.append({
        'date': t['date'],
        'entry_price': t['entry_price'],
        'exit_price': t['exit_price'],
        'direction': 'short' if t['direction'] == 'long' else 'long'
    })

r = backtest_trades(pd.DataFrame(trades_s10b), "S10b: Fade pre-mkt→1hr")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S11: 3+ down streak → buy pre-open 9:25, sell 10:30
# ─────────────────────────────────────────────────────────────────────────────
header("S11. 3+ DOWN DAYS → BUY PRE-OPEN 9:25 → SELL 10:30 (5m data)")
print("  Combines: strongest statistical edge + first hour.\n")

# Get dates where prev streak was <= -3
bounce_dates = set()
for idx, row in daily.iterrows():
    if row['prev_streak'] <= -3:
        bounce_dates.add(idx.date() if hasattr(idx, 'date') else idx)

trades_s11 = []
for date in m5['date'].unique():
    if date not in bounce_dates:
        continue
    day = m5[m5['date'] == date]
    entry_bar = day[(day['hour'] == 9) & (day['minute'] == 25)]
    exit_bar = day[(day['hour'] == 10) & (day['minute'] >= 25) & (day['minute'] <= 35)]
    if len(entry_bar) == 0 or len(exit_bar) == 0:
        continue
    trades_s11.append({
        'date': date,
        'entry_price': entry_bar.iloc[0]['Close'],
        'exit_price': exit_bar.iloc[-1]['Close'],
        'direction': 'long'
    })

r = backtest_trades(pd.DataFrame(trades_s11), "S11: 3+down+preopen→10:30")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S12: VIX < 15 long only
# ─────────────────────────────────────────────────────────────────────────────
header("S12. VIX < 15 LONG ONLY — Open to Close (10yr)")
print("  Logic: Only trade when VIX < 15 (calm market). Go long.\n")

trades_s12 = []
for idx, row in daily.iterrows():
    if pd.notna(row['vix']) and row['vix'] < 15:
        trades_s12.append({
            'date': idx,
            'entry_price': row['Open'],
            'exit_price': row['Close'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s12), "S12: VIX<15 long")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S13: Opening Range Breakout — first 15 min
# ─────────────────────────────────────────────────────────────────────────────
header("S13. OPENING RANGE BREAKOUT (15-min range, 5m data)")
print("  Logic: First 15 min (9:30-9:45) sets the range.")
print("  If price breaks above range high → LONG.")
print("  If price breaks below range low → SHORT.")
print("  Exit at 16:00. First breakout only.\n")

trades_s13 = []
for date in m5['date'].unique():
    day = m5[m5['date'] == date]
    
    # Opening range: 9:30-9:45 (3 bars of 5-min)
    orb = day[(day['hour'] == 9) & (day['minute'] >= 30) & (day['minute'] < 45)]
    if len(orb) < 2:
        continue
    
    or_high = orb['High'].max()
    or_low = orb['Low'].min()
    
    # Rest of day (after 9:45)
    rest = day[day.index > orb.index[-1]]
    rest = rest[(rest['hour'] >= 9) & (rest['hour'] < 16)]
    if len(rest) < 5:
        continue
    
    # Find first breakout
    entered = False
    for i, (idx, bar) in enumerate(rest.iterrows()):
        if bar['High'] > or_high and not entered:
            # Long breakout
            entry_price = or_high
            # Exit at day close
            exit_price = rest.iloc[-1]['Close']
            trades_s13.append({
                'date': date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'long'
            })
            entered = True
            break
        elif bar['Low'] < or_low and not entered:
            # Short breakout
            entry_price = or_low
            exit_price = rest.iloc[-1]['Close']
            trades_s13.append({
                'date': date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'short'
            })
            entered = True
            break

r = backtest_trades(pd.DataFrame(trades_s13), "S13: ORB 15min")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S14: Overnight hold — Buy at close, sell at open
# ─────────────────────────────────────────────────────────────────────────────
header("S14. OVERNIGHT HOLD — Buy at Close, Sell at Open (10yr)")
print("  Logic: Capture the overnight gap (mean +0.05%/day).\n")

trades_s14 = []
for i in range(len(daily) - 1):
    trades_s14.append({
        'date': daily.index[i],
        'entry_price': daily.iloc[i]['Close'],
        'exit_price': daily.iloc[i + 1]['Open'],
        'direction': 'long'
    })

r = backtest_trades(pd.DataFrame(trades_s14), "S14: Overnight long")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S15: Mon+Wed + VIX<20 + first hour only
# ─────────────────────────────────────────────────────────────────────────────
header("S15. MON+WED + VIX<20 + FIRST HOUR (5m data)")
print("  Logic: Combine day-of-week + VIX filter + first hour trade.\n")

# Get qualifying dates
s15_dates = set()
for idx, row in daily.iterrows():
    if row['dow'] in [0, 2] and pd.notna(row['vix']) and row['vix'] < 20:
        s15_dates.add(idx.date() if hasattr(idx, 'date') else idx)

trades_s15 = []
for date in m5['date'].unique():
    if date not in s15_dates:
        continue
    day = m5[m5['date'] == date]
    entry_bar = day[(day['hour'] == 9) & (day['minute'] == 30)]
    exit_bar = day[(day['hour'] == 10) & (day['minute'] >= 25) & (day['minute'] <= 35)]
    if len(entry_bar) == 0 or len(exit_bar) == 0:
        continue
    trades_s15.append({
        'date': date,
        'entry_price': entry_bar.iloc[0]['Open'],
        'exit_price': exit_bar.iloc[-1]['Close'],
        'direction': 'long'
    })

r = backtest_trades(pd.DataFrame(trades_s15), "S15: Mon+Wed+VIX<20+1hr")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S16: 3+ down days, hold 3 days (extended bounce)
# ─────────────────────────────────────────────────────────────────────────────
header("S16. 3+ DOWN DAYS → HOLD 3 DAYS (10yr)")

trades_s16 = []
for i in range(len(daily) - 2):
    row = daily.iloc[i]
    if row['prev_streak'] <= -3:
        exit_row = daily.iloc[i + 2]
        trades_s16.append({
            'date': daily.index[i],
            'entry_price': row['Open'],
            'exit_price': exit_row['Close'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s16), "S16: 3+down hold 3 days")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S17: First hour + VIX < 20 (hourly data, 2yr)
# ─────────────────────────────────────────────────────────────────────────────
header("S17. FIRST HOUR LONG + VIX < 20 (hourly, 2yr)")

# Get VIX dates
vix_dates = {}
for idx, row in daily.iterrows():
    d = idx.date() if hasattr(idx, 'date') else idx
    if pd.notna(row['vix']):
        vix_dates[d] = row['vix']

trades_s17 = []
for date in h1['date'].unique():
    if date not in vix_dates or vix_dates[date] >= 20:
        continue
    day = h1[h1['date'] == date]
    entry_bar = day[day['hour'] == 9]
    exit_bar = day[day['hour'] == 10]
    if len(entry_bar) == 0 or len(exit_bar) == 0:
        continue
    trades_s17.append({
        'date': date,
        'entry_price': entry_bar.iloc[0]['Open'],
        'exit_price': exit_bar.iloc[-1]['Close'],
        'direction': 'long'
    })

r = backtest_trades(pd.DataFrame(trades_s17), "S17: 1st hr+VIX<20")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S18: Buy dip — if first 30 min drops >0.3%, buy for rest of day
# ─────────────────────────────────────────────────────────────────────────────
header("S18. BUY THE DIP — First 30min Down >0.3% → Long Rest of Day (5m)")
print("  Logic: If 9:30-10:00 drops more than 0.3%, buy at 10:00.\n")

trades_s18 = []
for date in m5['date'].unique():
    day = m5[m5['date'] == date]
    
    first_30 = day[(day['hour'] == 9) & (day['minute'] >= 30)]
    if len(first_30) < 3:
        continue
    
    open_price = first_30.iloc[0]['Open']
    price_at_10 = first_30.iloc[-1]['Close']
    first_30_ret = (price_at_10 - open_price) / open_price * 100
    
    if first_30_ret < -0.3:
        # Buy the dip at 10:00
        rest = day[(day['hour'] >= 10) & (day['hour'] < 16)]
        if len(rest) < 5:
            continue
        trades_s18.append({
            'date': date,
            'entry_price': price_at_10,
            'exit_price': rest.iloc[-1]['Close'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s18), "S18: Buy 30min dip")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S19: Tuesday overnight — buy Tue close, sell Wed open
# ─────────────────────────────────────────────────────────────────────────────
header("S19. TUESDAY NIGHT HOLD — Buy Tue Close, Sell Wed Open (10yr)")
print("  Tuesday has strongest overnight return.\n")

trades_s19 = []
for i in range(len(daily) - 1):
    if daily.iloc[i]['dow'] == 1:  # Tuesday
        trades_s19.append({
            'date': daily.index[i],
            'entry_price': daily.iloc[i]['Close'],
            'exit_price': daily.iloc[i + 1]['Open'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s19), "S19: Tue night hold")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S20: COMBO — 3+ down + Mon/Wed + VIX filter (any match)
# ─────────────────────────────────────────────────────────────────────────────
header("S20. COMBO STRATEGY — Score-Based (10yr)")
print("  Logic: Score system: +1 for each condition met:")
print("    - 3+ consecutive down days")
print("    - Monday or Wednesday")  
print("    - VIX < 20")
print("    - Previous day was down")
print("  Trade if score >= 2\n")

trades_s20 = []
for idx, row in daily.iterrows():
    score = 0
    if row['prev_streak'] <= -3:
        score += 2  # strong edge, double weight
    if row['dow'] in [0, 2]:
        score += 1
    if pd.notna(row['vix']) and row['vix'] < 20:
        score += 1
    if row['prev_streak'] <= -1:
        score += 1
    
    if score >= 3:
        trades_s20.append({
            'date': idx,
            'entry_price': row['Open'],
            'exit_price': row['Close'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s20), "S20: Combo score>=3")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S21: Gap-up + VIX < 15 continuation (ride the momentum)
# ─────────────────────────────────────────────────────────────────────────────
header("S21. GAP-UP CONTINUATION in Low VIX (10yr)")
print("  Logic: If gap > +0.3% AND VIX < 15, go LONG at open → close.\n")

trades_s21 = []
for idx, row in daily.iterrows():
    if row['gap_pct'] > 0.3 and pd.notna(row['vix']) and row['vix'] < 15:
        trades_s21.append({
            'date': idx,
            'entry_price': row['Open'],
            'exit_price': row['Close'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s21), "S21: GapUp+VIX<15")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S22: Buy pre-open 9:25, sell at CLOSE (full day, 5m data)  
# ─────────────────────────────────────────────────────────────────────────────
header("S22. BUY PRE-OPEN 9:25 → SELL AT CLOSE (5m data)")

trades_s22 = []
for date in m5['date'].unique():
    day = m5[m5['date'] == date]
    entry_bar = day[(day['hour'] == 9) & (day['minute'] == 25)]
    # Close of day
    rth = day[(day['hour'] >= 9) & (day['hour'] < 16)]
    if len(entry_bar) == 0 or len(rth) < 10:
        continue
    trades_s22.append({
        'date': date,
        'entry_price': entry_bar.iloc[0]['Close'],
        'exit_price': rth.iloc[-1]['Close'],
        'direction': 'long'
    })

r = backtest_trades(pd.DataFrame(trades_s22), "S22: PreOpen→Close")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S23: RANGE-BOUND FADE — Sell if first hour rips >0.5%
# ─────────────────────────────────────────────────────────────────────────────
header("S23. FADE FIRST HOUR RIP — Short if 1st hr > +0.5% (5m data)")
print("  Logic: If open→10:30 is up >0.5%, SHORT at 10:30 → close.\n")

trades_s23 = []
for date in m5['date'].unique():
    day = m5[m5['date'] == date]
    open_bar = day[(day['hour'] == 9) & (day['minute'] == 30)]
    bar_1030 = day[(day['hour'] == 10) & (day['minute'] >= 25) & (day['minute'] <= 35)]
    rth_end = day[(day['hour'] >= 15) & (day['hour'] < 16)]
    
    if len(open_bar) == 0 or len(bar_1030) == 0 or len(rth_end) == 0:
        continue
    
    first_hr_ret = (bar_1030.iloc[-1]['Close'] - open_bar.iloc[0]['Open']) / open_bar.iloc[0]['Open'] * 100
    
    if first_hr_ret > 0.5:
        trades_s23.append({
            'date': date,
            'entry_price': bar_1030.iloc[-1]['Close'],
            'exit_price': rth_end.iloc[-1]['Close'],
            'direction': 'short'
        })
    elif first_hr_ret < -0.5:
        trades_s23.append({
            'date': date,
            'entry_price': bar_1030.iloc[-1]['Close'],
            'exit_price': rth_end.iloc[-1]['Close'],
            'direction': 'long'
        })

r = backtest_trades(pd.DataFrame(trades_s23), "S23: Fade 1st hr >0.5%")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S24: Pre-market momentum continuation (2-min data)
# ─────────────────────────────────────────────────────────────────────────────
header("S24. PRE-MARKET MOMENTUM — 2-min bars, 9:00-9:28 trend (2m data)")
print("  Logic: If 9:00-9:28 is UP (pre-market), buy at 9:28, sell at 10:30.")
print("  If DOWN, short at 9:28, sell at 10:30.\n")

trades_s24 = []
for date in m2['date'].unique():
    day = m2[m2['date'] == date]
    
    # Pre-market: 9:00 to 9:28
    pre = day[(day['hour'] == 9) & (day['minute'] >= 0) & (day['minute'] < 28)]
    if len(pre) < 3:
        continue
    
    pre_ret = (pre.iloc[-1]['Close'] - pre.iloc[0]['Open']) / pre.iloc[0]['Open']
    
    # Entry at 9:28
    entry_bar = day[(day['hour'] == 9) & (day['minute'] == 28)]
    if len(entry_bar) == 0:
        continue
    
    # Exit at ~10:30
    exit_bar = day[(day['hour'] == 10) & (day['minute'] >= 28) & (day['minute'] <= 32)]
    if len(exit_bar) == 0:
        continue
    
    direction = 'long' if pre_ret > 0 else 'short'
    trades_s24.append({
        'date': date,
        'entry_price': entry_bar.iloc[0]['Close'],
        'exit_price': exit_bar.iloc[-1]['Close'],
        'direction': direction
    })

r = backtest_trades(pd.DataFrame(trades_s24), "S24: PreMkt 2min→10:30")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S25: Buy pre-open 9:28, with stop loss at -0.3%, TP at +0.5%
# ─────────────────────────────────────────────────────────────────────────────
header("S25. BUY 9:28 WITH SL/TP (2-min data)")
print("  Logic: Buy at 9:28. SL = -0.3%, TP = +0.5%. Max hold 1hr.\n")

trades_s25 = []
for date in m2['date'].unique():
    day = m2[m2['date'] == date]
    entry_bar = day[(day['hour'] == 9) & (day['minute'] == 28)]
    if len(entry_bar) == 0:
        continue
    
    entry_price = entry_bar.iloc[0]['Close']
    sl_price = entry_price * (1 - 0.003)  # -0.3%
    tp_price = entry_price * (1 + 0.005)  # +0.5%
    
    # Simulate forward
    post = day[day.index > entry_bar.index[0]]
    max_hold_bars = 30  # ~1 hour in 2-min bars
    
    exit_price = entry_price
    for j in range(min(max_hold_bars, len(post))):
        bar = post.iloc[j]
        if bar['Low'] <= sl_price:
            exit_price = sl_price
            break
        if bar['High'] >= tp_price:
            exit_price = tp_price
            break
        exit_price = bar['Close']
    
    trades_s25.append({
        'date': date,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'direction': 'long'
    })

r = backtest_trades(pd.DataFrame(trades_s25), "S25: Buy 9:28 SL/TP")
print_result(r)
if r: all_results.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY S26: SHORT 9:28 with SL/TP (opposite of S25)
# ─────────────────────────────────────────────────────────────────────────────
header("S26. SHORT 9:28 WITH SL/TP (2-min data)")
print("  Logic: Short at 9:28. SL = +0.3%, TP = -0.5%. Max hold 1hr.\n")

trades_s26 = []
for date in m2['date'].unique():
    day = m2[m2['date'] == date]
    entry_bar = day[(day['hour'] == 9) & (day['minute'] == 28)]
    if len(entry_bar) == 0:
        continue
    
    entry_price = entry_bar.iloc[0]['Close']
    sl_price = entry_price * (1 + 0.003)
    tp_price = entry_price * (1 - 0.005)
    
    post = day[day.index > entry_bar.index[0]]
    max_hold_bars = 30
    
    exit_price = entry_price
    for j in range(min(max_hold_bars, len(post))):
        bar = post.iloc[j]
        if bar['High'] >= sl_price:
            exit_price = sl_price
            break
        if bar['Low'] <= tp_price:
            exit_price = tp_price
            break
        exit_price = bar['Close']
    
    trades_s26.append({
        'date': date,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'direction': 'short'
    })

r = backtest_trades(pd.DataFrame(trades_s26), "S26: Short 9:28 SL/TP")
print_result(r)
if r: all_results.append(r)

# ═════════════════════════════════════════════════════════════════════════════
# FINAL RANKINGS
# ═════════════════════════════════════════════════════════════════════════════
header("FINAL STRATEGY RANKINGS")

# Sort by profit factor
print("  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
print(f"  │ {'Rank':>4}  {'Strategy':<30} {'PF':>5} {'WR%':>5} {'Sharpe':>7} {'Trades':>6} {'P&L$':>10} {'DD%':>6} {'p-val':>7} {'Sig':>3} │")
print("  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤")

sorted_results = sorted(all_results, key=lambda x: x['profit_factor'] if x['profit_factor'] < 100 else 0, reverse=True)

for i, r in enumerate(sorted_results):
    sig = "***" if r['p_value'] < 0.01 else "** " if r['p_value'] < 0.05 else "*  " if r['p_value'] < 0.10 else "   "
    pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "INF"
    print(f"  │ {i+1:4d}  {r['name']:<30} {pf_str:>5} {r['win_rate']:5.1f} {r['sharpe']:+7.2f} {r['trades']:6d} {r['total_pnl']:+10,.0f} {r['max_dd_pct']:5.2f}% {r['p_value']:7.4f} {sig} │")

print("  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘")

# Best by Sharpe
print(f"\n  Best by Sharpe:")
sorted_sharpe = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)
for i, r in enumerate(sorted_sharpe[:5]):
    print(f"    {i+1}. {r['name']:<35} Sharpe={r['sharpe']:+.2f}")

# Best by p-value (statistical significance)
print(f"\n  Best by Statistical Significance:")
sorted_p = sorted(all_results, key=lambda x: x['p_value'])
for i, r in enumerate(sorted_p[:5]):
    sig = "***" if r['p_value'] < 0.01 else "** " if r['p_value'] < 0.05 else "*  " if r['p_value'] < 0.10 else "   "
    print(f"    {i+1}. {r['name']:<35} p={r['p_value']:.4f} {sig}")

# Best by total P&L
print(f"\n  Best by Total P&L:")
sorted_pnl = sorted(all_results, key=lambda x: x['total_pnl'], reverse=True)
for i, r in enumerate(sorted_pnl[:5]):
    print(f"    {i+1}. {r['name']:<35} ${r['total_pnl']:+,.0f}")

# Recommended
print(f"\n  ╔═══════════════════════════════════════════════════════════════╗")
print(f"  ║  STRATEGIES THAT PASS ALL FILTERS:                          ║")
print(f"  ║  PF > 1.0, Sharpe > 0, p < 0.10, Trades > 20              ║")
print(f"  ╠═══════════════════════════════════════════════════════════════╣")

viable = [r for r in all_results if r['profit_factor'] > 1.0 and r['profit_factor'] < 100 
          and r['sharpe'] > 0 and r['p_value'] < 0.10 and r['trades'] > 20]
viable = sorted(viable, key=lambda x: x['sharpe'], reverse=True)

if viable:
    for r in viable:
        sig = "***" if r['p_value'] < 0.01 else "** " if r['p_value'] < 0.05 else "*  " if r['p_value'] < 0.10 else "   "
        print(f"  ║  {r['name']:<30} PF={r['profit_factor']:.2f} S={r['sharpe']:+.2f} p={r['p_value']:.4f} {sig}  ║")
else:
    print(f"  ║  No strategies pass all filters simultaneously.           ║")
    print(f"  ║  Relaxing to: PF > 1.0, Trades > 10                      ║")
    viable2 = [r for r in all_results if r['profit_factor'] > 1.0 and r['profit_factor'] < 100 and r['trades'] > 10]
    viable2 = sorted(viable2, key=lambda x: x['profit_factor'], reverse=True)
    for r in viable2[:5]:
        sig = "***" if r['p_value'] < 0.01 else "** " if r['p_value'] < 0.05 else "*  " if r['p_value'] < 0.10 else "   "
        print(f"  ║  {r['name']:<30} PF={r['profit_factor']:.2f} S={r['sharpe']:+.2f} p={r['p_value']:.4f} {sig}  ║")

print(f"  ╚═══════════════════════════════════════════════════════════════╝")

# Save JSON
results_json = {
    'generated': datetime.now().isoformat(),
    'capital': CAPITAL,
    'strategies': sorted_results
}
json_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'nasdaq_strategies_results.json')
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2, default=str)

print(f"\n  Results saved to:")
print(f"    outputs/nasdaq_strategies_results.txt")
print(f"    outputs/nasdaq_strategies_results.json")
print(f"\n  Analysis complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
