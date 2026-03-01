"""
NASDAQ Edge Analysis — Finding exploitable patterns
====================================================
Analyses:
  1. Hourly return patterns by time-of-day (multi-year)
  2. Open/Close volatility analysis
  3. Pre-NYSE-open strategy (trade 2-5 min before 9:30 ET open)
  4. Overnight gap analysis (close → open)
  5. First 30-min / last 30-min momentum patterns
  6. Day-of-week effects
  7. Mean-reversion after gaps
  8. Intraday session momentum (open→11:00, 11:00→14:00, 14:00→close)
  9. VIX regime interaction
  10. Pre-market momentum → regular session continuation
"""

import sys, os, warnings
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats

LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'nasdaq_edge_analysis.txt')

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

def subheader(title):
    print(f"\n  --- {title} ---\n")

# =============================================================================
# DATA DOWNLOAD
# =============================================================================
header("NASDAQ EDGE ANALYSIS")
print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  Instruments: QQQ (NASDAQ-100 ETF), ^VIX, NQ=F (futures)")

# --- 1. Hourly data (max ~2 years from yfinance) ---
print("\n[1] Downloading QQQ hourly data...")
qqq_1h = yf.download("QQQ", interval="1h", period="2y", progress=False)
if isinstance(qqq_1h.columns, pd.MultiIndex):
    qqq_1h.columns = qqq_1h.columns.get_level_values(0)
print(f"    QQQ 1H: {len(qqq_1h)} bars, {qqq_1h.index[0]} → {qqq_1h.index[-1]}")

# --- 2. 5-minute data (max ~60 days) WITH pre/post market ---
print("[2] Downloading QQQ 5-min data (incl pre/post market)...")
qqq_5m = yf.download("QQQ", interval="5m", period="60d", progress=False, prepost=True)
if isinstance(qqq_5m.columns, pd.MultiIndex):
    qqq_5m.columns = qqq_5m.columns.get_level_values(0)
print(f"    QQQ 5M: {len(qqq_5m)} bars, {qqq_5m.index[0]} → {qqq_5m.index[-1]}")

# --- 3. 2-minute data (max ~60 days) WITH pre/post market ---
print("[3] Downloading QQQ 2-min data (incl pre/post market)...")
qqq_2m = yf.download("QQQ", interval="2m", period="60d", progress=False, prepost=True)
if isinstance(qqq_2m.columns, pd.MultiIndex):
    qqq_2m.columns = qqq_2m.columns.get_level_values(0)
print(f"    QQQ 2M: {len(qqq_2m)} bars, {qqq_2m.index[0]} → {qqq_2m.index[-1]}")

# --- 4. 1-minute data (max ~7 days) WITH pre/post market ---
print("[4] Downloading QQQ 1-min data (incl pre/post market)...")
qqq_1m = yf.download("QQQ", interval="1m", period="7d", progress=False, prepost=True)
if isinstance(qqq_1m.columns, pd.MultiIndex):
    qqq_1m.columns = qqq_1m.columns.get_level_values(0)
print(f"    QQQ 1M: {len(qqq_1m)} bars, {qqq_1m.index[0]} → {qqq_1m.index[-1]}")

# --- 5. Daily data (long history for gap analysis) ---
print("[5] Downloading QQQ daily data (10yr)...")
qqq_daily = yf.download("QQQ", interval="1d", period="10y", progress=False)
if isinstance(qqq_daily.columns, pd.MultiIndex):
    qqq_daily.columns = qqq_daily.columns.get_level_values(0)
print(f"    QQQ Daily: {len(qqq_daily)} bars, {qqq_daily.index[0]} → {qqq_daily.index[-1]}")

# --- 6. VIX for regime analysis ---
print("[6] Downloading VIX data...")
vix_daily = yf.download("^VIX", interval="1d", period="10y", progress=False)
if isinstance(vix_daily.columns, pd.MultiIndex):
    vix_daily.columns = vix_daily.columns.get_level_values(0)
print(f"    VIX: {len(vix_daily)} bars")

# --- 7. NQ futures ---
print("[7] Downloading NQ=F 1H data...")
try:
    nq_1h = yf.download("NQ=F", interval="1h", period="2y", progress=False)
    if isinstance(nq_1h.columns, pd.MultiIndex):
        nq_1h.columns = nq_1h.columns.get_level_values(0)
    print(f"    NQ=F 1H: {len(nq_1h)} bars")
except:
    nq_1h = None
    print("    NQ=F: Failed, continuing with QQQ only")

print("\n  All data downloaded.\n")

# =============================================================================
# ANALYSIS 1: HOURLY RETURN PATTERNS BY TIME-OF-DAY
# =============================================================================
header("1. HOURLY RETURN PATTERNS BY TIME-OF-DAY")

df = qqq_1h.copy()
df['return'] = df['Close'].pct_change()
df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
# Convert to US Eastern
df.index = df.index.tz_convert('US/Eastern')
df['hour'] = df.index.hour
df['minute'] = df.index.minute
df['dow'] = df.index.dayofweek  # 0=Mon
df['date'] = df.index.date

# Only regular trading hours bars
rth = df[(df['hour'] >= 9) & (df['hour'] < 16)].copy()

hourly_stats = rth.groupby('hour').agg(
    mean_ret=('return', 'mean'),
    median_ret=('return', 'median'),
    std_ret=('return', 'std'),
    count=('return', 'count'),
    pos_pct=('return', lambda x: (x > 0).mean()),
    mean_volume=('Volume', 'mean')
).round(6)

# T-test for each hour: is mean significantly different from 0?
for h in hourly_stats.index:
    hour_rets = rth[rth['hour'] == h]['return'].dropna()
    t_stat, p_val = stats.ttest_1samp(hour_rets, 0)
    hourly_stats.loc[h, 't_stat'] = round(t_stat, 3)
    hourly_stats.loc[h, 'p_value'] = round(p_val, 4)

print("  Hour(ET)  Mean Ret%   Median%    Std%   Count  %Pos   Avg Vol     t-stat  p-val")
print("  " + "-"*95)
for h, row in hourly_stats.iterrows():
    sig = "***" if row['p_value'] < 0.01 else "** " if row['p_value'] < 0.05 else "*  " if row['p_value'] < 0.10 else "   "
    print(f"  {h:02d}:00     {row['mean_ret']*100:+.4f}%   {row['median_ret']*100:+.4f}%  {row['std_ret']*100:.4f}%  {int(row['count']):5d}  {row['pos_pct']*100:.1f}%  {row['mean_volume']:,.0f}  {row['t_stat']:+.3f}  {row['p_value']:.4f} {sig}")

# Annualized Sharpe per hour
print("\n  Annualized Sharpe by hour (252 trading days):")
for h in hourly_stats.index:
    if hourly_stats.loc[h, 'std_ret'] > 0:
        sharpe = (hourly_stats.loc[h, 'mean_ret'] / hourly_stats.loc[h, 'std_ret']) * np.sqrt(252)
        print(f"    {h:02d}:00  Sharpe = {sharpe:+.2f}")

# =============================================================================
# ANALYSIS 2: OPEN / CLOSE VOLATILITY — FIRST & LAST HOUR
# =============================================================================
header("2. FIRST HOUR vs LAST HOUR vs MID-DAY")

first_hour = rth[rth['hour'] == 9].copy()
last_hour = rth[rth['hour'] == 15].copy()
mid_day = rth[(rth['hour'] >= 11) & (rth['hour'] <= 13)].copy()

for label, subset in [("First Hour (09:30-10:30)", first_hour),
                       ("Last Hour (15:00-16:00)", last_hour),
                       ("Mid-Day (11:00-14:00)", mid_day)]:
    rets = subset['return'].dropna()
    vol = subset['Volume'].dropna()
    t_stat, p_val = stats.ttest_1samp(rets, 0) if len(rets) > 5 else (0, 1)
    print(f"  {label}:")
    print(f"    Mean return:  {rets.mean()*100:+.4f}%  (t={t_stat:.2f}, p={p_val:.4f})")
    print(f"    Median:       {rets.median()*100:+.4f}%")
    print(f"    Std:          {rets.std()*100:.4f}%")
    print(f"    Volatility:   {rets.std()*100*np.sqrt(252):.1f}% annualised")
    print(f"    %Positive:    {(rets > 0).mean()*100:.1f}%")
    print(f"    Avg Volume:   {vol.mean():,.0f}")
    print(f"    Count:        {len(rets)}")
    print()

# =============================================================================
# ANALYSIS 3: PRE-NYSE-OPEN STRATEGY (Trade 2 min before 9:30 ET)
# =============================================================================
header("3. PRE-NYSE OPEN STRATEGY")
print("  Idea: Enter a trade 2 minutes before NYSE open (9:28 ET),")
print("  ride the opening volatility burst.\n")

subheader("3a. Using 2-minute bars around the open")

df_2m = qqq_2m.copy()
df_2m.index = df_2m.index.tz_convert('US/Eastern')
df_2m['hour'] = df_2m.index.hour
df_2m['minute'] = df_2m.index.minute
df_2m['date'] = df_2m.index.date
df_2m['return'] = df_2m['Close'].pct_change()

# Find the 9:28 bar (entry point) and subsequent bars
pre_open_bars = df_2m[(df_2m['hour'] == 9) & (df_2m['minute'] == 28)].copy()

results_pre_open = []
for idx, row in pre_open_bars.iterrows():
    date = idx.date()
    entry_price = row['Close']  # buy at close of 9:28 bar
    
    # Find subsequent bars on the same date
    day_bars = df_2m[(df_2m['date'] == date) & (df_2m.index > idx)]
    if len(day_bars) < 5:
        continue
    
    # Various exit horizons
    for mins, label in [(2, '2min'), (4, '4min'), (10, '10min'), (30, '30min'), (60, '60min')]:
        n_bars = mins // 2  # 2-min bars
        if n_bars <= len(day_bars):
            exit_price = day_bars.iloc[min(n_bars-1, len(day_bars)-1)]['Close']
            ret = (exit_price - entry_price) / entry_price
            results_pre_open.append({
                'date': date, 'horizon': label,
                'entry': entry_price, 'exit': exit_price,
                'return': ret, 'direction': 'long'
            })
            # Also test SHORT
            ret_short = (entry_price - exit_price) / entry_price
            results_pre_open.append({
                'date': date, 'horizon': label,
                'entry': entry_price, 'exit': exit_price,
                'return': ret_short, 'direction': 'short'
            })

df_pre = pd.DataFrame(results_pre_open)

if len(df_pre) > 0:
    print(f"  Days analysed: {df_pre['date'].nunique()}\n")
    print(f"  {'Horizon':<10} {'Dir':<6} {'Mean%':>8} {'Median%':>8} {'Std%':>8} {'%Pos':>6} {'t-stat':>7} {'p-val':>7} {'Sharpe':>7}")
    print(f"  {'-'*75}")
    
    for horizon in ['2min', '4min', '10min', '30min', '60min']:
        for direction in ['long', 'short']:
            subset = df_pre[(df_pre['horizon'] == horizon) & (df_pre['direction'] == direction)]
            if len(subset) < 5:
                continue
            rets = subset['return']
            t_stat, p_val = stats.ttest_1samp(rets, 0)
            sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
            sig = "***" if p_val < 0.01 else "** " if p_val < 0.05 else "*  " if p_val < 0.10 else "   "
            print(f"  {horizon:<10} {direction:<6} {rets.mean()*100:+.4f}% {rets.median()*100:+.4f}% {rets.std()*100:.4f}% {(rets>0).mean()*100:5.1f}% {t_stat:+.3f}  {p_val:.4f} {sharpe:+.2f} {sig}")
else:
    print("  No 9:28 bars found in 2-min data")

subheader("3b. Using 5-minute bars around the open")

df_5m = qqq_5m.copy()
df_5m.index = df_5m.index.tz_convert('US/Eastern')
df_5m['hour'] = df_5m.index.hour
df_5m['minute'] = df_5m.index.minute
df_5m['date'] = df_5m.index.date
df_5m['return'] = df_5m['Close'].pct_change()

# The 9:30 bar is the first regular session 5-min bar
open_bars = df_5m[(df_5m['hour'] == 9) & (df_5m['minute'] == 30)].copy()

results_5m = []
for idx, row in open_bars.iterrows():
    date = idx.date()
    
    # Pre-market bar just before open (9:25 bar)
    pre_bars = df_5m[(df_5m['date'] == date) & (df_5m['hour'] == 9) & (df_5m['minute'] == 25)]
    if len(pre_bars) == 0:
        continue
    
    pre_close = pre_bars.iloc[0]['Close']
    open_price = row['Open']
    
    # Post-open bars
    day_bars = df_5m[(df_5m['date'] == date) & (df_5m.index >= idx)]
    if len(day_bars) < 6:
        continue
    
    # Pre-market direction (last 30 min before open)
    pre_30 = df_5m[(df_5m['date'] == date) & (df_5m['hour'] == 9) & (df_5m['minute'] < 30)]
    if len(pre_30) >= 2:
        pre_momentum = (pre_30.iloc[-1]['Close'] - pre_30.iloc[0]['Open']) / pre_30.iloc[0]['Open']
    else:
        pre_momentum = 0
    
    for n_bars, label in [(1, '5min'), (2, '10min'), (3, '15min'), (6, '30min'), (12, '60min')]:
        if n_bars <= len(day_bars):
            exit_price = day_bars.iloc[min(n_bars-1, len(day_bars)-1)]['Close']
            ret_long = (exit_price - open_price) / open_price
            ret_short = -ret_long
            
            results_5m.append({
                'date': date, 'horizon': label,
                'return_long': ret_long, 'return_short': ret_short,
                'pre_momentum': pre_momentum,
                'open_price': open_price
            })

df_5m_res = pd.DataFrame(results_5m)

if len(df_5m_res) > 0:
    print(f"  Days analysed: {df_5m_res['date'].nunique()}\n")
    print(f"  Strategy: BUY at open (9:30), sell after N minutes")
    print(f"  {'Horizon':<10} {'Mean%':>8} {'Median%':>8} {'Std%':>8} {'%Pos':>6} {'t-stat':>7} {'p-val':>7} {'Sharpe':>7}")
    print(f"  {'-'*65}")
    
    for horizon in ['5min', '10min', '15min', '30min', '60min']:
        subset = df_5m_res[df_5m_res['horizon'] == horizon]
        rets = subset['return_long']
        t_stat, p_val = stats.ttest_1samp(rets, 0)
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        sig = "***" if p_val < 0.01 else "** " if p_val < 0.05 else "*  " if p_val < 0.10 else "   "
        print(f"  {horizon:<10} {rets.mean()*100:+.4f}% {rets.median()*100:+.4f}% {rets.std()*100:.4f}% {(rets>0).mean()*100:5.1f}% {t_stat:+.3f}  {p_val:.4f} {sharpe:+.2f} {sig}")
    
    subheader("3c. Pre-market momentum → opening trade direction")
    print("  If pre-market (9:00-9:30) is UP → go LONG at open")
    print("  If pre-market (9:00-9:30) is DOWN → go SHORT at open\n")
    
    for horizon in ['5min', '10min', '15min', '30min', '60min']:
        subset = df_5m_res[df_5m_res['horizon'] == horizon].copy()
        # Continuation: follow pre-market direction
        subset['continuation_ret'] = np.where(
            subset['pre_momentum'] > 0,
            subset['return_long'],   # pre-market up → go long
            subset['return_short']   # pre-market down → go short
        )
        # Reversal: fade pre-market direction
        subset['reversal_ret'] = np.where(
            subset['pre_momentum'] > 0,
            subset['return_short'],  # pre-market up → go short (fade)
            subset['return_long']    # pre-market down → go long (fade)
        )
        
        for strat, col in [("Continuation", 'continuation_ret'), ("Reversal", 'reversal_ret')]:
            rets = subset[col].dropna()
            if len(rets) < 5:
                continue
            t_stat, p_val = stats.ttest_1samp(rets, 0)
            sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
            sig = "***" if p_val < 0.01 else "** " if p_val < 0.05 else "*  " if p_val < 0.10 else "   "
            print(f"  {horizon:<8} {strat:<14} mean={rets.mean()*100:+.4f}% %pos={((rets>0).mean())*100:.1f}% sharpe={sharpe:+.2f} p={p_val:.4f} {sig}")

# =============================================================================
# ANALYSIS 4: OVERNIGHT GAP ANALYSIS
# =============================================================================
header("4. OVERNIGHT GAP ANALYSIS (10 YEARS)")

daily = qqq_daily.copy()
daily['prev_close'] = daily['Close'].shift(1)
daily['gap_pct'] = (daily['Open'] - daily['prev_close']) / daily['prev_close'] * 100
daily['intraday_ret'] = (daily['Close'] - daily['Open']) / daily['Open'] * 100
daily['overnight_ret'] = daily['gap_pct']
daily['full_day_ret'] = daily['Close'].pct_change() * 100
daily['dow'] = daily.index.dayofweek
daily = daily.dropna()

print(f"  Period: {daily.index[0].date()} → {daily.index[-1].date()} ({len(daily)} days)\n")

subheader("4a. Gap Statistics")
print(f"  Mean overnight gap:    {daily['gap_pct'].mean():+.4f}%")
print(f"  Median overnight gap:  {daily['gap_pct'].median():+.4f}%")
print(f"  Std:                   {daily['gap_pct'].std():.4f}%")
print(f"  % Gap Up:              {(daily['gap_pct'] > 0).mean()*100:.1f}%")
print(f"  % Gap Down:            {(daily['gap_pct'] < 0).mean()*100:.1f}%")

subheader("4b. Gap Fill Rate (does the gap get filled during the day?)")

# Gap up: does price come back down to prev close?
gap_up = daily[daily['gap_pct'] > 0.1].copy()
gap_up['filled'] = gap_up['Low'] <= gap_up['prev_close']
gap_down = daily[daily['gap_pct'] < -0.1].copy()
gap_down['filled'] = gap_down['High'] >= gap_down['prev_close']

print(f"  Gap UP (>0.1%) fill rate:   {gap_up['filled'].mean()*100:.1f}% of {len(gap_up)} days")
print(f"  Gap DOWN (<-0.1%) fill rate: {gap_down['filled'].mean()*100:.1f}% of {len(gap_down)} days")

# By gap size
for threshold in [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]:
    gu = daily[daily['gap_pct'] > threshold]
    gd = daily[daily['gap_pct'] < -threshold]
    if len(gu) > 10:
        gu_filled = (gu['Low'] <= gu['prev_close']).mean()
        gu_intraday = gu['intraday_ret'].mean()
        print(f"    Gap > +{threshold:.1f}%: {len(gu):4d} days, fill={gu_filled*100:.0f}%, intraday mean={gu_intraday:+.3f}%")
    if len(gd) > 10:
        gd_filled = (gd['High'] >= gd['prev_close']).mean()
        gd_intraday = gd['intraday_ret'].mean()
        print(f"    Gap < -{threshold:.1f}%: {len(gd):4d} days, fill={gd_filled*100:.0f}%, intraday mean={gd_intraday:+.3f}%")

subheader("4c. Gap Fade Strategy (trade against the gap direction)")
print("  Strategy: If gap UP → SHORT at open, exit at close")
print("            If gap DOWN → LONG at open, exit at close\n")

for min_gap in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
    fade_up = daily[daily['gap_pct'] > min_gap].copy()
    fade_up['fade_ret'] = -fade_up['intraday_ret']  # short
    fade_down = daily[daily['gap_pct'] < -min_gap].copy()
    fade_down['fade_ret'] = fade_down['intraday_ret']  # long
    
    all_fades = pd.concat([fade_up['fade_ret'], fade_down['fade_ret']])
    if len(all_fades) < 10:
        continue
    t_stat, p_val = stats.ttest_1samp(all_fades, 0)
    sharpe = (all_fades.mean() / all_fades.std()) * np.sqrt(252) if all_fades.std() > 0 else 0
    sig = "***" if p_val < 0.01 else "** " if p_val < 0.05 else "*  " if p_val < 0.10 else "   "
    print(f"  Gap >{min_gap:.2f}%: {len(all_fades):4d} trades, mean={all_fades.mean():+.4f}%, %pos={(all_fades>0).mean()*100:.1f}%, sharpe={sharpe:+.2f}, p={p_val:.4f} {sig}")

# =============================================================================
# ANALYSIS 5: DAY-OF-WEEK EFFECTS
# =============================================================================
header("5. DAY-OF-WEEK EFFECTS (10 YEARS)")

dow_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}

print(f"  {'Day':<12} {'Mean%':>8} {'Median%':>8} {'Std%':>8} {'%Pos':>6} {'Count':>6} {'t-stat':>7} {'p-val':>7}")
print(f"  {'-'*70}")

for d in range(5):
    rets = daily[daily['dow'] == d]['full_day_ret']
    t_stat, p_val = stats.ttest_1samp(rets, 0)
    sig = "***" if p_val < 0.01 else "** " if p_val < 0.05 else "*  " if p_val < 0.10 else "   "
    print(f"  {dow_names[d]:<12} {rets.mean():+.4f}% {rets.median():+.4f}% {rets.std():.4f}% {(rets>0).mean()*100:5.1f}% {len(rets):5d}  {t_stat:+.3f}  {p_val:.4f} {sig}")

# Overnight vs Intraday by day
subheader("5b. Overnight vs Intraday by Day-of-Week")
print(f"  {'Day':<12} {'Overnight%':>10} {'Intraday%':>10} {'Better':>10}")
print(f"  {'-'*45}")
for d in range(5):
    subset = daily[daily['dow'] == d]
    ovn = subset['overnight_ret'].mean()
    intra = subset['intraday_ret'].mean()
    better = "OVERNIGHT" if abs(ovn) > abs(intra) and ovn > 0 else "INTRADAY" if intra > 0 else "NEITHER"
    print(f"  {dow_names[d]:<12} {ovn:+.4f}%    {intra:+.4f}%    {better}")

# =============================================================================
# ANALYSIS 6: INTRADAY SESSION ANALYSIS (HOURLY DATA)
# =============================================================================
header("6. INTRADAY SESSION RETURNS (Hourly Data)")

# Session returns
rth_daily = rth.groupby('date').agg(
    open_price=('Open', 'first'),
    close_price=('Close', 'last'),
    high_price=('High', 'max'),
    low_price=('Low', 'min')
)

# First 30 min vs rest (approximate with hourly data)
# Open→10:00 (first 30 min roughly)
# 10:00→14:00 (mid-day chop)
# 14:00→16:00 (power hour)

sessions = {
    'Open Rush (09-10)': (9, 10),
    'Morning (10-12)': (10, 12),
    'Lunch Chop (12-14)': (12, 14),
    'Power Hour (14-16)': (14, 16),
}

print(f"  {'Session':<22} {'Mean%':>8} {'Median%':>8} {'Std%':>8} {'%Pos':>6} {'Sharpe':>7} {'Count':>6}")
print(f"  {'-'*72}")

for label, (h_start, h_end) in sessions.items():
    session_bars = rth[(rth['hour'] >= h_start) & (rth['hour'] < h_end)]
    
    # Calculate session return per day
    session_by_day = session_bars.groupby('date').agg(
        s_open=('Open', 'first'),
        s_close=('Close', 'last')
    )
    session_by_day['ret'] = (session_by_day['s_close'] - session_by_day['s_open']) / session_by_day['s_open'] * 100
    
    rets = session_by_day['ret'].dropna()
    if len(rets) < 10:
        continue
    t_stat, p_val = stats.ttest_1samp(rets, 0)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    sig = "***" if p_val < 0.01 else "** " if p_val < 0.05 else "*  " if p_val < 0.10 else "   "
    print(f"  {label:<22} {rets.mean():+.4f}% {rets.median():+.4f}% {rets.std():.4f}% {(rets>0).mean()*100:5.1f}% {sharpe:+.2f}  {len(rets):5d} {sig}")

# =============================================================================
# ANALYSIS 7: MEAN REVERSION AFTER LARGE MOVES
# =============================================================================
header("7. MEAN REVERSION AFTER LARGE HOURLY MOVES")

rth_sorted = rth.sort_index()
rth_sorted['next_ret'] = rth_sorted['return'].shift(-1)

for threshold in [0.5, 0.75, 1.0, 1.5, 2.0]:
    # After big up move → does it reverse?
    big_up = rth_sorted[rth_sorted['return'] * 100 > threshold]
    big_down = rth_sorted[rth_sorted['return'] * 100 < -threshold]
    
    if len(big_up) > 10:
        next_after_up = big_up['next_ret'].dropna() * 100
        t_stat, p_val = stats.ttest_1samp(next_after_up, 0)
        print(f"  After hourly move > +{threshold:.1f}%: next hour = {next_after_up.mean():+.4f}% (n={len(next_after_up)}, %neg={(next_after_up<0).mean()*100:.0f}%, p={p_val:.3f})")
    
    if len(big_down) > 10:
        next_after_down = big_down['next_ret'].dropna() * 100
        t_stat, p_val = stats.ttest_1samp(next_after_down, 0)
        print(f"  After hourly move < -{threshold:.1f}%: next hour = {next_after_down.mean():+.4f}% (n={len(next_after_down)}, %pos={(next_after_down>0).mean()*100:.0f}%, p={p_val:.3f})")
    print()

# =============================================================================
# ANALYSIS 8: VIX REGIME × INTRADAY PATTERNS
# =============================================================================
header("8. VIX REGIME × INTRADAY PATTERNS")

daily_with_vix = daily.copy()
daily_with_vix.index = pd.to_datetime(daily_with_vix.index)
vix_close = vix_daily['Close'].copy()
vix_close.index = pd.to_datetime(vix_close.index)
if hasattr(vix_close.index, 'tz') and vix_close.index.tz is not None:
    vix_close.index = vix_close.index.tz_localize(None)
if hasattr(daily_with_vix.index, 'tz') and daily_with_vix.index.tz is not None:
    daily_with_vix.index = daily_with_vix.index.tz_localize(None)

daily_with_vix['vix'] = vix_close.reindex(daily_with_vix.index, method='ffill')
daily_with_vix = daily_with_vix.dropna(subset=['vix'])

# VIX regimes
vix_bins = [(0, 15, 'Low VIX (<15)'), (15, 20, 'Med VIX (15-20)'), (20, 30, 'High VIX (20-30)'), (30, 100, 'Very High (30+)')]

print(f"  {'VIX Regime':<20} {'Count':>6} {'Mean Ret%':>10} {'Std%':>8} {'%Pos':>6} {'Gap Mean%':>10} {'Intraday%':>10}")
print(f"  {'-'*75}")

for low, high, label in vix_bins:
    subset = daily_with_vix[(daily_with_vix['vix'] >= low) & (daily_with_vix['vix'] < high)]
    if len(subset) < 10:
        continue
    print(f"  {label:<20} {len(subset):5d}  {subset['full_day_ret'].mean():+.4f}%  {subset['full_day_ret'].std():.4f}% {(subset['full_day_ret']>0).mean()*100:5.1f}% {subset['overnight_ret'].mean():+.4f}%   {subset['intraday_ret'].mean():+.4f}%")

subheader("8b. Gap Fade by VIX Regime")
print("  Does gap fading work better in high/low VIX?\n")

for low, high, label in vix_bins:
    subset = daily_with_vix[(daily_with_vix['vix'] >= low) & (daily_with_vix['vix'] < high)]
    if len(subset) < 20:
        continue
    
    gap_up = subset[subset['gap_pct'] > 0.25]
    gap_down = subset[subset['gap_pct'] < -0.25]
    
    fade_rets = []
    if len(gap_up) > 0:
        fade_rets.extend((-gap_up['intraday_ret']).tolist())
    if len(gap_down) > 0:
        fade_rets.extend(gap_down['intraday_ret'].tolist())
    
    if len(fade_rets) > 10:
        fade_rets = pd.Series(fade_rets)
        t_stat, p_val = stats.ttest_1samp(fade_rets, 0)
        sharpe = (fade_rets.mean() / fade_rets.std()) * np.sqrt(252) if fade_rets.std() > 0 else 0
        sig = "***" if p_val < 0.01 else "** " if p_val < 0.05 else "*  " if p_val < 0.10 else "   "
        print(f"  {label:<20} n={len(fade_rets):4d}, mean={fade_rets.mean():+.4f}%, sharpe={sharpe:+.2f}, p={p_val:.4f} {sig}")

# =============================================================================
# ANALYSIS 9: FIRST 5-MIN BAR MOMENTUM → REST-OF-DAY
# =============================================================================
header("9. FIRST 5-MIN BAR → REST-OF-DAY PREDICTION")

if len(df_5m_res) > 0:
    df_5m_day = df_5m.copy()
    dates = df_5m_day['date'].unique()
    
    first_bar_results = []
    for date in dates:
        day_data = df_5m_day[df_5m_day['date'] == date]
        # First 5-min bar at 9:30
        first_bar = day_data[(day_data['hour'] == 9) & (day_data['minute'] == 30)]
        if len(first_bar) == 0:
            continue
        
        first_ret = (first_bar.iloc[0]['Close'] - first_bar.iloc[0]['Open']) / first_bar.iloc[0]['Open']
        
        # Rest of day: from 9:35 to close
        rest = day_data[day_data.index > first_bar.index[0]]
        if len(rest) < 10:
            continue
        
        rest_ret = (rest.iloc[-1]['Close'] - first_bar.iloc[0]['Close']) / first_bar.iloc[0]['Close']
        
        first_bar_results.append({
            'date': date,
            'first_5min_ret': first_ret,
            'rest_of_day_ret': rest_ret,
            'first_dir': 'up' if first_ret > 0 else 'down'
        })
    
    fb_df = pd.DataFrame(first_bar_results)
    
    if len(fb_df) > 10:
        # If first bar is UP, does rest of day continue?
        up_days = fb_df[fb_df['first_dir'] == 'up']
        down_days = fb_df[fb_df['first_dir'] == 'down']
        
        print(f"  Days analysed: {len(fb_df)}")
        print(f"  First 5-min UP:   {len(up_days)} days ({len(up_days)/len(fb_df)*100:.0f}%)")
        print(f"  First 5-min DOWN: {len(down_days)} days ({len(down_days)/len(fb_df)*100:.0f}%)\n")
        
        if len(up_days) > 5:
            cont = up_days['rest_of_day_ret']
            print(f"  First bar UP → rest of day:    mean={cont.mean()*100:+.4f}%, %pos={(cont>0).mean()*100:.0f}%")
        if len(down_days) > 5:
            cont = down_days['rest_of_day_ret']
            print(f"  First bar DOWN → rest of day:  mean={cont.mean()*100:+.4f}%, %pos={(cont>0).mean()*100:.0f}%")
        
        # Continuation vs reversal strategy
        fb_df['continuation_pnl'] = np.where(
            fb_df['first_dir'] == 'up',
            fb_df['rest_of_day_ret'],
            -fb_df['rest_of_day_ret']
        )
        fb_df['reversal_pnl'] = -fb_df['continuation_pnl']
        
        for strat, col in [("Continuation", 'continuation_pnl'), ("Reversal", 'reversal_pnl')]:
            rets = fb_df[col]
            t_stat, p_val = stats.ttest_1samp(rets, 0)
            sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
            sig = "***" if p_val < 0.01 else "** " if p_val < 0.05 else "*  " if p_val < 0.10 else "   "
            print(f"\n  {strat} strategy: mean={rets.mean()*100:+.4f}%, sharpe={sharpe:+.2f}, p={p_val:.4f} {sig}")

# =============================================================================
# ANALYSIS 10: CONSECUTIVE DAYS PATTERN
# =============================================================================
header("10. CONSECUTIVE DAYS — MOMENTUM vs REVERSAL")

daily_sorted = daily.sort_index()
daily_sorted['prev_ret'] = daily_sorted['full_day_ret'].shift(1)
daily_sorted['prev2_ret'] = daily_sorted['full_day_ret'].shift(2)
daily_sorted = daily_sorted.dropna()

# After N consecutive up/down days
daily_sorted['streak'] = 0
streak = 0
for i in range(len(daily_sorted)):
    ret = daily_sorted.iloc[i]['prev_ret']
    if ret > 0:
        streak = streak + 1 if streak > 0 else 1
    elif ret < 0:
        streak = streak - 1 if streak < 0 else -1
    else:
        streak = 0
    daily_sorted.iloc[i, daily_sorted.columns.get_loc('streak')] = streak

print(f"  {'Streak':<20} {'Count':>6} {'Next Day%':>10} {'%Pos':>6} {'t-stat':>7} {'p-val':>7}")
print(f"  {'-'*60}")

for s_val, s_label in [(1, "After 1 up day"), (2, "After 2 up days"), (3, "After 3+ up days"),
                        (-1, "After 1 down day"), (-2, "After 2 down days"), (-3, "After 3+ down days")]:
    if s_val > 0:
        mask = daily_sorted['streak'] >= s_val
        if s_val < 3:
            mask = daily_sorted['streak'] == s_val
    else:
        mask = daily_sorted['streak'] <= s_val
        if s_val > -3:
            mask = daily_sorted['streak'] == s_val
    
    subset = daily_sorted[mask]
    if len(subset) < 10:
        continue
    
    rets = subset['full_day_ret']
    t_stat, p_val = stats.ttest_1samp(rets, 0)
    sig = "***" if p_val < 0.01 else "** " if p_val < 0.05 else "*  " if p_val < 0.10 else "   "
    print(f"  {s_label:<20} {len(subset):5d}  {rets.mean():+.4f}%  {(rets>0).mean()*100:5.1f}% {t_stat:+.3f}  {p_val:.4f} {sig}")

# =============================================================================
# ANALYSIS 11: OPENING RANGE BREAKOUT
# =============================================================================
header("11. OPENING RANGE BREAKOUT (5-min data)")

if len(qqq_5m) > 100:
    df_orb = df_5m.copy()
    dates = df_orb['date'].unique()
    
    orb_results = []
    for date in dates:
        day = df_orb[df_orb['date'] == date]
        
        # First 15 min (3 bars of 5-min) → opening range
        first_15 = day[(day['hour'] == 9) & (day['minute'] >= 30) & (day['minute'] < 45)]
        if len(first_15) < 2:
            continue
        
        or_high = first_15['High'].max()
        or_low = first_15['Low'].min()
        or_range = (or_high - or_low) / or_low * 100
        
        # Rest of day
        rest = day[day.index > first_15.index[-1]]
        if len(rest) < 10:
            continue
        
        # Did it break above OR high?
        broke_up = (rest['High'] > or_high).any()
        # Did it break below OR low?
        broke_down = (rest['Low'] < or_low).any()
        
        # If breakout up → long from OR_high to close
        day_close = rest.iloc[-1]['Close']
        
        if broke_up:
            long_ret = (day_close - or_high) / or_high * 100
        else:
            long_ret = None
        
        if broke_down:
            short_ret = (or_low - day_close) / or_low * 100
        else:
            short_ret = None
        
        orb_results.append({
            'date': date, 'or_range': or_range,
            'broke_up': broke_up, 'broke_down': broke_down,
            'long_ret': long_ret, 'short_ret': short_ret,
            'both_broke': broke_up and broke_down
        })
    
    orb_df = pd.DataFrame(orb_results)
    
    if len(orb_df) > 10:
        print(f"  Days analysed: {len(orb_df)}")
        print(f"  Avg opening range (15min): {orb_df['or_range'].mean():.3f}%")
        print(f"  Broke UP:    {orb_df['broke_up'].mean()*100:.1f}%")
        print(f"  Broke DOWN:  {orb_df['broke_down'].mean()*100:.1f}%")
        print(f"  Both sides:  {orb_df['both_broke'].mean()*100:.1f}%\n")
        
        # Breakout long performance
        long_trades = orb_df[orb_df['long_ret'].notna()]
        if len(long_trades) > 5:
            rets = long_trades['long_ret']
            print(f"  Breakout LONG:  n={len(rets)}, mean={rets.mean():+.3f}%, %pos={(rets>0).mean()*100:.0f}%")
        
        short_trades = orb_df[orb_df['short_ret'].notna()]
        if len(short_trades) > 5:
            rets = short_trades['short_ret']
            print(f"  Breakout SHORT: n={len(rets)}, mean={rets.mean():+.3f}%, %pos={(rets>0).mean()*100:.0f}%")

# =============================================================================
# ANALYSIS 12: 2 MINUTES BEFORE NYSE OPEN — DETAILED
# =============================================================================
header("12. TRADE 2 MINUTES BEFORE NYSE OPEN — DETAILED")

if len(qqq_1m) > 100:
    df_1min = qqq_1m.copy()
    df_1min.index = df_1min.index.tz_convert('US/Eastern')
    df_1min['hour'] = df_1min.index.hour
    df_1min['minute'] = df_1min.index.minute
    df_1min['date'] = df_1min.index.date
    
    # Find 9:28 bar (2 min before open)
    entry_bars = df_1min[(df_1min['hour'] == 9) & (df_1min['minute'] == 28)]
    
    print(f"  1-minute data: {len(df_1min)} bars, {df_1min.index[0]} → {df_1min.index[-1]}")
    print(f"  9:28 entry bars found: {len(entry_bars)}\n")
    
    results_1m = []
    for idx, row in entry_bars.iterrows():
        date = idx.date()
        entry_price = row['Close']
        
        day_bars = df_1min[(df_1min['date'] == date) & (df_1min.index > idx)]
        if len(day_bars) < 30:
            continue
        
        # Pre-open momentum (9:00-9:28)
        pre_open = df_1min[(df_1min['date'] == date) & (df_1min['hour'] == 9) & (df_1min['minute'] < 28)]
        pre_mom = 0
        if len(pre_open) > 5:
            pre_mom = (pre_open.iloc[-1]['Close'] - pre_open.iloc[0]['Open']) / pre_open.iloc[0]['Open']
        
        for mins in [1, 2, 3, 5, 10, 15, 30, 60]:
            if mins <= len(day_bars):
                exit_price = day_bars.iloc[min(mins-1, len(day_bars)-1)]['Close']
                ret = (exit_price - entry_price) / entry_price
                results_1m.append({
                    'date': date, 'mins': mins,
                    'entry': entry_price, 'exit': exit_price,
                    'ret_long': ret, 'ret_short': -ret,
                    'pre_momentum': pre_mom
                })
    
    df_1m_res = pd.DataFrame(results_1m)
    
    if len(df_1m_res) > 0:
        print(f"  Enter at 9:28 (2 min before open), results:\n")
        print(f"  {'Exit After':<12} {'LONG%':>8} {'SHORT%':>8} {'Std%':>8} {'Long%Pos':>8} {'Short%Pos':>9}")
        print(f"  {'-'*55}")
        
        for mins in [1, 2, 3, 5, 10, 15, 30, 60]:
            subset = df_1m_res[df_1m_res['mins'] == mins]
            if len(subset) < 3:
                continue
            print(f"  {mins} min       {subset['ret_long'].mean()*100:+.4f}% {subset['ret_short'].mean()*100:+.4f}% {subset['ret_long'].std()*100:.4f}% {(subset['ret_long']>0).mean()*100:6.1f}% {(subset['ret_short']>0).mean()*100:7.1f}%")
        
        print(f"\n  With pre-market momentum filter:")
        for mins in [5, 10, 15, 30]:
            subset = df_1m_res[df_1m_res['mins'] == mins]
            if len(subset) < 3:
                continue
            # Follow pre-market momentum
            subset_copy = subset.copy()
            subset_copy['strat_ret'] = np.where(
                subset_copy['pre_momentum'] > 0,
                subset_copy['ret_long'],
                subset_copy['ret_short']
            )
            rets = subset_copy['strat_ret']
            print(f"    {mins} min: follow pre-mkt → mean={rets.mean()*100:+.4f}%, %pos={(rets>0).mean()*100:.0f}%")
else:
    print("  Insufficient 1-minute data for this analysis.")

# =============================================================================
# SUMMARY
# =============================================================================
header("SUMMARY OF FINDINGS")

print("""
  This analysis examined NASDAQ (QQQ) across multiple timeframes looking
  for exploitable edges. Key areas investigated:

  1. TIME-OF-DAY: Which hours have statistically significant returns?
  2. OPEN/CLOSE: Is the first/last hour more profitable?
  3. PRE-OPEN ENTRY: Does entering 2 min before NYSE open capture edge?
  4. GAP ANALYSIS: Do overnight gaps get filled? Is gap-fading profitable?
  5. DAY-OF-WEEK: Do certain days have consistent directional bias?
  6. SESSION SPLITS: Open rush vs mid-day vs power hour
  7. MEAN REVERSION: Do large hourly moves reverse?
  8. VIX REGIMES: Do patterns change with volatility?
  9. FIRST BAR DIRECTION: Does the first 5-min bar predict the day?
  10. CONSECUTIVE DAYS: Momentum vs reversal after streaks
  11. OPENING RANGE BREAKOUT: Classic 15-min ORB strategy
  12. 2-MIN PRE-OPEN: Detailed 1-minute analysis

  Results saved to: outputs/nasdaq_edge_analysis.txt
""")

print(f"\n  Analysis complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
