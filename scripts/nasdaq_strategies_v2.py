"""
NASDAQ Strategy Backtester V2 — Enhanced with Buy-and-Hold Benchmark
=====================================================================
Compares all strategies against a buy-and-hold benchmark, then implements
multiple profitability improvements:

IMPROVEMENTS IMPLEMENTED:
  E1. SMA200 Trend Filter — Only trade long when QQQ > 200-day SMA
  E2. RSI Oversold Confirmation — Only mean-revert when RSI < 35
  E3. ATR-Based Dynamic SL/TP — Volatility-scaled stops
  E4. Multi-Day Holds (2-5 days) — Capture extended bounce
  E5. Kelly Criterion Position Sizing — Optimal bet fraction
  E6. Depth Filter — Bigger streak drops = bigger positions
  E7. Monthly Seasonality Filter — Nov-Apr (best 6 months)
  E8. Combined Portfolio — Stack uncorrelated edges for smoother equity curve
  E9. Leveraged Sizing on High-Conviction — 10% equity when multiple signals agree
  E10. Trailing Stop Exits — Lock in profits on multi-day holds
  E11. VIX Contango Filter — VIX term structure as sentiment
  E12. Drawdown-based risk throttle — Reduce size during drawdowns

Capital: $25,000 (prop firm context)
Instrument: QQQ / NAS100
"""

import sys, os, warnings, json
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats

LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'nasdaq_v2_results.txt')

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
# DATA DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════
header("NASDAQ STRATEGY V2 — ENHANCED BACKTESTER + BUY & HOLD BENCHMARK")
print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

print("\n[1] QQQ daily (10yr)...")
qqq_daily = yf.download("QQQ", interval="1d", period="10y", progress=False)
if isinstance(qqq_daily.columns, pd.MultiIndex):
    qqq_daily.columns = qqq_daily.columns.get_level_values(0)
print(f"    {len(qqq_daily)} bars | {qqq_daily.index[0].strftime('%Y-%m-%d')} → {qqq_daily.index[-1].strftime('%Y-%m-%d')}")

print("[2] VIX daily (10yr)...")
vix_daily = yf.download("^VIX", interval="1d", period="10y", progress=False)
if isinstance(vix_daily.columns, pd.MultiIndex):
    vix_daily.columns = vix_daily.columns.get_level_values(0)
print(f"    {len(vix_daily)} bars")

print("[3] VIX 3-month futures proxy (VIX3M)...")
try:
    vix3m = yf.download("^VIX3M", interval="1d", period="10y", progress=False)
    if isinstance(vix3m.columns, pd.MultiIndex):
        vix3m.columns = vix3m.columns.get_level_values(0)
    print(f"    {len(vix3m)} bars")
    has_vix3m = len(vix3m) > 100
except:
    vix3m = pd.DataFrame()
    has_vix3m = False
    print("    Not available — skipping VIX contango filter")

print("  Downloads complete.\n")

# ═════════════════════════════════════════════════════════════════════════════
# PREPARE DATA
# ═════════════════════════════════════════════════════════════════════════════
daily = qqq_daily.copy()
daily['prev_close'] = daily['Close'].shift(1)
daily['gap_pct'] = (daily['Open'] - daily['prev_close']) / daily['prev_close'] * 100
daily['intraday_ret_pct'] = (daily['Close'] - daily['Open']) / daily['Open'] * 100
daily['full_ret_pct'] = daily['Close'].pct_change() * 100
daily['overnight_ret_pct'] = daily['gap_pct']
daily['dow'] = daily.index.dayofweek
daily['month'] = daily.index.month
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

# Add VIX3M for contango
if has_vix3m:
    vix3m_close = vix3m['Close'].copy()
    vix3m_close.index = pd.to_datetime(vix3m_close.index)
    if hasattr(vix3m_close.index, 'tz') and vix3m_close.index.tz is not None:
        vix3m_close.index = vix3m_close.index.tz_localize(None)
    daily['vix3m'] = vix3m_close.reindex(daily.index, method='ffill')
    daily['vix_contango'] = (daily['vix3m'] - daily['vix']) / daily['vix'] * 100  # positive = contango (bullish)
else:
    daily['vix_contango'] = 0

# Technical indicators
daily['sma50'] = daily['Close'].rolling(50).mean()
daily['sma200'] = daily['Close'].rolling(200).mean()
daily['rsi'] = 100 - 100 / (1 + daily['Close'].diff().clip(lower=0).rolling(14).mean() /
                              daily['Close'].diff().clip(upper=0).abs().rolling(14).mean())
daily['atr'] = pd.concat([
    daily['High'] - daily['Low'],
    (daily['High'] - daily['Close'].shift(1)).abs(),
    (daily['Low'] - daily['Close'].shift(1)).abs()
], axis=1).max(axis=1).rolling(14).mean()
daily['atr_pct'] = daily['atr'] / daily['Close'] * 100

# Consecutive days streak
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

# Cumulative drop during down streak
daily['streak_drop'] = 0.0
cum_drop = 0.0
drops = []
for i in range(len(daily)):
    if daily.iloc[i]['full_ret_pct'] < 0:
        cum_drop += daily.iloc[i]['full_ret_pct']
    else:
        cum_drop = 0.0
    drops.append(cum_drop)
daily['streak_drop'] = drops
daily['prev_streak_drop'] = daily['streak_drop'].shift(1)

print(f"  Data ready: {len(daily)} trading days")
print(f"  Period: {daily.index[0].strftime('%Y-%m-%d')} → {daily.index[-1].strftime('%Y-%m-%d')}")
print(f"  SMA200 valid from: {daily.dropna(subset=['sma200']).index[0].strftime('%Y-%m-%d')}")

# ═════════════════════════════════════════════════════════════════════════════
# BUY AND HOLD BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════
header("BUY AND HOLD BENCHMARK — QQQ")

bh_start = daily.iloc[0]['Close']
bh_end = daily.iloc[-1]['Close']
bh_years = (daily.index[-1] - daily.index[0]).days / 365.25
bh_total_return = (bh_end - bh_start) / bh_start * 100
bh_cagr = ((bh_end / bh_start) ** (1 / bh_years) - 1) * 100

# B&H drawdown
bh_equity = daily['Close'] / daily['Close'].iloc[0]
bh_peak = bh_equity.cummax()
bh_dd = (bh_peak - bh_equity) / bh_peak * 100
bh_max_dd = bh_dd.max()

# B&H Sharpe
bh_daily_returns = daily['full_ret_pct'] / 100
bh_sharpe = (bh_daily_returns.mean() / bh_daily_returns.std()) * np.sqrt(252) if bh_daily_returns.std() > 0 else 0

# B&H with 5% position (matching our strategies)
bh_5pct_total = bh_total_return * 0.05  # Linear approximation for small positions

print(f"  QQQ: ${bh_start:.2f} → ${bh_end:.2f}")
print(f"  Period: {bh_years:.1f} years")
print(f"  Total Return: {bh_total_return:+.1f}%")
print(f"  CAGR: {bh_cagr:+.2f}%")
print(f"  Max Drawdown: {bh_max_dd:.1f}%")
print(f"  Sharpe (daily): {bh_sharpe:+.2f}")
print(f"")
print(f"  --- Adjusted for 5% position sizing (like our strategies) ---")
print(f"  Total Return on Capital: {bh_5pct_total:+.2f}%")
print(f"  Dollar P&L on $25K: ${25000 * bh_5pct_total / 100:+,.0f}")
print(f"  Max DD on capital: {bh_max_dd * 0.05:.2f}%")

bh_benchmark = {
    'total_return_pct': bh_total_return,
    'cagr_pct': bh_cagr,
    'max_dd_pct': bh_max_dd,
    'sharpe': bh_sharpe,
    'years': bh_years,
    'adjusted_5pct_return': bh_5pct_total,
    'adjusted_5pct_dd': bh_max_dd * 0.05,
    'adjusted_5pct_pnl': 25000 * bh_5pct_total / 100,
}


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE (ENHANCED)
# ═════════════════════════════════════════════════════════════════════════════
CAPITAL = 25000.0
SLIPPAGE_PCT = 0.01  # 1bp per side

def backtest_trades(trades_df, name, capital=CAPITAL, default_size_pct=5.0):
    """Enhanced backtest engine with variable position sizing."""
    if len(trades_df) == 0:
        return None
    
    results = []
    equity = capital
    peak = capital
    max_dd = 0
    max_dd_pct = 0
    equity_curve = [capital]
    
    for _, t in trades_df.iterrows():
        entry = t['entry_price']
        exit_p = t['exit_price']
        direction = t.get('direction', 'long')
        size_pct = t.get('size_pct', default_size_pct)
        
        # Slippage
        if direction == 'long':
            adj_entry = entry * (1 + SLIPPAGE_PCT / 100)
            adj_exit = exit_p * (1 - SLIPPAGE_PCT / 100)
            pnl_pct = (adj_exit - adj_entry) / adj_entry * 100
        else:
            adj_entry = entry * (1 - SLIPPAGE_PCT / 100)
            adj_exit = exit_p * (1 + SLIPPAGE_PCT / 100)
            pnl_pct = (adj_entry - adj_exit) / adj_entry * 100
        
        position_size = equity * size_pct / 100
        pnl_dollar = position_size * pnl_pct / 100
        
        equity += pnl_dollar
        peak = max(peak, equity)
        dd = peak - equity
        dd_pct_val = dd / peak * 100
        max_dd = max(max_dd, dd)
        max_dd_pct = max(max_dd_pct, dd_pct_val)
        
        results.append({
            'date': t['date'],
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'equity': equity,
            'drawdown_pct': dd_pct_val,
            'size_pct': size_pct,
        })
        equity_curve.append(equity)
    
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
    
    daily_rets = res_df['pnl_pct']
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() > 0 else 0
    expectancy = daily_rets.mean()
    
    t_stat, p_val = stats.ttest_1samp(daily_rets, 0) if len(daily_rets) > 2 else (0, 1)
    
    if len(res_df) > 1:
        first_date = pd.to_datetime(res_df['date'].iloc[0])
        last_date = pd.to_datetime(res_df['date'].iloc[-1])
        years = max((last_date - first_date).days / 365.25, 0.1)
        trades_per_year = total_trades / years
    else:
        trades_per_year = 0
        years = 0
    
    # vs Buy & Hold
    bh_annual = bh_benchmark['cagr_pct']
    strat_annual = total_pnl_pct / max(years, 0.1)
    alpha_vs_bh = strat_annual - (bh_annual * 0.05)  # Adjusted for 5% sizing

    # Calmar ratio (return / max DD)
    calmar = strat_annual / max_dd_pct if max_dd_pct > 0 else float('inf')
    
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
        'calmar': calmar,
        'expectancy_pct': expectancy,
        't_stat': t_stat,
        'p_value': p_val,
        'final_equity': equity,
        'alpha_vs_bh': alpha_vs_bh,
        'equity_curve': equity_curve,
    }


def print_result(r, show_bh=True):
    if r is None:
        print("  NO TRADES\n")
        return
    sig = "***" if r['p_value'] < 0.01 else "** " if r['p_value'] < 0.05 else "*  " if r['p_value'] < 0.10 else "   "
    print(f"  Trades:         {r['trades']:,d} ({r['trades_per_year']:.0f}/yr over {r['years']:.1f} yrs)")
    print(f"  Win Rate:       {r['win_rate']:.1f}%")
    print(f"  Avg Win:        +{r['avg_win_pct']:.3f}%    Avg Loss: -{r['avg_loss_pct']:.3f}%")
    print(f"  Profit Factor:  {r['profit_factor']:.2f}")
    print(f"  Total P&L:      ${r['total_pnl']:+,.2f} ({r['total_pnl_pct']:+.2f}%)")
    print(f"  Max Drawdown:   {r['max_dd_pct']:.2f}%")
    print(f"  Sharpe (ann):   {r['sharpe']:+.2f}")
    print(f"  Calmar Ratio:   {r['calmar']:.2f}")
    print(f"  Expectancy:     {r['expectancy_pct']:+.4f}% per trade")
    print(f"  t-stat:         {r['t_stat']:+.3f}  p={r['p_value']:.4f} {sig}")
    print(f"  Final Equity:   ${r['final_equity']:,.2f} (from $25,000)")
    if show_bh:
        print(f"  vs Buy&Hold:    {r['alpha_vs_bh']:+.2f}% annual alpha")
    print()


all_results = []

# ═════════════════════════════════════════════════════════════════════════════
# BASELINE STRATEGIES (from V1 — best performers)
# ═════════════════════════════════════════════════════════════════════════════

# --- B1: 3+ down days + VIX<25 (V1 winner) ---
header("B1. 3+ DOWN DAYS + VIX < 25 — V1 Winner (Baseline)")

trades = []
for idx, row in daily.iterrows():
    if row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25:
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "B1: 3+down+VIX<25 (baseline)")
print_result(r)
if r: all_results.append(r)

# --- B2: Mon+Wed+VIX<20 (V1 runner-up) ---
header("B2. MON+WED + VIX<20 — V1 Runner-Up (Baseline)")

trades = []
for idx, row in daily.iterrows():
    if row['dow'] in [0, 2] and pd.notna(row['vix']) and row['vix'] < 20:
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "B2: Mon+Wed+VIX<20 (baseline)")
print_result(r)
if r: all_results.append(r)

# --- B3: VIX<15 long (V1 third) ---
header("B3. VIX<15 LONG — V1 Third Place (Baseline)")

trades = []
for idx, row in daily.iterrows():
    if pd.notna(row['vix']) and row['vix'] < 15:
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "B3: VIX<15 long (baseline)")
print_result(r)
if r: all_results.append(r)

# --- B4: Combo score>=3 ---
header("B4. COMBO SCORE ≥ 3 (Baseline)")

trades = []
for idx, row in daily.iterrows():
    score = 0
    if row['prev_streak'] <= -3: score += 2
    if row['dow'] in [0, 2]: score += 1
    if pd.notna(row['vix']) and row['vix'] < 20: score += 1
    if row['prev_streak'] <= -1: score += 1
    if score >= 3:
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "B4: Combo score>=3 (baseline)")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E1: SMA200 TREND FILTER
# Only go long when QQQ is above 200-day SMA (strong uptrend)
# ═════════════════════════════════════════════════════════════════════════════
header("E1a. 3+DOWN + VIX<25 + ABOVE SMA200")
print("  Adds: Only trade when QQQ > 200-day SMA (uptrend confirmation)\n")

trades = []
for idx, row in daily.iterrows():
    if (row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25
            and pd.notna(row['sma200']) and row['Close'] > row['sma200']):
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "E1a: 3+down+VIX<25+SMA200")
print_result(r)
if r: all_results.append(r)

header("E1b. MON+WED + VIX<20 + ABOVE SMA200")

trades = []
for idx, row in daily.iterrows():
    if (row['dow'] in [0, 2] and pd.notna(row['vix']) and row['vix'] < 20
            and pd.notna(row['sma200']) and row['Close'] > row['sma200']):
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "E1b: Mon+Wed+VIX<20+SMA200")
print_result(r)
if r: all_results.append(r)

header("E1c. VIX<15 + ABOVE SMA200")

trades = []
for idx, row in daily.iterrows():
    if (pd.notna(row['vix']) and row['vix'] < 15
            and pd.notna(row['sma200']) and row['Close'] > row['sma200']):
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "E1c: VIX<15+SMA200")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E2: RSI OVERSOLD CONFIRMATION
# Only buy when RSI < 35 (true oversold, not just down days)
# ═════════════════════════════════════════════════════════════════════════════
header("E2a. 3+DOWN + VIX<25 + RSI<35")
print("  Adds: RSI must be oversold (< 35) for stronger bounce signal\n")

trades = []
for idx, row in daily.iterrows():
    if (row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25
            and pd.notna(row['rsi']) and row['rsi'] < 35):
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "E2a: 3+down+VIX<25+RSI<35")
print_result(r)
if r: all_results.append(r)

header("E2b. 3+DOWN + RSI<40 (relaxed)")

trades = []
for idx, row in daily.iterrows():
    if (row['prev_streak'] <= -3 and pd.notna(row['rsi']) and row['rsi'] < 40):
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "E2b: 3+down+RSI<40")
print_result(r)
if r: all_results.append(r)

header("E2c. RSI<30 STANDALONE (pure oversold bounce)")
print("  Buy any time RSI drops below 30\n")

trades = []
for idx, row in daily.iterrows():
    if pd.notna(row['rsi']) and row['rsi'] < 30:
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "E2c: RSI<30 bounce")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E3: ATR-BASED DYNAMIC SL/TP
# Instead of open→close, use intraday ATR stops
# ═════════════════════════════════════════════════════════════════════════════
header("E3. 3+DOWN + VIX<25 + ATR SL/TP")
print("  SL = 1.5 × ATR below entry | TP = 2.5 × ATR above entry")
print("  Simulated using daily bars: hit TP if High reaches, SL if Low reaches\n")

trades = []
for i in range(len(daily)):
    row = daily.iloc[i]
    if not (row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25):
        continue
    if pd.isna(row['atr']):
        continue
    
    entry_price = row['Open']
    sl_price = entry_price - 1.5 * row['atr']
    tp_price = entry_price + 2.5 * row['atr']
    
    # Check same day
    if row['Low'] <= sl_price:
        exit_price = sl_price
    elif row['High'] >= tp_price:
        exit_price = tp_price
    else:
        exit_price = row['Close']
    
    trades.append({'date': daily.index[i], 'entry_price': entry_price, 'exit_price': exit_price, 'direction': 'long'})

r = backtest_trades(pd.DataFrame(trades), "E3: 3+down+VIX<25+ATR SL/TP")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E4: MULTI-DAY HOLDS (2, 3, 5 days)
# Some bounces take longer to develop
# ═════════════════════════════════════════════════════════════════════════════
for hold_days in [2, 3, 5]:
    header(f"E4-{hold_days}d. 3+DOWN + VIX<25 + HOLD {hold_days} DAYS")
    
    trades = []
    for i in range(len(daily)):
        row = daily.iloc[i]
        if not (row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25):
            continue
        exit_idx = min(i + hold_days, len(daily) - 1)
        exit_row = daily.iloc[exit_idx]
        trades.append({'date': daily.index[i], 'entry_price': row['Open'], 'exit_price': exit_row['Close'], 'direction': 'long'})
    
    r = backtest_trades(pd.DataFrame(trades), f"E4: 3+down+VIX<25 hold {hold_days}d")
    print_result(r)
    if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E5: KELLY CRITERION POSITION SIZING
# Size = edge / odds, capped at 2× base size
# ═════════════════════════════════════════════════════════════════════════════
header("E5. 3+DOWN + VIX<25 + KELLY SIZING")
print("  Uses rolling Kelly estimate to scale position from 2% to 10%\n")

# First calculate rolling edge for the mean reversion signal
# Use expanding window of past signals
trades = []
past_wins = 0
past_losses = 0
past_win_sum = 0
past_loss_sum = 0

for idx, row in daily.iterrows():
    if not (row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25):
        continue
    
    # Kelly sizing based on historical edge
    if past_wins + past_losses >= 10:
        wr = past_wins / (past_wins + past_losses)
        avg_w = past_win_sum / max(past_wins, 1)
        avg_l = past_loss_sum / max(past_losses, 1)
        if avg_l > 0:
            kelly = wr - (1 - wr) / (avg_w / avg_l)
            kelly = max(0.02, min(kelly, 0.15))  # 2% to 15% of equity
        else:
            kelly = 0.05
    else:
        kelly = 0.05  # Default 5%
    
    size_pct = kelly * 100
    
    pnl = (row['Close'] - row['Open']) / row['Open'] * 100
    if pnl > 0:
        past_wins += 1
        past_win_sum += pnl
    else:
        past_losses += 1
        past_loss_sum += abs(pnl)
    
    trades.append({
        'date': idx,
        'entry_price': row['Open'],
        'exit_price': row['Close'],
        'direction': 'long',
        'size_pct': size_pct
    })

r = backtest_trades(pd.DataFrame(trades), "E5: 3+down+VIX<25+Kelly")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E6: DEPTH FILTER — Bigger drops = bigger positions
# ═════════════════════════════════════════════════════════════════════════════
header("E6. DEPTH-WEIGHTED MEAN REVERSION")
print("  3+ down days, position size scales with cumulative drop depth")
print("  Drop >= 5%: 10% size | Drop 3-5%: 7% | Drop < 3%: 4%\n")

trades = []
for idx, row in daily.iterrows():
    if not (row['prev_streak'] <= -3):
        continue
    
    drop = abs(row['prev_streak_drop'])
    if drop >= 5:
        size = 10.0
    elif drop >= 3:
        size = 7.0
    else:
        size = 4.0
    
    trades.append({
        'date': idx,
        'entry_price': row['Open'],
        'exit_price': row['Close'],
        'direction': 'long',
        'size_pct': size
    })

r = backtest_trades(pd.DataFrame(trades), "E6: Depth-weighted reversion")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E6b: DEPTH-WEIGHTED + VIX<25
# ═════════════════════════════════════════════════════════════════════════════
header("E6b. DEPTH-WEIGHTED + VIX<25")

trades = []
for idx, row in daily.iterrows():
    if not (row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25):
        continue
    
    drop = abs(row['prev_streak_drop'])
    if drop >= 5:
        size = 10.0
    elif drop >= 3:
        size = 7.0
    else:
        size = 4.0
    
    trades.append({
        'date': idx,
        'entry_price': row['Open'],
        'exit_price': row['Close'],
        'direction': 'long',
        'size_pct': size
    })

r = backtest_trades(pd.DataFrame(trades), "E6b: Depth+VIX<25")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E7: SEASONALITY FILTER
# "Sell in May" — best 6 months: Nov to April
# ═════════════════════════════════════════════════════════════════════════════
header("E7a. MON+WED + VIX<20 + BEST MONTHS (Nov-Apr)")
print("  Adds: Seasonal filter — only trade during Nov–Apr\n")

trades = []
for idx, row in daily.iterrows():
    if (row['dow'] in [0, 2] and pd.notna(row['vix']) and row['vix'] < 20
            and (row['month'] >= 11 or row['month'] <= 4)):
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "E7a: Mon+Wed+VIX<20+NovApr")
print_result(r)
if r: all_results.append(r)

header("E7b. VIX<15 + BEST MONTHS (Nov-Apr)")

trades = []
for idx, row in daily.iterrows():
    if (pd.notna(row['vix']) and row['vix'] < 15
            and (row['month'] >= 11 or row['month'] <= 4)):
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "E7b: VIX<15+NovApr")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E8: COMBINED PORTFOLIO — Stack multiple strategies
# Run ALL winning strategies simultaneously for smoother equity curve
# ═════════════════════════════════════════════════════════════════════════════
header("E8. COMBINED PORTFOLIO — All Top Edges Stacked")
print("  Runs B1 + B2 + B3 as a portfolio with 3% each for diversification")
print("  Each strategy fires independently, multiple can trigger same day\n")

portfolio_trades = []

for idx, row in daily.iterrows():
    # B1: 3+ down + VIX<25
    if row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25:
        portfolio_trades.append({
            'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'],
            'direction': 'long', 'size_pct': 3.0
        })
    
    # B2: Mon+Wed + VIX<20
    if row['dow'] in [0, 2] and pd.notna(row['vix']) and row['vix'] < 20:
        portfolio_trades.append({
            'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'],
            'direction': 'long', 'size_pct': 3.0
        })
    
    # B3: VIX<15
    if pd.notna(row['vix']) and row['vix'] < 15:
        portfolio_trades.append({
            'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'],
            'direction': 'long', 'size_pct': 2.0
        })

portfolio_trades = sorted(portfolio_trades, key=lambda x: x['date'])
r = backtest_trades(pd.DataFrame(portfolio_trades), "E8: Portfolio (B1+B2+B3)")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E9: HIGH-CONVICTION SCALING
# When MULTIPLE signals agree, size up
# ═════════════════════════════════════════════════════════════════════════════
header("E9. HIGH-CONVICTION SCALING — Size Up on Signal Agreement")
print("  Score-based sizing: more edges = bigger position (3-12% equity)")
print("  +2 for 3+down, +1 Mon/Wed, +1 VIX<20, +1 RSI<40, +1 above SMA200")
print("  Size = score * 2%, capped at 12%\n")

trades = []
for idx, row in daily.iterrows():
    score = 0
    if row['prev_streak'] <= -3: score += 2
    if row['dow'] in [0, 2]: score += 1
    if pd.notna(row['vix']) and row['vix'] < 20: score += 1
    if pd.notna(row['rsi']) and row['rsi'] < 40: score += 1
    if pd.notna(row['sma200']) and row['Close'] > row['sma200']: score += 1
    if row['prev_streak'] <= -1: score += 1
    
    if score >= 3:
        size = min(score * 2.0, 12.0)
        trades.append({
            'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'],
            'direction': 'long', 'size_pct': size
        })

r = backtest_trades(pd.DataFrame(trades), "E9: Conviction scaling")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E10: MULTI-DAY HOLD WITH TRAILING STOP
# Hold for up to 5 days, but exit if drops back below entry
# ═════════════════════════════════════════════════════════════════════════════
header("E10. 3+DOWN + VIX<25 + TRAILING STOP (up to 5 days)")
print("  Hold up to 5 days. Trail stop at entry price (lock in breakeven).")
print("  Also trail at 50% of max profit.\n")

trades = []
for i in range(len(daily)):
    row = daily.iloc[i]
    if not (row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25):
        continue
    
    entry_price = row['Open']
    max_price = entry_price
    trail_stop = entry_price * 0.99  # Initial stop 1% below entry
    exit_price = None
    
    for d in range(5):
        idx = i + d
        if idx >= len(daily):
            break
        day_row = daily.iloc[idx]
        
        # Update max and trail
        if day_row['High'] > max_price:
            max_price = day_row['High']
            profit = max_price - entry_price
            trail_stop = max(trail_stop, entry_price + 0.5 * profit)  # Trail at 50% of max profit
        
        # Check if stopped out
        if day_row['Low'] <= trail_stop and d > 0:  # Don't stop on entry day
            exit_price = trail_stop
            break
    
    if exit_price is None:
        end_idx = min(i + 4, len(daily) - 1)
        exit_price = daily.iloc[end_idx]['Close']
    
    trades.append({'date': daily.index[i], 'entry_price': entry_price, 'exit_price': exit_price, 'direction': 'long'})

r = backtest_trades(pd.DataFrame(trades), "E10: 3+down+VIX<25+trail 5d")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E11: VIX CONTANGO FILTER
# When VIX is in contango (VIX < VIX3M), market is calm — bullish
# ═════════════════════════════════════════════════════════════════════════════
if has_vix3m:
    header("E11a. 3+DOWN + VIX CONTANGO")
    print("  VIX contango (VIX < VIX3M) = market calm, mean reversion works better\n")
    
    trades = []
    for idx, row in daily.iterrows():
        if (row['prev_streak'] <= -3 and pd.notna(row['vix_contango'])
                and row['vix_contango'] > 0):  # Contango
            trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
    r = backtest_trades(pd.DataFrame(trades), "E11a: 3+down+VIX contango")
    print_result(r)
    if r: all_results.append(r)
    
    header("E11b. MON+WED + VIX CONTANGO")
    
    trades = []
    for idx, row in daily.iterrows():
        if (row['dow'] in [0, 2] and pd.notna(row['vix_contango'])
                and row['vix_contango'] > 0):
            trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
    r = backtest_trades(pd.DataFrame(trades), "E11b: Mon+Wed+contango")
    print_result(r)
    if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E12: DRAWDOWN-BASED RISK THROTTLE
# Reduce position size during drawdowns
# ═════════════════════════════════════════════════════════════════════════════
header("E12. COMBO + DRAWDOWN THROTTLE")
print("  Base 5% size, reduce to 2% when strategy is in 0.3%+ drawdown")
print("  Protects against regime shifts\n")

trades = []
equity = 25000.0
peak_eq = 25000.0

for idx, row in daily.iterrows():
    score = 0
    if row['prev_streak'] <= -3: score += 2
    if row['dow'] in [0, 2]: score += 1
    if pd.notna(row['vix']) and row['vix'] < 20: score += 1
    if row['prev_streak'] <= -1: score += 1
    
    if score >= 3:
        # Drawdown throttle
        dd_pct = (peak_eq - equity) / peak_eq * 100
        if dd_pct > 0.3:
            size = 2.0  # Throttled
        else:
            size = 5.0  # Full size
        
        pnl_pct = (row['Close'] - row['Open']) / row['Open'] * 100
        equity += equity * size / 100 * pnl_pct / 100
        peak_eq = max(peak_eq, equity)
        
        trades.append({
            'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'],
            'direction': 'long', 'size_pct': size
        })

r = backtest_trades(pd.DataFrame(trades), "E12: Combo+DD throttle")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E13: THE ULTIMATE COMBO
# Best of everything: SMA200 + VIX<25 + RSI<40 + 3+down + Kelly + trail
# ═════════════════════════════════════════════════════════════════════════════
header("E13. ULTIMATE COMBO — All Filters Stacked")
print("  3+down + VIX<25 + SMA200 + RSI<45 + depth sizing")
print("  Maximum conviction trades only\n")

trades = []
for idx, row in daily.iterrows():
    if not (row['prev_streak'] <= -3): continue
    if not (pd.notna(row['vix']) and row['vix'] < 25): continue
    if not (pd.notna(row['sma200']) and row['Close'] > row['sma200']): continue
    if not (pd.notna(row['rsi']) and row['rsi'] < 45): continue
    
    drop = abs(row['prev_streak_drop'])
    if drop >= 5: size = 10.0
    elif drop >= 3: size = 7.0
    else: size = 5.0
    
    trades.append({
        'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'],
        'direction': 'long', 'size_pct': size
    })

r = backtest_trades(pd.DataFrame(trades), "E13: Ultimate combo")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E14: SMA50 CROSSOVER FILTER
# Only trade when SMA50 > SMA200 (golden cross = bull market)
# ═════════════════════════════════════════════════════════════════════════════
header("E14a. 3+DOWN + VIX<25 + GOLDEN CROSS (SMA50>SMA200)")

trades = []
for idx, row in daily.iterrows():
    if (row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25
            and pd.notna(row['sma50']) and pd.notna(row['sma200']) and row['sma50'] > row['sma200']):
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "E14a: 3+down+VIX+GoldenX")
print_result(r)
if r: all_results.append(r)

header("E14b. MON+WED + VIX<20 + GOLDEN CROSS")

trades = []
for idx, row in daily.iterrows():
    if (row['dow'] in [0, 2] and pd.notna(row['vix']) and row['vix'] < 20
            and pd.notna(row['sma50']) and pd.notna(row['sma200']) and row['sma50'] > row['sma200']):
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "E14b: Mon+Wed+VIX+GoldenX")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E15: OVERNIGHT HOLD ON BOUNCE DAYS
# Buy at close of bounce day, sell at next open (capture gap-up)
# ═════════════════════════════════════════════════════════════════════════════
header("E15. 3+DOWN BOUNCE → OVERNIGHT HOLD")
print("  If 3+ down days trigger fires AND bounce day closes UP,")
print("  hold overnight to capture continuation gap-up\n")

trades = []
for i in range(len(daily) - 1):
    row = daily.iloc[i]
    if not (row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25):
        continue
    # Only if the bounce day actually bounced
    if row['Close'] <= row['Open']:
        continue
    # Hold overnight
    next_row = daily.iloc[i + 1]
    trades.append({
        'date': daily.index[i],
        'entry_price': row['Close'],
        'exit_price': next_row['Open'],
        'direction': 'long'
    })

r = backtest_trades(pd.DataFrame(trades), "E15: 3+down bounce→overnight")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E16: INTRADAY MEAN REVERSION + NEXT DAY CONTINUATION
# Buy at open after 3+ down, sell at NEXT day's close (2-day total hold)
# ═════════════════════════════════════════════════════════════════════════════
header("E16. 3+DOWN + VIX<25 + SMA200 + HOLD 2 DAYS")
print("  Combine trend filter with extended hold for larger gains\n")

trades = []
for i in range(len(daily) - 1):
    row = daily.iloc[i]
    if not (row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25
            and pd.notna(row['sma200']) and row['Close'] > row['sma200']):
        continue
    next_row = daily.iloc[i + 1]
    trades.append({'date': daily.index[i], 'entry_price': row['Open'], 'exit_price': next_row['Close'], 'direction': 'long'})

r = backtest_trades(pd.DataFrame(trades), "E16: 3+down+VIX+SMA200 2d")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E17: AGGRESSIVE PORTFOLIO — All edges, sized by conviction
# ═════════════════════════════════════════════════════════════════════════════
header("E17. AGGRESSIVE PORTFOLIO — Every Edge, Conviction-Sized")
print("  Runs ALL profitable signals simultaneously, sizes by conviction:")
print("  3+down+VIX<25: 5% | Mon+Wed+VIX<20: 3% | VIX<15: 2%")
print("  Bonus +2% if SMA200 bullish, +1% if RSI<40\n")

portfolio_trades = []

for idx, row in daily.iterrows():
    # Mean reversion signal
    if row['prev_streak'] <= -3 and pd.notna(row['vix']) and row['vix'] < 25:
        size = 5.0
        if pd.notna(row['sma200']) and row['Close'] > row['sma200']: size += 2.0
        if pd.notna(row['rsi']) and row['rsi'] < 40: size += 1.0
        portfolio_trades.append({
            'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'],
            'direction': 'long', 'size_pct': size
        })
    
    # Day-of-week signal (independent)
    if row['dow'] in [0, 2] and pd.notna(row['vix']) and row['vix'] < 20:
        size = 3.0
        if pd.notna(row['sma200']) and row['Close'] > row['sma200']: size += 1.0
        portfolio_trades.append({
            'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'],
            'direction': 'long', 'size_pct': size
        })
    
    # Calm market signal (independent)
    if pd.notna(row['vix']) and row['vix'] < 15:
        size = 2.0
        portfolio_trades.append({
            'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'],
            'direction': 'long', 'size_pct': size
        })

portfolio_trades = sorted(portfolio_trades, key=lambda x: x['date'])
r = backtest_trades(pd.DataFrame(portfolio_trades), "E17: Aggressive portfolio")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E18: FRIDAY CLOSE → MONDAY CLOSE (WEEKEND HOLD)
# ═════════════════════════════════════════════════════════════════════════════
header("E18. FRIDAY→MONDAY HOLD (Weekend Effect)")
print("  Buy Friday at close, sell Monday at close (capture weekend gap + Monday)\n")

trades = []
for i in range(len(daily) - 3):
    row = daily.iloc[i]
    if row['dow'] != 4:  # Friday
        continue
    # Find next Monday
    for j in range(1, 4):
        if i + j < len(daily) and daily.iloc[i + j]['dow'] == 0:
            monday = daily.iloc[i + j]
            trades.append({
                'date': daily.index[i],
                'entry_price': row['Close'],
                'exit_price': monday['Close'],
                'direction': 'long'
            })
            break

r = backtest_trades(pd.DataFrame(trades), "E18: Fri→Mon hold")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E19: VIX MEAN REVERSION (Buy when VIX spikes)
# ═════════════════════════════════════════════════════════════════════════════
header("E19. VIX SPIKE FADE — Buy QQQ when VIX jumps > 2pts in a day")
print("  When VIX spikes, fear is overdone → buy QQQ next day\n")

daily['vix_change'] = daily['vix'].diff()

trades = []
for idx, row in daily.iterrows():
    if pd.notna(row['vix_change']) and row['vix_change'] > 2.0:
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "E19: VIX spike fade")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCEMENT E20: 4+ DOWN DAYS (MORE EXTREME BOUNCE)
# ═════════════════════════════════════════════════════════════════════════════
header("E20a. 4+ DOWN DAYS (more extreme streak)")

trades = []
for idx, row in daily.iterrows():
    if row['prev_streak'] <= -4:
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "E20a: 4+ down days")
print_result(r)
if r: all_results.append(r)

header("E20b. 4+ DOWN DAYS + VIX<30")

trades = []
for idx, row in daily.iterrows():
    if row['prev_streak'] <= -4 and pd.notna(row['vix']) and row['vix'] < 30:
        trades.append({'date': idx, 'entry_price': row['Open'], 'exit_price': row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "E20b: 4+down+VIX<30")
print_result(r)
if r: all_results.append(r)

header("E20c. 4+ DOWN DAYS + VIX<30 + HOLD 2 DAYS")

trades = []
for i in range(len(daily) - 1):
    row = daily.iloc[i]
    if row['prev_streak'] <= -4 and pd.notna(row['vix']) and row['vix'] < 30:
        next_row = daily.iloc[i + 1]
        trades.append({'date': daily.index[i], 'entry_price': row['Open'], 'exit_price': next_row['Close'], 'direction': 'long'})
r = backtest_trades(pd.DataFrame(trades), "E20c: 4+down+VIX<30 2d")
print_result(r)
if r: all_results.append(r)


# ═════════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════════════════
header("FINAL RESULTS — ALL STRATEGIES vs BUY & HOLD")

print(f"  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
print(f"  │ BUY & HOLD BENCHMARK (QQQ, {bh_years:.1f}yr)                                                                                          │")
print(f"  │   Total Return: {bh_total_return:+.1f}%   CAGR: {bh_cagr:+.1f}%   MaxDD: {bh_max_dd:.1f}%   Sharpe: {bh_sharpe:+.2f}                                              │")
print(f"  │   On $25K with 5% sizing: ${bh_benchmark['adjusted_5pct_pnl']:+,.0f} P&L   DD: {bh_benchmark['adjusted_5pct_dd']:.2f}%                                                     │")
print(f"  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘\n")

# Table header
print(f"  {'Rank':>4}  {'Strategy':<38} {'PF':>5} {'WR%':>5} {'Sharpe':>7} {'Calmar':>7} {'Trades':>6} {'P&L$':>10} {'DD%':>6} {'p-val':>7} {'Sig':>3} {'α vs B&H':>9}")
print(f"  {'─'*4}  {'─'*38} {'─'*5} {'─'*5} {'─'*7} {'─'*7} {'─'*6} {'─'*10} {'─'*6} {'─'*7} {'─'*3} {'─'*9}")

sorted_results = sorted(all_results, key=lambda x: x['profit_factor'] if x['profit_factor'] < 100 else 0, reverse=True)

for i, r in enumerate(sorted_results):
    sig = "***" if r['p_value'] < 0.01 else "** " if r['p_value'] < 0.05 else "*  " if r['p_value'] < 0.10 else "   "
    pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "INF"
    alpha_str = f"{r['alpha_vs_bh']:+.2f}%"
    calmar_str = f"{r['calmar']:.2f}" if r['calmar'] < 100 else "INF"
    print(f"  {i+1:4d}  {r['name']:<38} {pf_str:>5} {r['win_rate']:5.1f} {r['sharpe']:+7.2f} {calmar_str:>7} {r['trades']:6d} {r['total_pnl']:+10,.0f} {r['max_dd_pct']:5.2f}% {r['p_value']:7.4f} {sig} {alpha_str:>9}")

# Beat B&H analysis
print(f"\n\n  {'='*80}")
print(f"  STRATEGIES THAT BEAT BUY & HOLD (risk-adjusted)")
print(f"  {'='*80}")
print(f"")
print(f"  Buy & Hold Sharpe: {bh_sharpe:+.2f} (unadjusted for position size)")
print(f"  Buy & Hold CAGR:   {bh_cagr:+.2f}% (full investment)")
print(f"")

# Compare on Sharpe (most fair comparison)
beat_bh = [r for r in sorted_results if r['sharpe'] > bh_sharpe and r['trades'] > 20]
print(f"  Strategies with Sharpe ABOVE buy-and-hold ({bh_sharpe:+.2f}):")
if beat_bh:
    for r in beat_bh:
        sharpe_edge = r['sharpe'] - bh_sharpe
        sig = "***" if r['p_value'] < 0.01 else "** " if r['p_value'] < 0.05 else "*  " if r['p_value'] < 0.10 else "   "
        print(f"    ✅ {r['name']:<38} Sharpe={r['sharpe']:+.2f} (+{sharpe_edge:.2f}) PF={r['profit_factor']:.2f} {sig}")
else:
    print("    None with 20+ trades")

# Compare on drawdown-adjusted returns (Calmar)
print(f"\n  Strategies with better risk/reward than B&H:")
bh_calmar = bh_cagr / bh_max_dd if bh_max_dd > 0 else 0
print(f"  (Buy & Hold Calmar ≈ {bh_calmar:.3f})")

beat_calmar = [r for r in sorted_results if r['calmar'] > bh_calmar and r['trades'] > 20]
if beat_calmar:
    for r in beat_calmar[:10]:
        print(f"    ✅ {r['name']:<38} Calmar={r['calmar']:.2f} vs B&H {bh_calmar:.3f}")
else:
    print("    Note: B&H Calmar uses full-sized position, our strategies use 5% sizing")

# Summary
print(f"\n\n  {'='*80}")
print(f"  KEY INSIGHTS")
print(f"  {'='*80}")
print(f"")
print(f"  1. BUY & HOLD CONTEXT:")
print(f"     QQQ returned {bh_total_return:+.0f}% over {bh_years:.0f} years ({bh_cagr:+.1f}% CAGR)")
print(f"     BUT suffered a {bh_max_dd:.0f}% max drawdown")
print(f"     Sharpe: {bh_sharpe:.2f} (decent but not great)")
print(f"")
print(f"  2. OUR STRATEGIES vs B&H:")
print(f"     Our strategies use only 5% position sizing = MUCH less capital at risk")
print(f"     Key advantage: RISK-ADJUSTED returns (Sharpe, Calmar)")
print(f"")

# Best strategy
if sorted_results:
    best = sorted_results[0]
    print(f"  3. BEST STRATEGY: {best['name']}")
    print(f"     PF={best['profit_factor']:.2f} Sharpe={best['sharpe']:+.2f} MaxDD={best['max_dd_pct']:.2f}%")
    print(f"     On $25K: ${best['total_pnl']:+,.0f} ({best['total_pnl_pct']:+.1f}%)")
    
    if best['sharpe'] > bh_sharpe:
        print(f"     ✅ BEATS buy-and-hold on Sharpe ({best['sharpe']:+.2f} vs {bh_sharpe:+.2f})")
    else:
        print(f"     ❌ Below buy-and-hold Sharpe ({best['sharpe']:+.2f} vs {bh_sharpe:+.2f})")
    
    if best['max_dd_pct'] < bh_max_dd * 0.05:
        print(f"     ✅ MUCH lower drawdown ({best['max_dd_pct']:.2f}% vs {bh_max_dd*0.05:.2f}% adjusted)")
    
    print(f"")
    print(f"  4. IMPROVEMENT vs V1 BASELINE:")
    baselines = [r for r in sorted_results if 'baseline' in r['name']]
    enhanced = [r for r in sorted_results if 'baseline' not in r['name']]
    if baselines and enhanced:
        best_base = max(baselines, key=lambda x: x['profit_factor'])
        best_enh = max(enhanced, key=lambda x: x['profit_factor'])
        pf_improve = (best_enh['profit_factor'] / best_base['profit_factor'] - 1) * 100
        pnl_improve = best_enh['total_pnl'] - best_base['total_pnl']
        print(f"     Best baseline: {best_base['name']} PF={best_base['profit_factor']:.2f}")
        print(f"     Best enhanced: {best_enh['name']} PF={best_enh['profit_factor']:.2f}")
        print(f"     PF improvement: {pf_improve:+.1f}%")
        print(f"     P&L improvement: ${pnl_improve:+,.0f}")

# Save JSON
results_json = {
    'generated': datetime.now().isoformat(),
    'capital': CAPITAL,
    'buy_and_hold': bh_benchmark,
    'strategies': [{k: v for k, v in r.items() if k != 'equity_curve'} for r in sorted_results]
}
json_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'nasdaq_v2_results.json')
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2, default=str)

print(f"\n  Results saved to:")
print(f"    outputs/nasdaq_v2_results.txt")
print(f"    outputs/nasdaq_v2_results.json")
print(f"\n  Analysis complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
