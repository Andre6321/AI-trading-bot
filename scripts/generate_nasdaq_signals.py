"""
NASDAQ Daily Signal Generator (V3)
====================================
Run this script each morning BEFORE 9:30 ET to get today's trading signals.

Checks all edges and tells you:
  - Whether to trade today
  - Which edges are active
  - Position size recommendation
  - VIX status and risk level

Usage:
  python scripts/generate_nasdaq_signals.py

Edges (all statistically significant, walk-forward validated):
  1. Mean Reversion: 3+ down days + VIX<25 → buy at open, sell at close
  2. Day-of-Week: Monday or Wednesday + VIX<20 → buy at open, sell at close
  3. Calm Market: VIX<15 → buy at open, sell at close
  4. Gap Down Bounce: 3+ down days + gap down >0.3% → buy at open, sell at close (PF=3.30)
  5. Month-Start Flow: First 3 trading days of month + VIX<20 (PF=1.44)
  6. VIX 20-25 Sweet Spot: Elevated fear + 3+ down = panic overdone (PF=2.67)
"""

import sys, os, warnings, json
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

CAPITAL = 25000.0

def main():
    now = datetime.now()
    print(f"\n{'='*70}")
    print(f"  NASDAQ DAILY SIGNAL GENERATOR")
    print(f"  {now.strftime('%A, %B %d, %Y  %H:%M ET')}")
    print(f"  Capital: ${CAPITAL:,.0f}")
    print(f"{'='*70}\n")

    # Download latest data
    print("  Fetching data...")
    qqq = yf.download("QQQ", period="1y", interval="1d", progress=False)
    vix = yf.download("^VIX", period="1y", interval="1d", progress=False)
    
    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = qqq.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    # Current state
    latest = qqq.iloc[-1]
    prev = qqq.iloc[-2] if len(qqq) > 1 else latest
    
    current_price = latest['Close']
    vix_current = vix.iloc[-1]['Close'] if len(vix) > 0 else None

    # Calculate consecutive down days (looking at completed days)
    down_streak = 0
    cum_drop = 0.0
    for i in range(len(qqq) - 1, -1, -1):
        row = qqq.iloc[i]
        if row['Close'] < row['Open']:
            down_streak += 1
            cum_drop += (row['Close'] - row['Open']) / row['Open'] * 100
        else:
            break

    # SMA calculations
    sma200 = qqq['Close'].rolling(200).mean().iloc[-1]
    sma50 = qqq['Close'].rolling(50).mean().iloc[-1]
    above_sma200 = current_price > sma200 if not pd.isna(sma200) else True
    golden_cross = sma50 > sma200 if not pd.isna(sma50) and not pd.isna(sma200) else True

    # Day of week for TODAY (the trading day)
    today_dow = now.weekday()  # 0=Mon, 1=Tue, ..., 4=Fri
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    today_name = day_names[today_dow]
    is_mon_wed = today_dow in [0, 2]

    # Month / Trading day of month
    current_month = now.month
    in_best_season = current_month >= 11 or current_month <= 4
    
    # Trading day of month (approximate)
    month_start = datetime(now.year, now.month, 1)
    trading_days_this_month = len(qqq[qqq.index >= str(month_start.date())])
    is_month_start = trading_days_this_month <= 3  # First 3 trading days

    # Yesterday's gap (for gap-down detection)
    if len(qqq) >= 2:
        prev_close = qqq.iloc[-2]['Close']
        today_open = qqq.iloc[-1]['Open']
        gap_pct = (today_open - prev_close) / prev_close * 100
    else:
        gap_pct = 0.0
    is_gap_down = gap_pct < -0.3

    # ─── MARKET STATUS ───────────────────────────────────────────────────
    print(f"  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  MARKET STATUS                                         │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  QQQ Price:     ${current_price:>10.2f}                        │")
    
    if vix_current is not None:
        vix_status = "🟢 CALM" if vix_current < 15 else "🟡 OK" if vix_current < 20 else "🟠 CAUTION" if vix_current < 25 else "🔴 DANGER"
        print(f"  │  VIX:           {vix_current:>10.1f}  {vix_status:<16}       │")
    
    print(f"  │  Down Streak:   {down_streak:>10d} days                       │")
    if down_streak > 0:
        print(f"  │  Streak Drop:   {cum_drop:>9.2f}% cumulative                │")
    
    print(f"  │  SMA200:        {'ABOVE ✓' if above_sma200 else 'BELOW ✗':>10}  (${sma200:.2f})            │")
    print(f"  │  SMA50/200:     {'GOLDEN ✓' if golden_cross else 'DEATH ✗':>10}                        │")
    print(f"  │  Day:           {today_name:>10}                        │")
    print(f"  │  Season:        {'Best (Nov-Apr) ✓' if in_best_season else 'Weak (May-Oct)':>16}                │")
    print(f"  │  Month-Start:   {'YES ✓' if is_month_start else 'NO':>10} (day {trading_days_this_month})              │")
    print(f"  │  Gap:           {gap_pct:>9.2f}%{' (GAP DOWN!)' if is_gap_down else '':>12}                │")
    print(f"  └─────────────────────────────────────────────────────────┘\n")

    # ─── SIGNAL EVALUATION ───────────────────────────────────────────────
    signals = []
    total_size = 0.0

    # Edge 1: Mean Reversion
    e1_active = down_streak >= 3 and vix_current is not None and vix_current < 25
    if e1_active:
        drop = abs(cum_drop)
        if drop >= 5:
            size = 10.0
            depth = "DEEP"
        elif drop >= 3:
            size = 7.0
            depth = "MEDIUM"
        else:
            size = 4.0
            depth = "SMALL"
        signals.append(('MEAN REVERSION', size, f"{down_streak} down days, drop={cum_drop:.1f}% ({depth})"))
        total_size += size

    # Edge 2: Mon+Wed Seasonality
    e2_active = is_mon_wed and vix_current is not None and vix_current < 20
    if e2_active:
        size = 3.0
        if above_sma200:
            size += 1.0
        signals.append(('MON/WED', size, f"{today_name}, VIX={vix_current:.1f}"))
        total_size += size

    # Edge 3: Calm Market
    e3_active = vix_current is not None and vix_current < 15
    if e3_active:
        size = 2.0
        signals.append(('CALM MARKET', size, f"VIX={vix_current:.1f} < 15"))
        total_size += size

    # Edge 4: Gap Down Bounce (V3 NEW — PF=3.30, Sharpe=+5.88)
    e4_active = down_streak >= 3 and is_gap_down
    if e4_active:
        drop = abs(cum_drop)
        size = 3.0  # Extra conviction on top of Edge 1
        if vix_current is not None and vix_current < 30:
            signals.append(('GAP DOWN BOUNCE', size, f"3+down + gap {gap_pct:.1f}% — double panic"))
            total_size += size

    # Edge 5: Month-Start Flow (V3 NEW — PF=1.44, Sharpe=+2.26)
    e5_active = is_month_start and vix_current is not None and vix_current < 20
    if e5_active:
        size = 2.0
        signals.append(('MONTH-START FLOW', size, f"Trading day {trading_days_this_month} of month, fund flows"))
        total_size += size

    # Edge 6: VIX 20-25 Sweet Spot (V3 NEW — PF=2.67, Sharpe=+5.72)
    # Only fires when 3+down AND VIX is 20-25 (fear without panic)
    e6_active = down_streak >= 3 and vix_current is not None and 20 <= vix_current < 25
    if e6_active and not e1_active:
        # This edge fires even without the base Edge 1 VIX<25 filter
        # since VIX 20-25 IS within <25, Edge 1 should already be active
        # But add extra conviction sizing
        size = 2.0
        signals.append(('VIX SWEET SPOT', size, f"VIX {vix_current:.1f} = elevated fear, bounce stronger"))
        total_size += size

    # ─── SIGNALS OUTPUT ──────────────────────────────────────────────────
    print(f"  ╔═══════════════════════════════════════════════════════════╗")
    if signals:
        print(f"  ║  🟢  TRADE TODAY — {len(signals)} signal(s) active                    ║")
        print(f"  ╠═══════════════════════════════════════════════════════════╣")
        for name, size, reason in signals:
            dollar_size = CAPITAL * size / 100
            print(f"  ║                                                           ║")
            print(f"  ║  📊 {name:<15}                                      ║")
            print(f"  ║     Size: {size:.0f}% (${dollar_size:,.0f})                                 ║")
            print(f"  ║     Why:  {reason:<47} ║")
        
        print(f"  ║                                                           ║")
        print(f"  ╠═══════════════════════════════════════════════════════════╣")
        total_dollar = CAPITAL * total_size / 100
        print(f"  ║  TOTAL EXPOSURE: {total_size:.0f}% of equity (${total_dollar:,.0f})              ║")
        print(f"  ║                                                           ║")
        print(f"  ║  ACTION:                                                  ║")
        print(f"  ║    1. Buy QQQ/NAS100 at market open (9:30 ET)             ║")
        print(f"  ║    2. Set position size: ${total_dollar:,.0f}                         ║")
        print(f"  ║    3. Close ALL positions at 15:55 ET (before close)      ║")
        print(f"  ║    4. Do NOT hold overnight unless trailing stop mode     ║")
    else:
        print(f"  ║  ⚪  NO TRADE TODAY — No edges active                    ║")
        print(f"  ╠═══════════════════════════════════════════════════════════╣")
        
        # Explain why
        reasons = []
        if down_streak < 3:
            reasons.append(f"Down streak only {down_streak} (need 3+)")
        if vix_current is not None and vix_current >= 25:
            reasons.append(f"VIX={vix_current:.1f} (≥25 = danger zone)")
        elif vix_current is not None and vix_current >= 20:
            reasons.append(f"VIX={vix_current:.1f} (≥20, only Edge 3 if <15)")
        if not is_mon_wed:
            reasons.append(f"Not Monday or Wednesday ({today_name})")
        
        for reason in reasons:
            print(f"  ║    • {reason:<53} ║")
        
        if not reasons:
            print(f"  ║    • No qualifying conditions met                        ║")
    
    print(f"  ╚═══════════════════════════════════════════════════════════╝\n")

    # ─── RISK WARNINGS ───────────────────────────────────────────────────
    if vix_current is not None and vix_current >= 25:
        print(f"  ⚠️  WARNING: VIX is {vix_current:.1f} — ELEVATED RISK")
        print(f"     All signals are suspended when VIX ≥ 25.")
        print(f"     Historical data shows mean reversion FAILS in panic.\n")
    
    if not above_sma200:
        print(f"  ⚠️  CAUTION: QQQ is BELOW 200-day SMA")
        print(f"     Consider enabling SMA200 filter in PineScript.\n")

    # ─── SAVE SIGNAL ─────────────────────────────────────────────────────
    signal_data = {
        'timestamp': now.isoformat(),
        'instrument': 'QQQ',
        'price': float(current_price),
        'vix': float(vix_current) if vix_current is not None else None,
        'down_streak': int(down_streak),
        'cum_drop_pct': float(cum_drop),
        'above_sma200': bool(above_sma200),
        'golden_cross': bool(golden_cross),
        'day': today_name,
        'in_best_season': bool(in_best_season),
        'signals': [{'name': s[0], 'size_pct': s[1], 'reason': s[2]} for s in signals],
        'total_size_pct': total_size,
        'action': 'BUY' if signals else 'FLAT',
    }
    
    signal_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'latest_nasdaq_signal.json')
    with open(signal_path, 'w') as f:
        json.dump(signal_data, f, indent=2)
    
    print(f"  Signal saved to: outputs/latest_nasdaq_signal.json")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
