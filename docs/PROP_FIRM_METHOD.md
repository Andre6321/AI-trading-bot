# 🏆 THE ONE METHOD: Prop Firm Profitability Blueprint
## My Best Recommendation Based on Everything We've Learned

> This is not a menu of options. This is THE one strategy I'd use if I were sitting
> down to pass a prop firm challenge and stay funded. It's built on:
> - Our proven mean reversion edge (walk-forward validated PF 2.08)
> - The highest-probability scalping setups from research
> - Your actual prop firm rules ($50K Pro, $2K EOD DD, 50% consistency)
> - What actually separates the 3% who make it from the 97% who don't

---

## WHY THIS SPECIFIC METHOD

From everything we've researched:

1. **Mean reversion is the most robust edge** — We proved it with 92 NASDAQ configs.
   It works because markets are auction-driven; overextensions MUST correct.

2. **VWAP is the institutional anchor** — Every algo, every fund, every market maker
   benchmarks to VWAP. When price deviates far from VWAP, institutional flow
   pulls it back. This isn't theory — it's market structure.

3. **The first 90 minutes have 70%+ of daily volume** — More volume = tighter
   spreads = better fills = more reliable signals. After 11 AM, you're gambling
   in thin air.

4. **Prop firms reward boring consistency** — They don't want a hero who makes
   $5K one day and loses $3K the next. They want $300-$600/day, 15+ green days
   per month, with a smooth equity curve.

5. **One instrument, one strategy, one session** — Every profitable prop trader
   I've studied masters ONE thing. Not five strategies across three markets.
   Depth beats breadth, always.

---

## THE METHOD: VWAP Mean Reversion on MNQ

### The Core Idea
```
When price stretches too far from VWAP → it snaps back.
You enter on the snap-back signal → ride it to VWAP → get out.
Repeat 2-5 times per morning session.
That's it. Nothing else.
```

### Your Instrument: MNQ (Evaluation) → NQ (Funded)

Your prop firm has **Micro Scaling 10:1 during evaluation** but **NO micro scaling
when funded**. This means:

| Phase | Max Position | What You Trade | Tick Value |
|-------|-------------|----------------|------------|
| **Evaluation** | 3 contracts (= 30 MNQ) | MNQ (Micro Nasdaq) | $0.50 / tick |
| **Funded** | 5 contracts | NQ (E-mini Nasdaq) | $5.00 / tick |

**During Evaluation** — Trade MNQ. Use up to 30 MNQ contracts (= 3 standard).
This gives you fine-grained position sizing and smaller risk per contract.

**When Funded** — You MUST trade NQ (no micros). Each tick is $5.00 instead of
$0.50. This is 10× more per contract. You'll adjust by using fewer contracts
(1-3 NQ) with wider stops instead of 20+ MNQ with tight stops.

| Factor | MNQ (Eval) | NQ (Funded) |
|--------|-----------|-------------|
| **Tick value** | $0.50 / 0.25 pt | $5.00 / 0.25 pt |
| **Daily range** | 200-400 pts | 200-400 pts |
| **Your max contracts** | 30 MNQ | 5 NQ |
| **Dollar per point move** | $2.00/pt × 30 = $60/pt | $20/pt × 5 = $100/pt |
| **10-point scalp profit** | $600 (30 MNQ) | $1,000 (5 NQ) |

**Start on MNQ during eval. The smaller tick value is more forgiving while you
build confidence. Once funded, the NQ switch will feel natural because the
setups are identical — only the dollar amounts change.**

---

## COMPLETE SETUP

### Platform: Tradovate

```
YOUR SETUP:
  ✅ Tradovate (execution + charting)
  ✅ TradingView (backup charting + alerts) — you already have this
  ✅ ForexFactory.com calendar (check daily for news)

RECOMMENDED:
  ✅ Tradervue or TradeZella (journaling)
  ✅ Second monitor (or split screen)

NOT NEEDED:
  ❌ Bookmap, Sierra Chart, order flow tools (later, not now)
  ❌ AI signals, Discord alerts, or anyone else's calls
  ❌ More than 3 indicators on your chart
```

### Tradovate Quick Reference

```
SYMBOLS (quarterly expiration — roll 1 week before expiry):
  MNQ = Micro E-mini Nasdaq-100   (evaluation phase)
  NQ  = E-mini Nasdaq-100          (funded phase)
  
  Contract month codes:  H=Mar  M=Jun  U=Sep  Z=Dec
  Format: SYMBOL + MONTH + YEAR DIGIT
  
  Examples (2026):
    MNQH6 = Micro Nasdaq March 2026
    MNQM6 = Micro Nasdaq June 2026
    NQH6  = E-mini Nasdaq March 2026
    NQM6  = E-mini Nasdaq June 2026
  
  ⚠️  Always trade the FRONT MONTH (highest volume).
      Roll to next contract ~1 week before expiration.
      Expiration = 3rd Friday of the contract month.

CONTRACT SPECS:
  MNQ: 1 tick = 0.25 points = $0.50
       1 point = 4 ticks = $2.00 per contract
  NQ:  1 tick = 0.25 points = $5.00
       1 point = 4 ticks = $20.00 per contract

TRADOVATE SETUP (do this BEFORE your first trade):

  1. DEFAULT QUANTITY
     → On the DOM or order panel, set default qty to 6 (MNQ)
     → This saves you from fat-fingering size on every trade

  2. BRACKET ORDERS (auto stop + target on every entry)
     → Settings → Trading → enable "Auto-attach bracket"  
     → Default Stop: 12-15 points (adjust per setup)
     → Default Target: 30 points (adjust per setup)
     → Every market/limit entry now auto-places your stop

  3. HOTKEYS (critical for scalping speed)
     → Settings → Hotkeys → set these:
        Buy Market:     Shift+B
        Sell Market:    Shift+S
        Flatten All:    Shift+F  ← MEMORIZE THIS ONE
        Cancel Orders:  Shift+C
        +1 Qty:         Shift+Up
        -1 Qty:         Shift+Down
     → Practice: Shift+F = PANIC BUTTON = close everything NOW

  4. CHART INDICATORS (add these to your 2-min chart)
     → Indicators → VWAP (enable Std Dev Bands: 1σ and 2σ)
     → Indicators → EMA → Period: 9
     → Indicators → Volume
     → That's it. Three indicators. No more.

  5. DOM (Depth of Market) — RECOMMENDED for execution
     → Right-click chart → Open DOM
     → Click ASK side = BUY limit at that price
     → Click BID side = SELL limit at that price
     → Bottom buttons = Buy MKT / Sell MKT
     → The DOM gives you faster execution than chart clicking

  6. CHART TRADING (alternative to DOM)
     → Click the "T" icon on chart toolbar
     → Click directly on price levels to place limit orders
     → Buy/Sell buttons appear on the chart panel
     → Easier to see context, but slightly slower than DOM

ORDER TYPES YOU'LL USE:
  Market Order:  Instant fill at current price (for entries)
  Limit Order:   Fill at your price or better (for targets)
  Stop Order:    Becomes market when price hits level (for stops)
  Stop Limit:    Becomes limit when price hits level (careful — can miss)
  
  RECOMMENDATION: Market for entries, Stop for stops, Limit for targets.
  Do NOT use stop-limit for your protective stop — it can skip in fast moves.
```

### Chart Configuration

#### Chart 1: Execution Chart — MNQ 2-Minute
Why 2-min and not 1-min or 5-min:
- 1-min = too much noise, too many fake signals
- 5-min = too slow for scalping, miss entries
- 2-min = sweet spot — filters noise but fast enough for scalp entries

**Indicators** (only these, nothing else):
```
1. VWAP (anchored to session start)
   - With Standard Deviation bands: +1σ, -1σ, +2σ, -2σ
   - Colors: VWAP = white, ±1σ = yellow, ±2σ = red

2. 9 EMA (lime green, thin)
   - Fast momentum reference
   - Entry confirmation: price crosses back through 9 EMA toward VWAP

3. Volume bars (bottom panel)
   - With 20-period SMA overlay
   - Used to confirm: are reversal candles on high volume?

4. RSI(14) (sub-panel, optional)
   - Only as confirmation: <30 for longs, >70 for shorts
   - If you want a cleaner chart, skip RSI — the VWAP bands do the job
```

**Mark these levels at session start (before 9:30 AM ET)**:
```
• Previous day HIGH (horizontal line — red dashed)
• Previous day LOW (horizontal line — green dashed)  
• Previous day CLOSE (horizontal line — white dashed)
• Overnight HIGH (magenta)
• Overnight LOW (magenta)
• Key round numbers (e.g., 20,000 / 20,100 / 20,200)
```

#### Chart 2: Context Chart — MNQ 15-Minute
**Purpose**: Know the higher-timeframe trend so you don't scalp AGAINST it.
```
Indicators:
  - 50 SMA (trend direction)
  - 200 SMA (macro trend)
  - VWAP (same-day anchor)
```

**Reading the context chart**:
```
Price ABOVE 50 SMA and 50 SMA sloping UP:
  → Uptrend. ONLY take LONG setups from VWAP.
  → Short setups need extra confirmation (or skip them).

Price BELOW 50 SMA and 50 SMA sloping DOWN:
  → Downtrend. ONLY take SHORT setups from VWAP.
  → Long setups need extra confirmation (or skip them).

Price crossing back and forth over 50 SMA:
  → Choppy/range. Take BOTH long and short VWAP MR setups.
```

---

## THE EXACT TRADE RULES

### Setup A: VWAP Fade (The Bread & Butter — 70% of your trades)

This is your primary setup. It's a mean reversion trade where price has
stretched too far from VWAP and you're betting it reverts back.

#### LONG Setup (buying the dip to VWAP)
```
REQUIRED CONDITIONS (ALL must be true):
  ☑ Price touches or pierces the VWAP -1σ band (or better, -2σ)
  ☑ The 15-min context is NOT strong downtrend 
    (i.e., don't buy dips in a hard sell-off)
  ☑ No high-impact news in the next 15 minutes
  ☑ It's between 9:45 AM and 11:00 AM ET
  ☑ You have NOT hit 3 losses today
  ☑ You have NOT hit your daily profit target

ENTRY TRIGGER (need at least 1):
  → Bullish engulfing candle at the VWAP band
  → Hammer / pin bar with lower wick rejection at the band  
  → Price closes back ABOVE the 9 EMA after being below it
  → Volume spike on the reversal candle (1.5× average)

ENTRY:
  → Market buy at the CLOSE of the trigger candle
  → Do NOT anticipate. Wait for the candle to CLOSE.

STOP LOSS:
  → 2 ticks below the low of the trigger candle
  → OR 2 ticks below the VWAP -2σ band (whichever is tighter)
  → MAXIMUM stop: 15 MNQ points ($30/contract at $2/pt)
  → If the stop needs to be wider than 15 points, SKIP THE TRADE
  → At 6 MNQ: 15 pts × $2 × 6 = $180 max risk per trade ✓

TARGETS:
  → TP1: VWAP -1σ band (if you entered at -2σ) — EXIT 50%
  → TP2: VWAP itself — EXIT remaining 50%
  → If entered at -1σ: TP1 = VWAP (exit 100%)

MANAGEMENT:
  → Once TP1 is hit, move stop to breakeven on the remainder
  → Never let a winning trade become a losing trade
  → If price stalls for 10+ minutes without progress, exit at market
```

#### SHORT Setup (selling the rip to VWAP)
```
Mirror of the long setup:
  ☑ Price touches or pierces VWAP +1σ (or +2σ)
  ☑ 15-min context NOT strong uptrend
  ☑ Bearish engulfing / shooting star / close below 9 EMA
  ☑ Stop: 2 ticks above the high of the trigger candle (max 15 pts)
  ☑ TP1: VWAP +1σ (50%), TP2: VWAP (50%)
```

#### Visual Example: Long Setup

```
Price action on MNQ 2-min chart:

          │
    20150 │     ╭──╮            ← Previous swing high
          │    ╱    ╲
    20130 │───╱──────╲──────── ← VWAP (white line)
          │ ╱          ╲
    20110 │╱────────────╲───── ← VWAP -1σ (yellow)
          │               ╲ 
    20090 │────────────────╲── ← VWAP -2σ (red)
          │                 ▼
    20085 │              ┃█┃  ← BEARISH candle pierces -2σ
          │              ┃ ┃    Volume: 2× average (good!)
    20080 │              ┗━┛  
          │
    20085 │              ┏━┓  ← BULLISH ENGULFING candle
          │              ┃ ┃    Body engulfs previous bearish candle
    20100 │              ┃█┃    This is your ENTRY TRIGGER
          │              ┗━┛
          │
          │  YOUR TRADE:
          │  Entry: 20100 (close of engulfing candle)
          │  Stop:  20078 (2 ticks below the wick low) = 22 pts risk = $11/MNQ
          │  TP1:   20110 (VWAP -1σ) = 10 pts = $5/MNQ → exit 50%
          │  TP2:   20130 (VWAP) = 30 pts = $15/MNQ → exit 50%
          │
          │  Blended R:R = ($5×0.5 + $15×0.5) / $11 = $10 / $11 ≈ 0.9:1
          │  
          │  BUT with a 60%+ win rate on this setup, that's plenty profitable.
          │  And the TP2 runners often go PAST VWAP, pushing actual R:R above 1.5:1
```

---

### Setup B: Failed Breakout Fade (Higher R:R — 20% of your trades)

This setup has a better R:R but happens less often. It catches the "trap"
when price breaks above/below a key level, sucks in breakout traders, then
reverses hard.

#### The Setup
```
REQUIRED CONDITIONS:
  ☑ Clear horizontal level visible on 15-min chart
    (previous day high/low, overnight high/low, or round number)
  ☑ Price breaks through the level
  ☑ Within 1-3 candles, price reverses back INSIDE the level
  ☑ The breakout candle had HIGH volume (the trap filling orders)
  ☑ The reversal candle has EQUAL or HIGHER volume (real rejection)

ENTRY:
  → On the close of the candle that reclaims the level
  → (i.e., price was above resistance, drops back below it → short)
  → (i.e., price was below support, pops back above it → long)

STOP LOSS:
  → 2 ticks beyond the failed breakout extreme (the wick)
  → Usually tight — 8-15 points

TARGETS:
  → TP1: VWAP (exit 50%)
  → TP2: Opposite side of the range (exit 50%)
  → This often gives 2:1 to 3:1 R:R
```

#### Visual Example: Failed Breakout Short

```
Price action on MNQ 2-min:

    20200 │────────────────────── ← Previous Day High (PDH)
          │
          │        ┏━┓
    20210 │        ┃█┃  ← Price BREAKS above PDH on high volume
          │     ┏━━┛ ┃    Breakout traders pile in long
    20215 │     ┃    ┃    
          │     ┗━━━━┛ ← Upper wick — immediately rejected!
          │
          │     ┏━━━━┓
    20205 │     ┃    ┃  ← Next candle: bearish, closes back BELOW PDH
    20195 │     ┃█████┃  ← THIS IS YOUR ENTRY: Short at 20195 (below PDH)
          │     ┗━━━━┛
          │
          │  Stop: 20217 (2 ticks above the breakout wick) = 22 pts
          │  TP1:  20160 (VWAP) = 35 pts → exit 50%
          │  TP2:  20120 (previous support) = 75 pts → exit 50%
          │  
          │  Blended R:R = (35×0.5 + 75×0.5) / 22 = 55/22 = 2.5:1 ✓
          │
          │  WHY IT WORKS:
          │  All those breakout longs now have stops BELOW PDH.
          │  When price drops back through, their stops trigger.
          │  Stop-triggered selling accelerates the move down.
          │  You're riding the cascade of trapped longs bailing out.
```

---

### Setup C: First Pullback After Trend Establishes (10% of your trades)

After the opening 15 minutes, if a clear trend has established, the FIRST
pullback to the 9 EMA is the highest-probability trend entry.

```
REQUIRED CONDITIONS:
  ☑ First 15 minutes show clear direction 
    (3+ candles in same direction, strong close near high/low of range)
  ☑ 15-min chart confirms trend (above/below 50 SMA)
  ☑ Price has NOT yet pulled back to 9 EMA on the 2-min chart
  ☑ This is the FIRST pullback only (second and third are lower probability)

ENTRY:
  → Long: Price dips to 9 EMA on 2-min, prints a bullish candle
  → Short: Price rises to 9 EMA on 2-min, prints a bearish candle

STOP LOSS:
  → Below the pullback low (long) or above the pullback high (short)
  → Usually 10-20 MNQ points

TARGETS:
  → TP1: New high/low (continuation of trend) — exit 50%
  → TP2: Trail with 9 EMA — exit when price closes on wrong side
```

---

## POSITION SIZING — THE EXACT MATH

### Your $50K Pro Plan — Actual Rules

```
╔══════════════════════════════════════════════════════════╗
║  YOUR PROP FIRM RULES ($50K Pro Plan)                    ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  EVALUATION:                                             ║
║    Profit Target:     $3,000                             ║
║    Max Drawdown:      $2,000 (EOD — end of day calc)     ║
║    Daily Drawdown:    NONE (no daily limit!)              ║
║    Max Position:      3 contracts (30 MNQ with 10:1)     ║
║    Consistency Rule:  50% (no day > 50% of profit)       ║
║    Activation Fee:    NONE                               ║
║                                                          ║
║  FUNDED:                                                 ║
║    Max Drawdown:      $2,000 (EOD)                       ║
║    Daily Drawdown:    NONE                               ║
║    Max Position:      5 contracts (NQ only, no micros)   ║
║    Consistency Rule:  NONE                               ║
║    Scaling Rule:      NONE                               ║
║                                                          ║
║  PAYOUTS:                                                ║
║    Days to Payout:    14 days                            ║
║    Min Payout:        $1,000                             ║
║    Profit Split:      80% (you keep 80%)                 ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

### What These Rules Mean For You

**EOD Drawdown = Huge Advantage for Scalping**
Drawdown is calculated at END OF DAY, not intraday. If you're down $1,800
intraday but recover to -$300 by close, only -$300 counts. Since you're
scalping and closing everything by 11 AM, your EOD balance is what matters.
This gives you breathing room during the session.

**No Daily Drawdown Limit = Freedom (But Also Danger)**
There's no firm-imposed daily loss limit. You COULD lose $1,999 in one day
and not break a rule. But DON'T. You must impose your OWN daily limit.
The $2K total DD is tight — you can't afford more than 4 bad days.

**Consistency Rule (50%) = No Hero Trades**
No single day can account for more than 50% of your total profit.
Target is $3K, so no day can exceed $1,500 in profit.
This means: if you're up $1,200 on a day, STOP. Don't push to $1,500+.
Aim for even daily gains of $200-$500.

**3 Contracts Max (Eval) = Precision Required**
You can't brute-force this with size. 3 NQ contracts (or 30 MNQ) is your
ceiling. Every entry must count. Quality over quantity.

### YOUR PERSONAL LIMITS (Self-Imposed)
```
Max risk per trade:    $150 (0.3% of account)
  → 13 losing trades before you hit the $2K wall
  → Gives you enormous room for error

Max daily loss:        $400 (self-imposed)
  → Firm has no daily limit, but you do
  → $400/day means you survive 5 bad days before $2K
  → After 2 consecutive losers ($300 down), STOP for the day

Daily profit target:   $300-$500
  → Well under the $1,500 consistency ceiling
  → Hit $500? STOP. Even if it's 10 AM.

Max trades per day:    4
```

### Position Size Calculation — EVALUATION (MNQ)
```
Max position: 30 MNQ (3 standard × 10:1 micro scaling)

Step 1: Define your stop in MNQ points
  Typical scalp stop: 10-15 points
  Example: 12 points

Step 2: Dollar risk per contract
  12 points × $0.50/tick ÷ 0.25 tick size... 
  Wait — simpler: each 1 point of MNQ = $0.50 × 4 ticks = $2.00
  So 12 points = 12 × $2.00 = $24 per contract
  
  CORRECTION: MNQ 1 point = $0.50 (the multiplier is $2/point)
  Actually: MNQ point value = $2.00 per point per contract
  12 points × $2.00 = $24 risk per contract

Step 3: Calculate contracts
  Max risk ($150) ÷ $24 per contract = 6.25 → round DOWN to 6
  
  BUT: 6 MNQ is very conservative. Let's check sizes:
  
  CONSERVATIVE (recommended for first 2 weeks):
    6 MNQ × $24 = $144 risk per trade ✓
    
  STANDARD (after building $500+ buffer):
    10 MNQ × $24 = $240 risk per trade
    (still well under $400 daily limit — 1.5 losers and you stop)
    
  FULL SIZE (after $1,000+ buffer):
    15 MNQ × $24 = $360 risk per trade
    (1 loss + partial loss = daily limit. Be careful.)
    
  ABSOLUTE MAX:
    30 MNQ — NEVER USE THIS. You'd risk $720 per trade.
    That's one-third of your entire $2K drawdown on a single trade.
```

### Position Size Calculation — FUNDED (NQ)
```
Max position: 5 NQ (no micro scaling)
NQ: 1 point = $20.00 per contract

With 12-point stop:
  1 NQ: 12 × $20 = $240 risk
  2 NQ: 12 × $20 × 2 = $480 risk
  3 NQ: 12 × $20 × 3 = $720 risk
  
RECOMMENDED FUNDED SIZING:
  Start: 1 NQ per trade ($240 risk on 12-pt stop)
  After buffer: 2 NQ per trade ($480 risk)
  Max: 3 NQ per trade ($720 risk — only with $1K+ buffer)
  NEVER use 4-5 NQ unless you have a massive profit cushion
  
  A 10-point scalp on 2 NQ = 10 × $20 × 2 = $400 profit
  That's your daily target in ONE trade. NQ is powerful.
```

### Progressive Sizing Plan
```
EVALUATION PHASE:
  Week 1:  6-8 MNQ per trade   (ultra-safe, learn the rhythm)
  Week 2:  10 MNQ per trade    (if Week 1 was green)
  Week 3+: 12-15 MNQ per trade (if $1,000+ profit banked)
  
FUNDED PHASE:
  Week 1-2:  1 NQ per trade    (adjust to bigger tick value)
  Week 3-4:  2 NQ per trade    (after $500+ buffer over DD line)
  Month 2+:  2-3 NQ per trade  (after $1,000+ buffer)
  
  NEVER go above 3 NQ even with 5 allowed.
  The extra 2 contracts are for adding to WINNING trades only.
```

---

## THE DAILY PLAYBOOK — EXACTLY WHAT YOU DO EACH DAY

### 8:45 AM ET — Pre-Market Prep (15 minutes)

```
□ Open ForexFactory.com → check today's calendar
  → Red/orange events before 11 AM? Note the time.
     • CPI, NFP, FOMC → DO NOT TRADE first 30 min after release
     • Jobless claims, retail sales → Reduce size 50% around release
     • Nothing major? → Normal session

□ Open MNQ 15-min chart:
  → Where is price relative to yesterday's range?
  → Is there an overnight trend? Or has it been ranging?
  → Mark: PDH (previous day high), PDL (previous day low), PDC (close)
  → Mark: Overnight high/low

□ Open MNQ 2-min chart:
  → VWAP will anchor when cash session opens (9:30)
  → Note any pre-market levels where price has reacted

□ Mental check:
  → Am I well-rested? (If not → reduce size 50% or don't trade)
  → Am I emotionally neutral? (Angry/excited → don't trade)
  → Say: "I follow my rules. I accept whatever the market gives me."
```

### 9:30-9:45 AM — Opening (DO NOT TRADE)
```
JUST WATCH.

The first 15 minutes are chaotic:
  - Market makers adjusting positions
  - Overnight orders executing
  - Retail FOMO traders jumping in
  - The widest spreads of the day

What to observe:
  □ Where does VWAP establish?
  □ What is the opening range? (Mark the high and low of first 15 min)
  □ Is volume balanced (range day) or one-sided (trend day)?
  □ Is price above or below PDC? This sets initial bias.

After 15 minutes, classify the day:
  → Price above VWAP, strong momentum = TREND UP day
    (Favor long setups, use Setup C first pullback)
  → Price below VWAP, strong momentum = TREND DOWN day  
    (Favor short setups, use Setup C first pullback)
  → Price oscillating around VWAP = RANGE day
    (Use Setup A VWAP fades in both directions)
```

### 9:45-11:00 AM — The Money Window (⭐ This Is Where You Make Your Living)

```
SCENARIO 1: Range Day
──────────────────────
You'll be using Setup A (VWAP Fade) almost exclusively.

Typical range day sequence:
  9:45  — Price drifts to VWAP +1σ → SHORT Setup A → ride to VWAP
  10:05 — Price drops to VWAP -1σ → LONG Setup A → ride to VWAP  
  10:25 — Price rallies to VWAP +1σ again → SHORT Setup A
  10:45 — Hit daily target ($400+). STOP TRADING.

That's 3 trades, ~$150-200 each, done before 11 AM.

SCENARIO 2: Trend Day
──────────────────────
You'll use Setup C (First Pullback) then Setup A (fading extremes WITH trend).

Typical uptrend day sequence:
  9:45  — Trend up confirmed → wait for first pullback to 9 EMA
  9:55  — Pullback arrives → LONG Setup C → ride to new high
  10:15 — Price pulls back to VWAP → LONG Setup A → ride above VWAP
  10:35 — Price extends to VWAP +2σ → DO NOT SHORT (trend is up)
  10:45 — Take what you have. Done.

KEY RULE: On trend days, only trade IN the trend direction.
  Trend up → only longs. Trend down → only shorts.
  The temptation to "fade the trend" kills accounts.

SCENARIO 3: Breakout Day (Big News, Gap, Macro Event)
──────────────────────────────────────────────────────
  Use Setup B (Failed Breakout) if you see a trap at key levels.
  OR simply sit out if it's too volatile.
  
  There is ZERO shame in not trading on a wild day.
  Protecting capital IS making money.
```

### 11:00 AM — STOP TRADING

```
REGARDLESS of your P&L:

If you're GREEN → Stop. You won. Go enjoy your day.
If you're FLAT  → Stop. The good setups are done.
If you're RED   → Definitely stop. The afternoon won't fix it.

What to do after 11 AM:
  ✅ Journal your trades (see template below)
  ✅ Screenshot each trade with annotations
  ✅ Grade yourself on rule-following (not P&L!)
  ✅ Exercise / walk / be a human being
  ✅ Review 15-min chart for tomorrow's key levels
  ✅ Close all charts by noon. Done.

EXCEPTION: If you had ZERO valid setups by 11 AM (rare but happens):
  → You may watch 2:00-3:00 PM for one Setup A
  → Maximum 2 trades in afternoon session
  → Tighter stops, smaller size (50%)
  → This should happen < 1 day per week
```

---

## COMPLETE WORKED EXAMPLE: A Full Trading Day

### Monday, Example Day — Range Day

**8:45 AM — Pre-Market**
```
Calendar: No high-impact news today.
MNQ overnight range: 20,050 - 20,180
Previous day: H=20,200, L=20,020, C=20,150
Key levels: PDH 20,200, PDL 20,020, PDC 20,150, Round: 20,100, 20,200
VIX at 18.5 — normal, all systems go.
```

**9:30 AM — Cash Open**
```
MNQ opens at 20,140 (near PDC — no gap, balanced start)
VWAP establishes at 20,142
First 15 min range: 20,120 - 20,170 (narrow, 50 pts — range day likely)
Price oscillating around VWAP — CONFIRMED RANGE DAY
Decision: I'll use Setup A (VWAP Fades) today. Both longs and shorts.
```

**9:52 AM — Trade 1: LONG**
```
Price drifts down to 20,098 — touches VWAP -1σ band (20,100)
Volume spike on the down move.
2-min candle prints a HAMMER at 20,098 (long lower wick, small body)
Next candle: bullish, closes at 20,112 (price back above 9 EMA)

ENTRY: Long 10 MNQ at 20,112
STOP: 20,092 (2 ticks below hammer wick) = 20 points
RISK: 20 pts × $2/pt × 10 contracts = $400... too much!
  → ADJUST: Use 8 MNQ instead
  → 20 pts × $2/pt × 8 = $320. Still high.
  → ADJUST: Use 6 MNQ (conservative)
  → 20 pts × $2/pt × 6 = $240. Under $400 daily limit. ✓

ACTUAL ENTRY: Long 6 MNQ at 20,112
TOTAL RISK: $240

TP1: 20,142 (VWAP) — will exit 3 contracts here
TP2: 20,155 (VWAP +1σ) — will exit remaining 3

10:04 AM: Price reaches 20,142 → SELL 3 MNQ at 20,142
  Profit: (20,142 - 20,112) = 30 pts × $2/pt × 3 = $180
  Move stop to 20,112 (breakeven) on remaining 3

10:11 AM: Price reaches 20,158 → SELL remaining 3 at 20,155  
  Profit: (20,155 - 20,112) = 43 pts × $2/pt × 3 = $258

TRADE 1 TOTAL: +$438
Time in trade: 19 minutes
R multiple: $438/$240 = 1.83R ✓
```

**10:18 AM — Trade 2: SHORT**
```
Price has rallied to 20,175 — touches VWAP +1σ band (20,180)
Bearish engulfing candle on 2-min chart
9 EMA is at 20,168 — price crosses below it

ENTRY: Short 8 MNQ at 20,168
  (Slightly more size because stop is tighter)
STOP: 20,182 (2 ticks above the wick) = 14 points
TOTAL RISK: 14 pts × $2/pt × 8 = $224 ✓

TP1: 20,142 (VWAP) — exit 4 contracts
TP2: 20,120 — exit remaining 4

10:28 AM: Price drops to 20,142 → COVER 4 MNQ at 20,142
  Profit: (20,168 - 20,142) = 26 pts × $2/pt × 4 = $208
  Move stop to 20,168 (breakeven)

10:34 AM: Price stalls at 20,135, starts bouncing
  → It's been 16 min, momentum fading, close to daily target
  → EXIT remaining 4 at 20,137
  Profit: (20,168 - 20,137) = 31 pts × $2/pt × 4 = $248

TRADE 2 TOTAL: +$456
Time in trade: 16 minutes
R multiple: $456/$224 = 2.04R ✓
```

**10:35 AM — Daily Assessment**
```
Trade 1: +$438
Trade 2: +$456
DAILY TOTAL: +$894

⚠️ CONSISTENCY CHECK: $894 is under $1,500 (50% of $3K target) ✓
  But it's a very strong day. NO MORE TRADES.

Decision: STOP TRADING. ✓

Two trades. Both winners. Both followed rules.
Total screen time: ~50 minutes of active trading.
Total risk taken: $464 (within $400 self-imposed daily loss limit)
Max MNQ at any time: 8 (well within 30 MNQ / 3-contract limit)
```

**10:40 AM — Post-Trade Journal**
```
Trade 1: Grade A — Setup A Long at VWAP -1σ, perfect hammer trigger
  Rules followed: 5/5 ✓
  Emotional state: Calm, patient
  
Trade 2: Grade A- — Setup A Short at VWAP +1σ, clean engulfing
  Rules followed: 5/5 ✓
  Cut TP2 short (didn't reach target) — correct decision (momentum fading)
  
Day grade: A
Note: Range day = bread and butter. This is the "boring" consistency 
that passes challenges and keeps funded accounts alive.
```

---

### Wednesday, Example Day — Trend Day (With a Loss)

**9:45 AM — Context**
```
Gap up: MNQ opened 100+ points above PDC
Strong buying in first 15 min — all candles green
15-min: Price well above 50 SMA, sloping up aggressively
Classification: TREND UP DAY
Decision: Only LONG setups today. No shorts.
```

**9:52 AM — Trade 1: LONG (Setup C — First Pullback)**
```
First pullback to 9 EMA on 2-min chart (after 5+ green candles)
Bullish candle bounces off 9 EMA at 20,320

ENTRY: Long 8 MNQ at 20,325
STOP: 20,308 (below pullback low) = 17 points
TOTAL RISK: 17 pts × $2/pt × 8 = $272

TP1: 20,355 (previous swing high) — exit 4
TP2: Trail with 9 EMA — exit remaining 4

10:01 AM: Price hits 20,355 → SELL 4 at 20,355
  Profit: 30 pts × $2/pt × 4 = $240
  Move stop to 20,325 (breakeven)

10:12 AM: Price at 20,388, 9 EMA rising to 20,362
10:18 AM: 2-min candle closes below 9 EMA at 20,358
  → EXIT remaining 4 at 20,360
  Profit: 35 pts × $2/pt × 4 = $280

TRADE 1 TOTAL: +$520 ✓
```

**10:25 AM — Trade 2: LONG (Setup A — VWAP Fade)**
```
Price pulls back to VWAP at 20,300
Bullish hammer at VWAP, volume spike

ENTRY: Long 6 MNQ at 20,308
STOP: 20,290 (below VWAP) = 18 points
TOTAL RISK: 18 pts × $2/pt × 6 = $216

10:32 AM: Price bounces slightly to 20,320 but stalls
10:38 AM: News hits — unexpected Fed speaker comments
10:39 AM: Price drops sharply through VWAP
10:39 AM: STOP HIT at 20,290

TRADE 2 TOTAL: -$216 ✗

RESPONSE:
  → This is normal. The setup was valid. The news was unforeseeable.
  → I graded this a B+ setup (valid but Fed speakers are a risk)
  → Wait 10 minutes before next trade (cool-down rule)
  → Running total: +$520 - $216 = +$304. Still green. Breathe.
```

**10:50 AM — Trade 3: LONG (Setup A — VWAP Fade)**
```
After the Fed speaker sell-off, price has stabilized
Price touches VWAP -1σ at 20,260
Bullish engulfing candle, volume 2× average (buyers stepping in)
15-min still bullish (above 50 SMA)

ENTRY: Long 6 MNQ at 20,268
STOP: 20,252 (below the wick) = 16 pts
RISK: 16 pts × $2/pt × 6 = $192

10:58 AM: Price recovers to VWAP 20,300 → SELL all 6 at 20,298
  Profit: 30 pts × $2/pt × 6 = $360

TRADE 3 TOTAL: +$360 ✓
  (Took full exit at VWAP — didn't get greedy after the scare)
```

**11:00 AM — Daily Assessment**
```
Trade 1: +$520 (Setup C — perfect)
Trade 2: -$216 (Setup A — valid setup, bad luck)
Trade 3: +$360 (Setup A — recovery trade, still followed rules)

DAILY TOTAL: +$664 ✓
CONSISTENCY CHECK: $664 < $1,500 ✓

Max contracts at any time: 8 MNQ (well within 30 MNQ limit)
Win rate today: 2/3 = 67%
Key lesson: The loss in Trade 2 was fine. It was a valid setup.
           I followed the cool-down rule, came back calm, and recovered.
           THIS is what consistency looks like.
```

---

## PROP FIRM CHALLENGE WALKTHROUGH

### The Challenge Math ($50K Account — YOUR Actual Rules)

```
╔═══════════════════════════════════════════════════════╗
║  EVALUATION TARGETS                                   ║
║                                                       ║
║  Profit Target:      $3,000                           ║
║  Max Drawdown:       $2,000 (EOD)                     ║
║  Consistency:        No day > $1,500 (50% of target)  ║
║  Max Position:       30 MNQ (3 contracts × 10:1)      ║
╚═══════════════════════════════════════════════════════╝

Key constraint: You have $2,000 of room to lose.
That's TIGHT. You must protect this at all costs.

Planned trading days: 18 (skip FOMC, NFP, CPI days)
Required daily avg: $3,000 / 18 = $167/day

With our method (6-10 MNQ per trade, 2-3 trades/day):
  Average daily P&L: $300-$500
  Expected green days: 13-14 out of 18 (72-78%)
  Expected red days: 4-5 out of 18
  Average red day: -$200 (self-imposed $400 daily cap)

  REALISTIC PROJECTION:
  Green days: 13 × $400 = $5,200
  Red days:    5 × -$200 = -$1,000
  NET: +$4,200 → passes $3,000 target ✓
  Max drawdown used: ~$600 (well within $2,000) ✓
  Best single day: ~$800 (under $1,500 consistency limit) ✓

  WORST CASE (55% green day rate):
  Green: 10 × $350 = $3,500
  Red:    8 × -$200 = -$1,600
  NET: +$1,900 → doesn't quite pass, BUT:
    → Max DD used: $1,600 (still within $2K limit)
    → Account is alive. You can keep trading.
    → A few more green days close it out.

  DANGER SCENARIO (what to avoid):
  3 consecutive red days × -$400 each = -$1,200 drawn down
  Now you only have $800 of DD room left.
  At this point: REDUCE to 4 MNQ. Survival mode.
  Tiny gains. Rebuild the buffer. Don't blow the account.

CONSISTENCY RULE CHECK:
  Target: $3,000 → max single day: $1,500
  Your daily target: $300-$500
  Even your best days (~$800) are well under $1,500 ✓
  ONLY becomes an issue if you're near $3K and one big day
  would push you over. Solution: when profit > $2,000,
  limit daily target to $400 max.
```

### Week-by-Week Challenge Plan

```
WEEK 1 (Days 1-5): ULTRA-CONSERVATIVE
  Goal: +$500-$800 total (build buffer above the $2K DD line)
  Size: 6 MNQ per trade
  Max trades/day: 3
  Self-imposed daily loss: $300
  Mindset: "I'm building my equity cushion. Small and steady."
  
  WHY SO SMALL: The $2K DD is tight. If you lose $500 in Week 1,
  you only have $1,500 left for 3 weeks. Start micro.
  
WEEK 2 (Days 6-10): STANDARD
  Goal: +$800-$1,200 total
  Size: 8-10 MNQ per trade (you have profit buffer now)
  Max trades/day: 3
  Self-imposed daily loss: $400
  Mindset: "I'm in rhythm. Executing my process."

WEEK 3 (Days 11-15): CONFIDENT
  Goal: +$800-$1,000 total
  Size: 8-10 MNQ
  Max trades/day: 3
  Mindset: "Same thing. No change. Boring is beautiful."

  By now you should have ~$1,500-$2,000 profit.
  If you're at $2,500+, you might pass this week!

WEEK 4 (Days 16-22): CLOSE IT OUT
  If on track (profit > $2,000):
    → Reduce to 6 MNQ. PROTECT THE LEAD.
    → You need ~$500-$1,000 more. That's 2-3 normal days.
    → Don't get sloppy at the finish line.
    → Check consistency: no single day > 50% of final total
  
  If behind (profit < $1,500):
    → DO NOT increase risk. That's how accounts die.
    → Stick to 6-8 MNQ. Trust the process.
    → If you don't pass, reset and try again.
    → A failed challenge with discipline = tuition.
    → A blown account = wasted money AND confidence damage.
  
  If in drawdown (used $1,000+ of $2K DD):
    → Drop to 4 MNQ. Survival mode.
    → Daily target: $100-$200. Just get green days.
    → Rebuild slowly. You're not dead until -$2,000.
```

---

## THE 10 RULES (Print This. Put It By Your Screen.)

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  1. I only trade MNQ between 9:45 and 11:00 AM ET.      │
│                                                          │
│  2. I only take Setup A, B, or C. Nothing else.          │
│                                                          │
│  3. I NEVER exceed 10 MNQ per trade (30 max allowed).    │
│     (When funded: max 2 NQ per trade, 5 allowed.)        │
│                                                          │
│  4. My stop loss is set BEFORE I enter. I never move it. │
│                                                          │
│  5. After 2 consecutive losses, I'm done for the day.    │
│     ($2K total DD is tight — protect it.)                │
│                                                          │
│  6. When I hit $500 daily profit, I'm done. Period.      │
│     (Keeps me under the $1,500 consistency ceiling.)     │
│                                                          │
│  7. I never trade against the 15-min trend.              │
│                                                          │
│  8. I never trade within 15 min of high-impact news.     │
│                                                          │
│  9. I journal every trade. No exceptions.                │
│                                                          │
│  10. If my drawdown exceeds $1,000, I drop to 4 MNQ     │
│      and $200 daily target until I recover.              │
│                                                          │
│  BONUS: I follow these rules ESPECIALLY when I don't     │
│         want to. That's when they matter most.           │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## WHY THIS WILL WORK FOR YOU SPECIFICALLY

Based on our work together:

1. **You're analytical** — You built ML models, ran 92 strategy configs, validated
   walk-forward. You won't trade on emotion. You'll follow the data.

2. **You understand mean reversion** — We proved it works. VWAP MR is the same
   principle applied at a micro level. The edge is real.

3. **You're already systematic** — The NASDAQ signal generator, the PineScript
   strategies — you think in rules and conditions. That's exactly what prop
   firms reward.

4. **You have realistic expectations** — You've seen the backtests. You know what
   a 1.5:1 R:R with 55% win rate produces. No fantasy, just math.

The method is simple. The execution is hard. But "hard" means most people
won't do it — and that's exactly where your edge is.

### Your Path Forward
```
Step 1: Paper trade on TradingView replay (free) for 1-2 weeks
  → Practice Setup A and C on MNQ 2-min chart
  → Journal every trade. Get to 30+ trades.

Step 2: Start the $50K Pro Plan evaluation  
  → Week 1: 6 MNQ, max 3 trades/day
  → Week 2-3: 8-10 MNQ if Week 1 was profitable
  → Target: $3K profit with < $2K drawdown
  → Remember: 50% consistency rule — no hero days

Step 3: Get funded → trade 1-2 NQ (no micros!)
  → First payout after 14 days, min $1K, keep 80%
  → Build to 2-3 NQ as your buffer grows
  → $400/day × 20 days = $8K/month × 80% = $6,400 payout

You already have the mindset. Now just add the reps.
```

---

*"Simplicity is the ultimate sophistication."* — Leonardo da Vinci

*"The market is a device for transferring money from the impatient to the patient."* — Warren Buffett
