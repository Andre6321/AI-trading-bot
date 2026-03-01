"""Analyze V5 experiment results - separate real from biased."""
import json

data = json.load(open('outputs/v5_experiment_results.json'))
print(f'Total results: {len(data)}')

# Split by source
sources = {}
for r in data:
    lbl = r['label']
    if lbl.startswith('4H|'):
        src = '4H'
    elif lbl.startswith('retrain_'):
        src = 'retrain'
    elif lbl.startswith('rule_'):
        src = 'rule'
    else:
        src = '1H_ML'
    sources.setdefault(src, []).append(r)

for s, rl in sources.items():
    prof = [r for r in rl if r['pf'] > 1.0]
    print(f'{s}: {len(rl)} total, {len(prof)} profitable')

# Show best 1H ML strategies with FIXED exits only (no trailing stop bias)
ml = sources.get('1H_ML', [])
fixed = [r for r in ml if r['pf'] > 1.0 and 'trail' not in r['label'].lower()]
fixed.sort(key=lambda x: x['pf'], reverse=True)
print(f'\n=== 1H ML FIXED-EXIT PROFITABLE STRATEGIES (no trailing bias): {len(fixed)} ===')
for r in fixed[:20]:
    print(f"  PF={r['pf']:.3f} WR={r['wr']:.1%} Ret={r['ret']:+.1f}% Trades={r['n_trades']} DD={r['max_dd']:.1f}% Sh={r['sharpe']:.2f}  {r['label']}")

# Show best 1H ML strategies with trailing stops
trail = [r for r in ml if r['pf'] > 1.0 and 'trail' in r['label'].lower()]
trail.sort(key=lambda x: x['pf'], reverse=True)
print(f'\n=== 1H ML TRAILING PROFITABLE STRATEGIES: {len(trail)} ===')
for r in trail[:10]:
    print(f"  PF={r['pf']:.3f} WR={r['wr']:.1%} Ret={r['ret']:+.1f}% Trades={r['n_trades']} DD={r['max_dd']:.1f}% Sh={r['sharpe']:.2f}  {r['label']}")

# Check 4H with fixed exits
h4 = sources.get('4H', [])
h4_fixed = [r for r in h4 if r['pf'] > 1.0 and 'trailing' not in r['label']]
print(f'\n=== 4H FIXED-EXIT PROFITABLE: {len(h4_fixed)} ===')
for r in h4_fixed[:10]:
    print(f"  PF={r['pf']:.3f} WR={r['wr']:.1%} Ret={r['ret']:+.1f}% Trades={r['n_trades']} DD={r['max_dd']:.1f}% Sh={r['sharpe']:.2f}  {r['label']}")

# 4H with breakeven
h4_be = [r for r in h4 if r['pf'] > 1.0 and 'breakeven' in r['label']]
print(f'\n=== 4H BREAKEVEN PROFITABLE: {len(h4_be)} ===')
for r in h4_be[:10]:
    print(f"  PF={r['pf']:.3f} WR={r['wr']:.1%} Ret={r['ret']:+.1f}% Trades={r['n_trades']} DD={r['max_dd']:.1f}% Sh={r['sharpe']:.2f}  {r['label']}")

# Analyze trailing stop stats more - check if WR is suspiciously high
print('\n=== TRAILING STOP SANITY CHECK ===')
trail_all = [r for r in data if r['pf'] > 1.0 and 'trailing' in r['label'] or 'trail0.75' in r['label']]
trail_all.sort(key=lambda x: x['pf'], reverse=True)
# Check if 1H trailing also has abnormally high WR
trail_1h = [r for r in ml if r['pf'] > 1.0 and ('trail' in r['label'].lower())]
if trail_1h:
    avg_wr = sum(r['wr'] for r in trail_1h) / len(trail_1h)
    avg_pf = sum(r['pf'] for r in trail_1h) / len(trail_1h)
    print(f'1H trailing: {len(trail_1h)} profitable, avg WR={avg_wr:.1%}, avg PF={avg_pf:.2f}')

trail_4h = [r for r in h4 if r['pf'] > 1.0 and 'trailing' in r['label']]
if trail_4h:
    avg_wr = sum(r['wr'] for r in trail_4h) / len(trail_4h)
    avg_pf = sum(r['pf'] for r in trail_4h) / len(trail_4h)
    print(f'4H trailing: {len(trail_4h)} profitable, avg WR={avg_wr:.1%}, avg PF={avg_pf:.2f}')

# Check the best FIXED 1H strategies in more detail
print('\n=== BEST REALISTIC STRATEGIES (1H, fixed exit) ===')
for r in fixed[:10]:
    print(f"\n  {r['label']}")
    print(f"    PF={r['pf']:.3f} WR={r['wr']:.1%} Trades={r['n_trades']}")
    print(f"    Return={r['ret']:+.1f}% MaxDD={r['max_dd']:.1f}%")
    print(f"    Sharpe={r['sharpe']:.2f}")
    print(f"    L/S: {r['n_long']}/{r['n_short']}")

# Also check breakeven exit strategies on 1H
be_1h = [r for r in ml if r['pf'] > 1.0 and 'breakeven' in r['label'].lower()]
print(f'\n=== 1H BREAKEVEN PROFITABLE: {len(be_1h)} ===')
for r in be_1h[:10]:
    print(f"  PF={r['pf']:.3f} WR={r['wr']:.1%} Ret={r['ret']:+.1f}% Trades={r['n_trades']} DD={r['max_dd']:.1f}%  {r['label']}")
