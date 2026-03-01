import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet('data/processed/btcusdt_1h_with_regimes.parquet')

print("="*70)
print("📊 MARKET REGIME DETECTION RESULTS")
print("="*70)
print(f"\nTotal samples analyzed: {len(df):,}")
print(f"Timeframe: 1-hour candles")
print(f"Duration: ~{len(df) / 24:.0f} days ({len(df) / 24 / 365:.1f} years)")

print(f"\n{'='*70}")
print("REGIME DISTRIBUTION")
print("="*70)

regime_counts = df["regime_name"].value_counts()
for regime in ["BEAR", "SIDEWAYS", "BULL"]:
    if regime in regime_counts.index:
        count = regime_counts[regime]
        pct = count / len(df) * 100
        print(f"\n{regime:>10}: {count:>6,} samples ({pct:>5.1f}%)")
        
        regime_data = df[df["regime_name"] == regime]
        
        # Calculate returns using log returns
        avg_ret_1h = regime_data['log_ret_1h'].mean() * 100
        avg_ret_4h = regime_data['log_ret_4h'].mean() * 100
        avg_ret_24h = regime_data['log_ret_24h'].mean() * 100
        
        print(f"             Avg 1h return:  {avg_ret_1h:>+6.3f}%")
        print(f"             Avg 4h return:  {avg_ret_4h:>+6.3f}%")
        print(f"             Avg 24h return: {avg_ret_24h:>+6.3f}%")
        
        # Price statistics
        avg_price = regime_data['close'].mean()
        min_price = regime_data['close'].min()
        max_price = regime_data['close'].max()
        
        print(f"             Avg price: ${avg_price:>10,.2f}")
        print(f"             Range: ${min_price:>10,.2f} - ${max_price:>10,.2f}")

print(f"\n{'='*70}")
print("KEY INSIGHTS")
print("="*70)

# Regime transitions
transitions = []
prev_regime = df["regime"].iloc[0]
for i in range(1, len(df)):
    curr_regime = df["regime"].iloc[i]
    if curr_regime != prev_regime:
        transitions.append({
            'from': df["regime_name"].iloc[i-1],
            'to': df["regime_name"].iloc[i],
            'index': i
        })
        prev_regime = curr_regime

print(f"\nTotal regime transitions: {len(transitions)}")
print(f"Avg regime duration: {len(df) / (len(transitions) + 1):.1f} hours")

# Current regime
current_regime = df["regime_name"].iloc[-1]
print(f"\n⚡ CURRENT MARKET REGIME: {current_regime}")

# Recent transitions
print(f"\nLast 5 regime changes:")
for t in transitions[-5:]:
    print(f"  Sample {t['index']}: {t['from']} → {t['to']}")

print(f"\n{'='*70}")
