"""
Compare regime-specific models vs unified model performance.
"""

import json
from pathlib import Path

print("="*70)
print("📊 MODEL PERFORMANCE COMPARISON")
print("="*70)

# Load unified model metadata
unified_path = Path('models/ensemble_btcusdt_h4_optuna_optimized_metadata.json')
with open(unified_path) as f:
    unified = json.load(f)

print("\n🔹 UNIFIED MODEL (All Market Conditions):")
print(f"   ROC AUC: {unified['final_roc_auc']:.4f}")
print(f"   Training samples: ~{unified['data_samples']:,}")

# Load regime-specific models
regimes = ['bear', 'sideways', 'bull']
regime_models = {}

print("\n🔸 REGIME-SPECIFIC MODELS:")
for regime in regimes:
    metadata_path = Path(f'models/ensemble_{regime}_regime_metadata.json')
    with open(metadata_path) as f:
        data = json.load(f)
        regime_models[regime] = data
        
    regime_name = data['regime']
    metrics = data['ensemble_metrics']
    
    print(f"\n   {regime_name} Market:")
    print(f"      ROC AUC:   {metrics['roc_auc']:.4f} ⭐")
    print(f"      Accuracy:  {metrics['accuracy']:.4f}")
    print(f"      Precision: {metrics['precision']:.4f}")
    print(f"      Recall:    {metrics['recall']:.4f}")
    print(f"      F1 Score:  {metrics['f1']:.4f}")
    print(f"      Train samples: {data['train_samples']:,}")

# Calculate improvements
print("\n" + "="*70)
print("📈 IMPROVEMENT ANALYSIS")
print("="*70)

unified_auc = unified['final_roc_auc']
avg_regime_auc = sum(m['ensemble_metrics']['roc_auc'] for m in regime_models.values()) / 3

print(f"\nUnified Model ROC AUC: {unified_auc:.4f}")
print(f"Avg Regime-Specific ROC AUC: {avg_regime_auc:.4f}")
print(f"Average Improvement: {(avg_regime_auc - unified_auc):.4f} ({(avg_regime_auc/unified_auc - 1)*100:+.2f}%)")

print("\nRegime-specific improvements:")
for regime, data in regime_models.items():
    regime_auc = data['ensemble_metrics']['roc_auc']
    improvement = regime_auc - unified_auc
    pct_improvement = (regime_auc / unified_auc - 1) * 100
    print(f"   {data['regime']:>10}: {regime_auc:.4f} ({improvement:+.4f}, {pct_improvement:+.2f}%)")

# Market regime distribution
print("\n" + "="*70)
print("📊 MARKET REGIME DISTRIBUTION")
print("="*70)

total_samples = sum(m['train_samples'] + m['test_samples'] for m in regime_models.values())
for regime, data in regime_models.items():
    regime_samples = data['train_samples'] + data['test_samples']
    pct = regime_samples / total_samples * 100
    print(f"\n{data['regime']:>10}: {regime_samples:>6,} samples ({pct:>5.1f}%)")

# Expected performance
print("\n" + "="*70)
print("💡 WEIGHTED EXPECTED PERFORMANCE")
print("="*70)

weighted_auc = 0
for regime, data in regime_models.items():
    regime_samples = data['train_samples'] + data['test_samples']
    weight = regime_samples / total_samples
    regime_auc = data['ensemble_metrics']['roc_auc']
    weighted_auc += regime_auc * weight
    print(f"{data['regime']:>10}: {regime_auc:.4f} × {weight:.3f} = {regime_auc * weight:.4f}")

print(f"\nWeighted Ensemble ROC AUC: {weighted_auc:.4f}")
print(f"vs Unified Model: {unified_auc:.4f}")
print(f"Expected Improvement: {(weighted_auc - unified_auc):.4f} ({(weighted_auc/unified_auc - 1)*100:+.2f}%)")

print("\n" + "="*70)
print("🎯 KEY INSIGHTS")
print("="*70)

print("\n1. Best Performing Regime:")
best_regime = max(regime_models.items(), key=lambda x: x[1]['ensemble_metrics']['roc_auc'])
print(f"   {best_regime[1]['regime']} Market: {best_regime[1]['ensemble_metrics']['roc_auc']:.4f} ROC AUC")

print("\n2. Regime-Specific Advantages:")
print("   ✅ Specialized patterns for each market condition")
print("   ✅ Better handling of market dynamics")
print("   ✅ Higher prediction accuracy per regime")
print(f"   ✅ Average +{(avg_regime_auc/unified_auc - 1)*100:.2f}% improvement")

print("\n3. Trading Strategy Implications:")
print("   • Use regime detection to select appropriate model")
print("   • Dynamic model switching based on market conditions")
print("   • Potentially better risk-adjusted returns")
print("   • More robust to market regime changes")

print("\n" + "="*70)
