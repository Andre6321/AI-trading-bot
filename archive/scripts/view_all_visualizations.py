"""
Quick visualization viewer - Opens all generated charts
"""
import webbrowser
from pathlib import Path
import time

viz_dir = Path('outputs/visualizations')

print("=" * 70)
print("🎨 OPENING AI TRADING SYSTEM VISUALIZATIONS")
print("=" * 70)

visualizations = [
    ('performance_showcase.png', '📈 Main Performance Comparison'),
    ('metrics_dashboard.png', '📊 Performance Metrics Dashboard'),
    ('regime_performance.png', '🎯 Regime-Specific Performance'),
    ('trade_distribution.png', '📉 Win/Loss Distribution Analysis')
]

for filename, description in visualizations:
    filepath = viz_dir / filename
    if filepath.exists():
        print(f"\n✅ {description}")
        print(f"   Opening: {filepath}")
        webbrowser.open(str(filepath.absolute()))
        time.sleep(0.5)  # Stagger openings
    else:
        print(f"\n⏳ {description}")
        print(f"   Waiting for: {filename}")

print("\n" + "=" * 70)
print("💡 TIP: Check outputs/PRESENTATION_SUMMARY.md for detailed explanations")
print("=" * 70)
