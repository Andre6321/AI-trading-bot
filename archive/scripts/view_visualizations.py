"""
Quick script to display all generated visualizations.
"""

import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def display_visualizations():
    """Display all generated visualizations."""
    viz_dir = Path('outputs/visualizations')
    
    if not viz_dir.exists():
        print("❌ Visualization directory not found!")
        print("   Run: python scripts/visualize_performance.py")
        return
    
    images = list(viz_dir.glob('*.png'))
    
    if not images:
        print("❌ No visualizations found!")
        return
    
    print("="*70)
    print("📊 AVAILABLE VISUALIZATIONS")
    print("="*70)
    
    for i, img_path in enumerate(sorted(images), 1):
        print(f"\n{i}. {img_path.name}")
        print(f"   Path: {img_path.absolute()}")
        print(f"   Size: {img_path.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "="*70)
    print("💡 To view these images:")
    print("="*70)
    print("   1. Navigate to: outputs/visualizations/")
    print("   2. Open any .png file")
    print("\nGenerated visualizations:")
    print("   • model_comparison.png - Unified vs regime-specific performance")
    print("   • regime_timeline.png - 5 years of BTC with regime detection")
    print("   • regime_characteristics.png - Regime behavior analysis")
    print("   • model_evolution.png - Model improvement journey")
    print("\n" + "="*70)

if __name__ == '__main__':
    display_visualizations()
