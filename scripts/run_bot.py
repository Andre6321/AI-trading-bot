"""
Launch script for the trading bot.

Usage:
    python scripts/run_bot.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bybit_ai_trader.main import main

if __name__ == "__main__":
    print("=" * 60)
    print("BYBIT AI TRADING BOT")
    print("=" * 60)
    print()
    print("Starting bot...")
    print()
    
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
