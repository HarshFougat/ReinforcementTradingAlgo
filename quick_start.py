#!/usr/bin/env python3
"""
Quick Start Script - USD/INR Advanced Trading Bot
Install dependencies and start training in one command
"""

import subprocess
import sys
import os

def main():
    print("\n" + "="*70)
    print(" USD/INR ADVANCED TRADING BOT - QUICK START".center(70))
    print("="*70)
    
    # Step 1: Install dependencies
    print("\n[STEP 1/3] Installing dependencies...")
    print("-"*70)
    
    packages = ['pandas', 'numpy', 'stable-baselines3', 'gymnasium', 'pandas-ta', 'matplotlib', 'seaborn']
    
    cmd = [sys.executable, '-m', 'pip', 'install', '--break-system-packages', '-q'] + packages
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode == 0:
        print("✅ Dependencies installed successfully!")
    else:
        print("⚠️  Installation had issues, but attempting to continue...")
        print("If errors occur, run manually:")
        print(f"  {' '.join(cmd)}")
    
    # Step 2: Start training
    print("\n[STEP 2/3] Starting model training...")
    print("-"*70)
    print("Expected time: 45-60 minutes")
    print("(This window will show training progress)\n")
    
    result = subprocess.run([sys.executable, 'train_agent.py'], timeout=3600)
    
    if result.returncode != 0:
        print("❌ Training failed!")
        return 1
    
    # Step 3: Run analysis
    print("\n[STEP 3/3] Running tests and analysis...")
    print("-"*70)
    
    subprocess.run([sys.executable, 'test_agent.py'], timeout=600)
    subprocess.run([sys.executable, 'advanced_analysis.py'], timeout=600)
    
    # Summary
    print("\n" + "="*70)
    print(" ✅ COMPLETE!".center(70))
    print("="*70)
    print("\nGenerated Files:")
    print("  📊 trading_analysis_advanced.png  (8-panel analysis)")
    print("  📈 equity_curve.png              (training curve)")
    print("  📈 test_equity_curve.png         (test curve)")
    print("  📋 trade_history_output.csv      (all trades)")
    print("  🤖 model_eurusd_best.zip         (trained model)")
    print("\n" + "="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
