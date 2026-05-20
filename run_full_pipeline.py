#!/usr/bin/env python3
"""
Advanced USD/INR Trading Bot - Full Pipeline
Trains, tests, and analyzes the model in one execution.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_step(name, command):
    """Run a pipeline step and report results."""
    print("\n" + "="*60)
    print(f"▶ {name}")
    print("="*60)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ {name} completed successfully")
            return True
        else:
            print(f"❌ {name} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ Error running {name}: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("  USD/INR ADVANCED TRADING BOT")
    print("  Full Pipeline Execution")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    steps = [
        ("Step 1/3: Training Model (1M timesteps, ~45-60 min)", 
         "python train_agent.py"),
        
        ("Step 2/3: Testing on OOS Data (2016-2022)",
         "python test_agent.py"),
        
        ("Step 3/3: Advanced Analysis (8-panel visualization)",
         "python advanced_analysis.py"),
    ]
    
    results = []
    for step_name, command in steps:
        success = run_step(step_name, command)
        results.append((step_name, success))
        
        if not success:
            print("\n⚠️  Pipeline stopped due to error")
            print(f"Failed at: {step_name}")
            break
    
    # Summary
    print("\n" + "="*60)
    print("  PIPELINE SUMMARY")
    print("="*60)
    
    for step_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {step_name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated Files:")
        print("  📊 model_eurusd_best.zip         - Trained model")
        print("  📈 equity_curve.png              - Training equity curve")
        print("  📈 test_equity_curve.png         - Test equity curve")
        print("  📊 trading_analysis_advanced.png - 8-panel analysis")
        print("  📋 trade_history_output.csv      - Detailed trade list")
        print("  📊 trading_analysis.png          - Original analysis")
        print("\n" + "="*60)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")
        return 0
    else:
        print("\n" + "="*60)
        print("❌ PIPELINE INCOMPLETE - See errors above")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
