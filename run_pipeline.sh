#!/bin/bash

# Advanced USD/INR Trading Bot Pipeline
# Trains model with 25+ indicators + risk management, then tests and analyzes

echo "=========================================="
echo "USD/INR Advanced Trading Bot"
echo "Full Pipeline Execution"
echo "=========================================="
echo ""

# Step 1: Train
echo "Step 1/3: Training enhanced model..."
echo "-------------------------------------------"
python train_agent.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi
echo ""

# Step 2: Test
echo "Step 2/3: Testing model on OOS data (2016-2022)..."
echo "-------------------------------------------"
python test_agent.py
if [ $? -ne 0 ]; then
    echo "❌ Testing failed!"
    exit 1
fi
echo ""

# Step 3: Analyze
echo "Step 3/3: Generating advanced analysis..."
echo "-------------------------------------------"
python advanced_analysis.py
if [ $? -ne 0 ]; then
    echo "❌ Analysis failed!"
    exit 1
fi
echo ""

echo "=========================================="
echo "✅ Pipeline Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - model_eurusd_best.zip (trained model)"
echo "  - equity_curve.png (training equity)"
echo "  - test_equity_curve.png (test equity)"
echo "  - trading_analysis_advanced.png (8-panel analysis)"
echo "  - trade_history_output.csv (trade details)"
echo ""
