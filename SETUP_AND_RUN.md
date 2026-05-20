# USD/INR Advanced Trading Bot - Setup & Execution Guide

**Status:** ✅ Implementation Complete - Ready for Training

## Overview

Your reinforcement learning trading bot has been upgraded with:

- **25+ Technical Indicators**: RSI (7,14,21), MACD, Stochastic, Bollinger Bands, ATR, Vortex, EMA/SMA, OBV, CCI, ROC
- **Market Sentiment Analysis**: Bull/Bear detection based on multiple indicator convergence
- **Advanced Risk Management**: 2% capital risk per trade, 1:3 minimum risk:reward ratio, dynamic position sizing
- **Enhanced Reward Shaping**: Bonus rewards for trades with good risk:reward ratios
- **Professional Visualizations**: 8-panel comprehensive analysis charts
- **Increased Training**: 1M timesteps (vs 600k) for better convergence

## Files Modified

| File | Changes |
|------|---------|
| `indicators.py` | Added 25 new indicators + market sentiment calculation |
| `trading_env.py` | Added risk management, position sizing, trade statistics |
| `train_agent.py` | Updated to use USD_INR data (2000-2015), 1M timesteps |
| `test_agent.py` | Enhanced statistics and visualization |
| `advanced_analysis.py` | NEW - 8-panel professional analysis tool |
| `run_full_pipeline.py` | NEW - Orchestration script |

## Installation Instructions

### Option 1: Automated Installation (Recommended)

```bash
# In terminal, run this single command:
python3 -m pip install --break-system-packages pandas numpy stable-baselines3 gymnasium pandas-ta matplotlib seaborn
```

Or using apt (if available):
```bash
sudo apt-get update
sudo apt-get install -y python3-pandas python3-numpy python3-matplotlib python3-pip
python3 -m pip install --break-system-packages stable-baselines3 gymnasium pandas-ta seaborn
```

### Option 2: Manual Steps

1. **Install system packages:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pandas python3-numpy python3-matplotlib
   ```

2. **Install pip packages:**
   ```bash
   pip install --break-system-packages stable-baselines3 gymnasium pandas-ta seaborn
   ```

## Execution Instructions

### Full Pipeline (Recommended)

Trains model, tests it, and generates analysis in one go:

```bash
python run_full_pipeline.py
```

**Expected Runtime:** ~60-90 minutes
- Training: ~45-60 min (1M timesteps)
- Testing: ~5-10 min
- Analysis: ~2-5 min

### Step-by-Step Execution

#### Step 1: Train the Model

```bash
python train_agent.py
```

**Output:**
- `model_eurusd_best.zip` - Best model checkpoint
- `equity_curve.png` - Training equity curve
- `checkpoints/` - All 10 model checkpoints (100k step intervals)

**Duration:** ~45-60 minutes

####Step 2: Test the Model

```bash
python test_agent.py
```

**Output:**
- `test_equity_curve.png` - Test equity curve  
- `trade_history_output.csv` - Detailed trade export
- Console: Performance statistics

**Duration:** ~5-10 minutes

#### Step 3: Generate Analysis

```bash
python advanced_analysis.py
```

**Output:**
- `trading_analysis_advanced.png` - 8-panel professional analysis chart
- Console: Comprehensive performance report

**Duration:** ~2-5 minutes

## Model Configuration

### Training Parameters
- **Data**: USD_INR Historical Data (April 2000 - present)
- **Training Period**: 2000-2015 (4,085 bars)
- **Test Period**: 2016-2022 (1,618 bars)
- **Total Timesteps**: 1,000,000
- **Checkpoint Frequency**: Every 100,000 steps
- **Algorithm**: PPO (Proximal Policy Optimization)

### Risk Management Settings
- **Capital per Trade**: 2% of account equity
- **Minimum Risk:Reward Ratio**: 1:3
- **SL/TP Options**: [5, 10, 15, 25, 30, 60, 90, 120] pips
- **Initial Capital**: $10,000

### Technical Indicators (25+)

**Momentum (9 indicators):**
- RSI-7, RSI-14, RSI-21
- MACD Line, MACD Signal, MACD Histogram
- Rate of Change (ROC-12)
- Stochastic K, Stochastic D

**Volatility (7 indicators):**
- ATR-14, ATR%
- Bollinger Bands Upper/Lower/Width
- Bollinger Bands %B

**Trend (8 indicators):**
- EMA-9 Slope, EMA-21 Slope
- Price vs EMA-9, Price vs EMA-21, Price vs SMA-200
- Vortex VI+, Vortex VI-

**Volume & Additional (3 indicators):**
- OBV Trend
- CCI-20

**Market Sentiment (1 indicator):**
- Composite Bull/Bear Score (-1 to +1)

## Expected Results (Baseline from Previous Model)

Your previous model achieved:
- **Final Equity**: $187,749 (from $10,000)
- **ROI**: 1,877%
- **Win Rate**: 13.8%
- **Risk:Reward**: 1:19.8 (excellent)
- **Trades**: 1,618 on test data

**Goal:** Beat or match these results with enhanced indicators and risk management

## Troubleshooting

### ImportError: No module named 'pandas'

**Solution:**
```bash
pip install --break-system-packages pandas numpy stable-baselines3 gymnasium pandas-ta matplotlib seaborn
```

### ModuleNotFoundError: gymnasium

**Solution:**
```bash
pip install --break-system-packages gymnasium
```

### Memory Error during training

**Solution:** The model uses ~2-4GB RAM. If you get memory errors:
1. Close other applications
2. Reduce `min_episode_steps` in `train_agent.py` from 1000 to 500
3. Or reduce `total_timesteps` from 1,000,000 to 500,000

### Slow training (< 100 steps/sec)

This is normal for complex RL training. Expected rates:
- CPU: 50-200 steps/sec
- GPU: 500-2000 steps/sec

At 100 steps/sec, 1M timesteps = ~2.8 hours

## File Structure

```
/workspaces/ReinforcementTradingAlgo/
├── indicators.py              # 25+ technical indicators + market sentiment
├── trading_env.py            # RL environment with risk management
├── train_agent.py            # Training script (1M steps)
├── test_agent.py             # Evaluation script
├── advanced_analysis.py       # 8-panel visualization analysis
├── run_full_pipeline.py       # Orchestration script
├── data/
│   └── USD_INR Historical Data.csv
├── checkpoints/              # Saved model checkpoints
├── model_eurusd_best.zip     # Best trained model
├── equity_curve.png          # Training equity curve
├── test_equity_curve.png     # Test equity curve
├── trading_analysis.png      # Original analysis (4-panel)
├── trading_analysis_advanced.png  # New analysis (8-panel)
└── trade_history_output.csv  # Detailed trade export
```

## Key Enhancements Made

### 1. Indicator Expansion
- **Before**: 8 basic indicators
- **After**: 25+ professional-grade indicators
- **Benefit**: Better market understanding and signal quality

### 2. Risk Management
- Added capital risk limits (2% per trade)
- Enforced 1:3 minimum risk:reward ratios
- Position sizing based on volatility
- **Benefit**: Sustainable trading with defined risk

### 3. Reward Shaping
- Bonus for trades with good risk:reward ratios
- Penalty for overtrading
- Time cost in trades
- **Benefit**: Agent learns optimal trade selection

### 4. Data & Training
- Date-based train/test split (not random)
- Increased training from 600k → 1M timesteps
- Checkpoint every 100k steps (vs 50k)
- **Benefit**: More stable, realistic evaluation

### 5. Analysis Tools
- 8-panel comprehensive visualization
- Detailed trade statistics
- Risk analysis and drawdown tracking
- **Benefit**: Professional-grade performance reporting

## Command Reference

```bash
# Full automated pipeline
python run_full_pipeline.py

# Individual steps
python train_agent.py           # Train model (45-60 min)
python test_agent.py            # Test model (5-10 min)
python advanced_analysis.py      # Analyze results (2-5 min)

# Quick test (without full training)
python test_agent.py            # Uses pre-trained model
python advanced_analysis.py      # Uses test trades
```

## Next Steps

1. **Install dependencies** (if not already done)
2. **Run full pipeline**: `python run_full_pipeline.py`
3. **Monitor progress**: Check console output for step-by-step progress
4. **Review results**: Open generated PNG files and review CSV trade data
5. **Optimize further** (optional): Adjust hyperparameters and retrain

## Support

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Verify all dependencies are installed: `pip list | grep -E 'pandas|numpy|stable|gymnasium'`
3. Ensure data file exists: `ls -lh data/USD_INR\ Historical\ Data.csv`
4. Check working directory: All scripts must run from `/workspaces/ReinforcementTradingAlgo/`

---

**Status**: ✅ Ready to train!

Everything is set up. You just need to install dependencies and run `python run_full_pipeline.py` to start training your enhanced trading bot.
