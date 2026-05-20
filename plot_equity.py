#!/usr/bin/env python3
"""Visualize trading results using matplotlib"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Read CSV and extract data
equity_data = []
pips_data = []

try:
    with open('trade_history_output.csv', 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 10:
                equity_data.append(float(parts[9]))  # equity_usd column
                pips_data.append(float(parts[8]))     # net_pips column
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

if not equity_data:
    print("No data found!")
    exit(1)

# Create visualization
fig = plt.figure(figsize=(16, 10))

# Subplot 1: Equity Curve
ax1 = plt.subplot(2, 1, 1)
ax1.plot(equity_data, linewidth=2.5, color='#1f77b4', label='Account Equity', marker='o', markersize=2, alpha=0.8)
ax1.fill_between(range(len(equity_data)), equity_data, alpha=0.2, color='#1f77b4')
ax1.set_title('Reinforcement Learning Trading Bot - Equity Curve\nUSD/INR Test Data (2016-2022)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Trade Number', fontsize=12)
ax1.set_ylabel('Equity ($)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11, loc='upper left')

# Add statistics box
initial = equity_data[0]
final = equity_data[-1]
profit = final - initial
roi = (profit / initial) * 100
max_equity = max(equity_data)
min_equity = min(equity_data)

stats = f'Start: ${initial:,.0f}\nEnd: ${final:,.0f}\nProfit: ${profit:,.0f}\nROI: {roi:.1f}%'
ax1.text(0.98, 0.05, stats, transform=ax1.transAxes, fontsize=11, verticalalignment='bottom', 
         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=2))

# Subplot 2: Per-Trade Results
ax2 = plt.subplot(2, 1, 2)
colors = ['#2ca02c' if p > 0 else '#d62728' for p in pips_data]
ax2.bar(range(len(pips_data)), pips_data, color=colors, alpha=0.8, width=0.9, edgecolor='black', linewidth=0.5)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_title('Net Pips Per Trade', fontsize=16, fontweight='bold')
ax2.set_xlabel('Trade Number', fontsize=12)
ax2.set_ylabel('Net Pips', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Trade statistics
wins = sum(1 for p in pips_data if p > 0)
losses = sum(1 for p in pips_data if p < 0)
total = len(pips_data)
win_rate = (wins/total)*100 if total > 0 else 0
avg_pips = sum(pips_data) / total if total > 0 else 0

trade_stats = f'Total: {total} | Wins: {wins} ({win_rate:.1f}%) | Losses: {losses} | Avg Pips: {avg_pips:.2f}'
ax2.text(0.5, 0.95, trade_stats, transform=ax2.transAxes, fontsize=11, verticalalignment='top', 
         horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='black', linewidth=2))

plt.tight_layout()
plt.savefig('trading_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Chart saved: trading_analysis.png\n")

# Print statistics
print("="*70)
print(" "*15 + "TRADING PERFORMANCE SUMMARY")
print("="*70)
print(f"Initial Capital:      ${initial:>15,.2f}")
print(f"Final Equity:         ${final:>15,.2f}")
print(f"Total Profit/Loss:    ${profit:>15,.2f}")
print(f"Return on Investment: {roi:>15.2f}%")
print(f"Maximum Equity:       ${max_equity:>15,.2f}")
print(f"Minimum Equity:       ${min_equity:>15,.2f}")
print("-"*70)
print(f"Total Trades:         {total:>15}")
print(f"Winning Trades:       {wins:>15} ({win_rate:.1f}%)")
print(f"Losing Trades:        {losses:>15} ({100-win_rate:.1f}%)")
print(f"Average Pips/Trade:   {avg_pips:>15.4f}")
print(f"Total Pips Gained:    {sum(pips_data):>15,.2f}")
print("="*70)
print("\n📊 Visualization complete! Open 'trading_analysis.png' to view the chart.")
