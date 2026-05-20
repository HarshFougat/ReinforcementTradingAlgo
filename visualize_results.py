import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load trade history
trades_df = pd.read_csv("trade_history_output.csv")

# Extract equity curve
equity_curve = trades_df["equity_usd"].values

# Create figure with better styling
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Equity Curve
ax1.plot(equity_curve, linewidth=2, color='#2E86AB', label='Equity')
ax1.fill_between(range(len(equity_curve)), equity_curve, alpha=0.3, color='#2E86AB')
ax1.set_title('Trading Bot Equity Curve (USD/INR Test Data: 2016-2022)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Trade Number', fontsize=12)
ax1.set_ylabel('Equity ($)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Add stats
initial_equity = equity_curve[0]
final_equity = equity_curve[-1]
max_equity = np.max(equity_curve)
min_equity = np.min(equity_curve)
total_return = final_equity - initial_equity
roi = (total_return / initial_equity) * 100

stats_text = f'Initial: ${initial_equity:,.0f}\nFinal: ${final_equity:,.0f}\nMax: ${max_equity:,.0f}\nMin: ${min_equity:,.0f}\nROI: {roi:.1f}%'
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 2: Net Pips per Trade
net_pips = trades_df["net_pips"].values
colors = ['green' if x > 0 else 'red' for x in net_pips]
ax2.bar(range(len(net_pips)), net_pips, color=colors, alpha=0.7, width=0.8)
ax2.set_title('Net Pips Per Trade', fontsize=14, fontweight='bold')
ax2.set_xlabel('Trade Number', fontsize=12)
ax2.set_ylabel('Net Pips', fontsize=12)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(True, alpha=0.3, axis='y')

# Statistics on plot
winning_trades = np.sum(net_pips > 0)
losing_trades = np.sum(net_pips < 0)
avg_pips = np.mean(net_pips)
win_rate = (winning_trades / len(net_pips)) * 100

stats_text2 = f'Total Trades: {len(net_pips)}\nWinning: {winning_trades}\nLosing: {losing_trades}\nWin Rate: {win_rate:.1f}%\nAvg Pips: {avg_pips:.2f}'
ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig("trading_analysis.png", dpi=150, bbox_inches="tight")
print("✅ Trading analysis plot saved to: trading_analysis.png")

# Also display it
plt.show()

# Print summary statistics
print("\n" + "="*50)
print("TRADING PERFORMANCE SUMMARY")
print("="*50)
print(f"Initial Capital:  ${initial_equity:,.2f}")
print(f"Final Equity:     ${final_equity:,.2f}")
print(f"Total Profit:     ${total_return:,.2f}")
print(f"ROI:              {roi:.2f}%")
print(f"\nTotal Trades:     {len(net_pips)}")
print(f"Winning Trades:   {winning_trades} ({win_rate:.1f}%)")
print(f"Losing Trades:    {losing_trades} ({100-win_rate:.1f}%)")
print(f"Average Pips:     {avg_pips:.2f}")
print(f"Total Pips Gain:  {np.sum(net_pips):.2f}")
print("="*50)
