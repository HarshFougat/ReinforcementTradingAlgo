#!/usr/bin/env python3
"""
Advanced Trading Analysis
Comprehensive visualization and statistics for RL trading bot results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'


def load_trade_data(csv_path):
    """Load trade history from CSV."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Run test_agent.py first.")
        return None


def calculate_equity_curve(trades_df):
    """Calculate equity curve from trades."""
    if trades_df is None or len(trades_df) == 0:
        return None
    
    initial_equity = 10000.0
    equity = [initial_equity]
    
    for _, trade in trades_df.iterrows():
        if trade['event'] == 'CLOSE':
            net_pips = trade['net_pips']
            # Convert pips to USD (1 pip ≈ 10 per 100k units)
            pip_value = 0.0001
            lot_size = 100000.0
            usd_per_pip = pip_value * lot_size
            pnl = net_pips * usd_per_pip
            equity.append(equity[-1] + pnl)
    
    return equity


def create_comprehensive_analysis():
    """Generate comprehensive trading analysis visualization."""
    trades_df = load_trade_data("trade_history_output.csv")
    
    if trades_df is None:
        return
    
    # Filter for closed trades only
    closed_trades = trades_df[trades_df['event'] == 'CLOSE'].copy()
    
    if len(closed_trades) == 0:
        print("No closed trades found in trade history.")
        return
    
    # Calculate metrics
    initial_equity = 10000.0
    equity_curve = calculate_equity_curve(closed_trades)
    final_equity = equity_curve[-1] if equity_curve else initial_equity
    total_pnl = final_equity - initial_equity
    roi = (total_pnl / initial_equity) * 100
    
    winning_trades = closed_trades[closed_trades['net_pips'] > 0]
    losing_trades = closed_trades[closed_trades['net_pips'] <= 0]
    
    num_trades = len(closed_trades)
    num_wins = len(winning_trades)
    num_losses = len(losing_trades)
    win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0
    
    avg_winner = winning_trades['net_pips'].mean() if len(winning_trades) > 0 else 0
    avg_loser = losing_trades['net_pips'].mean() if len(losing_trades) > 0 else 0
    risk_reward = abs(avg_winner / avg_loser) if avg_loser != 0 else 0
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Main equity curve with drawdown
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.fill_between(range(len(equity_curve)), equity_curve, initial_equity, 
                     where=[e >= initial_equity for e in equity_curve], 
                     interpolate=True, alpha=0.3, color='green', label='Profit Zone')
    ax1.fill_between(range(len(equity_curve)), equity_curve, initial_equity, 
                     where=[e < initial_equity for e in equity_curve], 
                     interpolate=True, alpha=0.3, color='red', label='Drawdown Zone')
    ax1.plot(equity_curve, label='Equity Curve', linewidth=2.5, color='#2E86AB')
    ax1.axhline(y=initial_equity, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_title('Equity Curve with Drawdown Analysis', fontsize=13, fontweight='bold')
    ax1.set_ylabel('USD ($)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Key metrics box
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    metrics_text = f"""
    PERFORMANCE METRICS
    {'─'*28}
    Initial Capital: ${initial_equity:>10,.0f}
    Final Capital:   ${final_equity:>10,.0f}
    Total P&L:       ${total_pnl:>10,.0f}
    ROI:             {roi:>12.2f}%
    
    TRADE STATISTICS
    {'─'*28}
    Total Trades:    {num_trades:>11}
    Winning Trades:  {num_wins:>11}
    Losing Trades:   {num_losses:>11}
    Win Rate:        {win_rate:>12.2f}%
    
    RISK/REWARD
    {'─'*28}
    Avg Winner:      {avg_winner:>10.2f} pips
    Avg Loser:       {avg_loser:>10.2f} pips
    Risk:Reward:     1:{risk_reward:>9.2f}
    """
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8))
    
    # 3. Win/Loss distribution
    ax3 = fig.add_subplot(gs[1, 0])
    bins = 30
    ax3.hist(winning_trades['net_pips'], bins=bins, alpha=0.6, label='Wins', color='green', edgecolor='darkgreen')
    ax3.hist(losing_trades['net_pips'], bins=bins, alpha=0.6, label='Losses', color='red', edgecolor='darkred')
    ax3.axvline(x=avg_winner, color='darkgreen', linestyle='--', linewidth=2, label=f'Avg Win: {avg_winner:.1f}')
    ax3.axvline(x=avg_loser, color='darkred', linestyle='--', linewidth=2, label=f'Avg Loss: {avg_loser:.1f}')
    ax3.set_title('Win/Loss Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Net Pips', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Cumulative P&L by trade
    ax4 = fig.add_subplot(gs[1, 1])
    closed_trades_sorted = closed_trades.reset_index(drop=True)
    cumulative_pnl = closed_trades_sorted['net_pips'].cumsum()
    colors = ['green' if x > 0 else 'red' for x in closed_trades_sorted['net_pips']]
    ax4.bar(range(len(closed_trades_sorted)), cumulative_pnl, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_title('Cumulative Profit/Loss', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Trade #', fontsize=10)
    ax4.set_ylabel('Cumulative Pips', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Win Rate Pie Chart
    ax5 = fig.add_subplot(gs[1, 2])
    sizes = [num_wins, num_losses]
    colors_pie = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)
    ax5.pie(sizes, explode=explode, labels=['Wins', 'Losses'], colors=colors_pie, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 11})
    ax5.set_title(f'Win Rate: {win_rate:.1f}%', fontsize=12, fontweight='bold')
    
    # 6. Trade duration analysis
    ax6 = fig.add_subplot(gs[2, 0])
    closed_trades_sorted = closed_trades.reset_index(drop=True)
    if 'time_in_trade' in closed_trades_sorted.columns:
        ax6.hist(closed_trades_sorted['time_in_trade'], bins=20, color='#3498db', 
                edgecolor='black', alpha=0.7, linewidth=1)
        ax6.set_title('Trade Duration Distribution', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Bars Held', fontsize=10)
        ax6.set_ylabel('Frequency', fontsize=10)
        ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Entry reasons breakdown
    ax7 = fig.add_subplot(gs[2, 1])
    if 'reason' in closed_trades.columns:
        reasons = closed_trades['reason'].value_counts()
        colors_reasons = plt.cm.Set3(np.linspace(0, 1, len(reasons)))
        ax7.barh(range(len(reasons)), reasons.values, color=colors_reasons, edgecolor='black', linewidth=1)
        ax7.set_yticks(range(len(reasons)))
        ax7.set_yticklabels(reasons.index, fontsize=10)
        ax7.set_title('Trade Close Reasons', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Count', fontsize=10)
        ax7.grid(True, alpha=0.3, axis='x')
    
    # 8. Monthly returns heatmap
    ax8 = fig.add_subplot(gs[2, 2])
    if 'step' in closed_trades.columns and len(closed_trades) > 1:
        closed_trades_sorted = closed_trades.reset_index(drop=True)
        closed_trades_sorted['bin'] = pd.cut(closed_trades_sorted.index, bins=12, labels=False)
        monthly_pnl = closed_trades_sorted.groupby('bin')['net_pips'].sum()
        colors_monthly = ['green' if x > 0 else 'red' for x in monthly_pnl.values]
        ax8.bar(range(len(monthly_pnl)), monthly_pnl.values, color=colors_monthly, alpha=0.7, edgecolor='black')
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax8.set_title('P&L by Trading Period', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Period', fontsize=10)
        ax8.set_ylabel('Pips', fontsize=10)
        ax8.grid(True, alpha=0.3, axis='y')
    
    # Main title
    fig.suptitle('Advanced Trading Analysis - USD/INR Model with 25+ Indicators & Risk Management', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('trading_analysis_advanced.png', dpi=150, bbox_inches='tight')
    print("\n✓ Advanced analysis saved: trading_analysis_advanced.png")
    plt.close()
    
    # Print detailed statistics
    print("\n" + "="*70)
    print(" "*15 + "COMPREHENSIVE TRADING ANALYSIS")
    print("="*70)
    print(f"\nCapital Management:")
    print(f"  Initial Capital:     ${initial_equity:>15,.2f}")
    print(f"  Final Capital:       ${final_equity:>15,.2f}")
    print(f"  Net Profit/Loss:     ${total_pnl:>15,.2f}")
    print(f"  Return on Investment: {roi:>13.2f}%")
    
    print(f"\nTrade Statistics:")
    print(f"  Total Trades:        {num_trades:>15}")
    print(f"  Winning Trades:      {num_wins:>15} ({win_rate:.2f}%)")
    print(f"  Losing Trades:       {num_losses:>15} ({100-win_rate:.2f}%)")
    
    print(f"\nProfitability Metrics:")
    print(f"  Average Winner:      {avg_winner:>14.2f} pips")
    print(f"  Average Loser:       {avg_loser:>14.2f} pips")
    print(f"  Risk:Reward Ratio:   1:{risk_reward:>13.2f}")
    print(f"  Total Win Pips:      {winning_trades['net_pips'].sum():>14.2f}")
    print(f"  Total Loss Pips:     {losing_trades['net_pips'].sum():>14.2f}")
    
    if 'time_in_trade' in closed_trades.columns:
        avg_duration = closed_trades['time_in_trade'].mean()
        max_duration = closed_trades['time_in_trade'].max()
        print(f"\nTrade Duration:")
        print(f"  Average Duration:    {avg_duration:>14.1f} bars")
        print(f"  Maximum Duration:    {max_duration:>14.0f} bars")
    
    # Drawdown analysis
    running_max = [max(equity_curve[:i+1]) for i in range(len(equity_curve))]
    drawdown = [equity_curve[i] - running_max[i] for i in range(len(equity_curve))]
    max_drawdown = min(drawdown)
    max_drawdown_pct = (max_drawdown / initial_equity) * 100
    
    print(f"\nRisk Analysis:")
    print(f"  Maximum Drawdown:    ${max_drawdown:>14,.2f} ({max_drawdown_pct:.2f}%)")
    print("="*70 + "\n")


if __name__ == "__main__":
    create_comprehensive_analysis()
