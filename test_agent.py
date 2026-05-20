import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def run_one_episode(model, vec_env, deterministic=True):
    obs = vec_env.reset()
    equity_curve = []
    closed_trades = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = vec_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        equity_curve.append(vec_env.get_attr("equity_usd")[0])

        trade_info = vec_env.get_attr("last_trade_info")[0]
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)

        if done:
            break

    return equity_curve, closed_trades


def main():
    # Using USD/INR historical data (2000-present)
    file_path = "data/USD_INR Historical Data.csv"
    df_full, feature_cols = load_and_preprocess_data(file_path)
    
    # Use test data from 2016-2022
    test_df_tmp, _ = load_and_preprocess_data(file_path, start_date="2016-01-01", end_date="2022-12-31")
    
    # If date-based split doesn't work, fallback to 80/20
    if len(test_df_tmp) == 0:
        split_idx = int(len(df_full) * 0.8)
        test_df = df_full.iloc[split_idx:].copy()
        print("Warning: Date-based test split not possible. Using 20% tail split instead.")
    else:
        test_df = test_df_tmp

    print(f"Test data: {len(test_df)} bars")
    print(f"Date range: {test_df.index.min()} to {test_df.index.max()}")

    # Must match training params
    SL_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    TP_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    WIN = 30

    test_env = ForexTradingEnv(
        df=test_df,
        window_size=WIN,
        sl_options=SL_OPTS,
        tp_options=TP_OPTS,
        spread_pips=1.0,
        commission_pips=0.0,
        max_slippage_pips=0.2,
        random_start=False,
        episode_max_steps=None,
        feature_columns=feature_cols,
        hold_reward_weight=0.00,
        open_penalty_pips=0.0,
        time_penalty_pips=0.0,
        unrealized_delta_weight=0.0,
        # Advanced risk management (must match training)
        capital_risk_pct=0.02,
        min_risk_reward_ratio=3.0,
        risk_reward_bonus=0.1,
    )

    vec_test_env = DummyVecEnv([lambda: test_env])

    # Load best model
    print("\nLoading best model: model_eurusd_best...")
    model = PPO.load("model_eurusd_best", env=vec_test_env)

    equity_curve, closed_trades = run_one_episode(model, vec_test_env, deterministic=True)

    # Save trades and print statistics
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        out_csv = "trade_history_output.csv"
        trades_df.to_csv(out_csv, index=False)
        print(f"\n✓ Trade history saved to {out_csv}")
        
        # Print statistics
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        
        initial_equity = 10000.0
        final_equity = equity_curve[-1] if equity_curve else initial_equity
        total_pnl = final_equity - initial_equity
        roi = (total_pnl / initial_equity) * 100
        
        avg_win = trades_df[trades_df['net_pips'] > 0]['net_pips'].mean() if len(trades_df[trades_df['net_pips'] > 0]) > 0 else 0
        avg_loss = trades_df[trades_df['net_pips'] <= 0]['net_pips'].mean() if len(trades_df[trades_df['net_pips'] <= 0]) > 0 else 0
        
        print(f"Initial Equity: ${initial_equity:>10,.2f}")
        print(f"Final Equity:   ${final_equity:>10,.2f}")
        print(f"Total P&L:      ${total_pnl:>10,.2f}")
        print(f"ROI:            {roi:>10.2f}%")
        print(f"\nTotal Trades:   {len(trades_df):>10}")
        print(f"Winning Trades: {len(trades_df[trades_df['net_pips'] > 0]):>10}")
        print(f"Losing Trades:  {len(trades_df[trades_df['net_pips'] <= 0]):>10}")
        print(f"Win Rate:       {(len(trades_df[trades_df['net_pips'] > 0]) / len(trades_df) * 100):>9.2f}%")
        print(f"\nAvg Winner:     {avg_win:>10.2f} pips")
        print(f"Avg Loser:      {avg_loss:>10.2f} pips")
        print(f"Risk:Reward:    1:{abs(avg_win/avg_loss) if avg_loss != 0 else 0:>9.2f}")
        print("="*60)
    else:
        print("No closed trades recorded.")

    # Plot equity curve
    if equity_curve:
        final_equity = equity_curve[-1]
        initial = 10000.0
        roi = ((final_equity - initial) / initial) * 100
        
        plt.figure(figsize=(14, 7))
        plt.plot(equity_curve, label="Equity (Test)", linewidth=2, color='#2E86AB')
        plt.axhline(y=initial, color='red', linestyle='--', linewidth=1, label='Initial Capital')
        plt.title(f"Test Equity Curve (Model with 25+ Indicators + Risk Mgmt)\nFinal: ${final_equity:,.0f} | ROI: {roi:.1f}%", 
                  fontsize=14, fontweight='bold')
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Equity ($)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("test_equity_curve.png", dpi=150)
        print("\n✓ Test equity curve saved: test_equity_curve.png")
        plt.close()


if __name__ == "__main__":
    main()
