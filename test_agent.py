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

    # Must match training params
    SL_OPTS = [10, 15, 25]
    TP_OPTS = [10, 15, 25]
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
            hold_reward_weight=0.005,
            open_penalty_pips=0.5,      # half a pip per open
            time_penalty_pips=0.02,     # 0.02 pips per bar in trade
            unrealized_delta_weight=0.0
    )

    vec_test_env = DummyVecEnv([lambda: test_env])

    # Load best model
    model = PPO.load("model_eurusd_best", env=vec_test_env)

    equity_curve, closed_trades = run_one_episode(model, vec_test_env, deterministic=True)

    # Save trades
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        out_csv = "trade_history_output.csv"
        trades_df.to_csv(out_csv, index=False)
        print(f"Closed trade history saved to {out_csv}")
    else:
        print("No closed trades recorded.")

    # Plot equity
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label="Equity (Test)")
    plt.title("Equity Curve - Evaluation")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
