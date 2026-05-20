import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def evaluate_model(model: PPO, eval_env: DummyVecEnv, deterministic: bool = True):
    obs = eval_env.reset()
    equity_curve = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = eval_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        if done:
            break

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity


def main():
    # Load BTC data
    file_path = "data/BTCUSDT_1h.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please download BTC data first.")
        return

    df_all, feature_cols = load_and_preprocess_data(file_path)

    # Date-based split: 2017-2021 training, 2022-2023 testing
    train_df = df_all[df_all.index.year <= 2021].copy()
    test_df = df_all[df_all.index.year >= 2022].copy()

    print(f"Training data: {len(train_df)} rows from {train_df.index.min()} to {train_df.index.max()}")
    print(f"Testing data: {len(test_df)} rows from {test_df.index.min()} to {test_df.index.max()}")

    # Create training environment
    train_env = DummyVecEnv([lambda: ForexTradingEnv(train_df, feature_cols)])
    eval_env = DummyVecEnv([lambda: ForexTradingEnv(test_df, feature_cols)])

    # PPO hyperparameters (tuned for forex/crypto)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/"
    )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # Save every 100k steps
        save_path="./models/",
        name_prefix="ppo_btcusd",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Train the model
    total_timesteps = 1000000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save final model
    model.save("model_btcusd_final.zip")
    print("Model saved as model_btcusd_final.zip")

    # Evaluate on test set
    equity_curve, final_equity = evaluate_model(model, eval_env)
    print(f"Test final equity: ${final_equity:.2f}")

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title("BTCUSD Test Equity Curve")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.savefig("btc_equity_curve.png")
    plt.show()

    print("Training complete. Check btc_equity_curve.png for results.")


if __name__ == "__main__":
    main()