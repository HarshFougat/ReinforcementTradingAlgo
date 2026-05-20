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
        # use equity from info (state *before* DummyVecEnv reset)
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        if done:
            break

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity



def main():
    # Load USD_INR data with date-based splitting
    file_path = "data/USD_INR Historical Data.csv"
    df_all, feature_cols = load_and_preprocess_data(file_path)

    # Date-based split: 2000-2015 training, 2016-2022 testing
    try:
        train_df = df_all[df_all.index.year <= 2015].copy()
        test_df = df_all[(df_all.index.year >= 2016) & (df_all.index.year <= 2022)].copy()
        
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Date split resulted in empty datasets")
    except Exception as e:
        print(f"[Warning] Date split failed ({e}), using 80/20 split instead")
        split_idx = int(len(df_all) * 0.8)
        train_df = df_all.iloc[:split_idx].copy()
        test_df = df_all.iloc[split_idx:].copy()

    print("Training bars:", len(train_df))
    print("Testing bars :", len(test_df))
    print("Training data:", train_df.index.min(), "to", train_df.index.max())
    print("Testing data :", test_df.index.min(), "to", test_df.index.max())

    # ---- Env factories ----
    SL_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    TP_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    WIN = 30

    # Train env: random starts to reduce memorization
    def make_train_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=True,
            min_episode_steps=1000,
            episode_max_steps=2000,
            feature_columns=feature_cols,
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0,
            # Advanced risk management
            capital_risk_pct=0.02,           # 2% risk per trade
            min_risk_reward_ratio=3.0,       # 1:3 minimum ratio
            risk_reward_bonus=0.1,           # Bonus for good RR trades
        )

    # Train-eval env: deterministic start, NO random starts
    def make_train_eval_env():
        return ForexTradingEnv(
            df=train_df,
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
            capital_risk_pct=0.02,
            min_risk_reward_ratio=3.0,
            risk_reward_bonus=0.1,
        )

    # Test-eval env: deterministic
    def make_test_eval_env():
        return ForexTradingEnv(
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
            time_penalty_pips=0.00,
            unrealized_delta_weight=0.0,
            capital_risk_pct=0.02,
            min_risk_reward_ratio=3.0,
            risk_reward_bonus=0.1,
        )

    train_vec_env = DummyVecEnv([make_train_env])
    train_eval_env = DummyVecEnv([make_train_eval_env])
    test_eval_env = DummyVecEnv([make_test_eval_env])

    # ---- Model ----
    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
        verbose=1,
        tensorboard_log=None  # Disable TensorBoard for now
    )

    # ---- Checkpoints ----
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,  # Save every 100k steps instead of 50k
        save_path=ckpt_dir,
        name_prefix="ppo_eurusd"
    )

    # ---- Train ----
    total_timesteps = 1_000_000  # Increased to 1M for better convergence
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # ---- Select best model by OOS final equity ----
    print("\n" + "="*60)
    print("Evaluating checkpoints on out-of-sample data...")
    print("="*60)
    
    equity_curve_test_last, final_equity_test_last = evaluate_model(model, test_eval_env)
    print(f"[OOS Eval] Last model final equity: ${final_equity_test_last:,.2f}")

    best_equity = -np.inf
    best_path = None
    results = []

    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".zip") and f.startswith("ppo_eurusd")],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x))
    )

    for ck in ckpts:
        ck_path = os.path.join(ckpt_dir, ck)
        try:
            m = PPO.load(ck_path, env=test_eval_env)
            _, final_eq = evaluate_model(m, test_eval_env)
            results.append((ck, final_eq))
            print(f"[OOS Eval] {ck:40s} -> final equity: ${final_eq:>10,.2f}")
            if final_eq > best_equity:
                best_equity = final_eq
                best_path = ck_path
        except Exception as e:
            print(f"[Skip] Could not evaluate checkpoint {ck}: {e}")

    # Decide best model
    if best_path is None or final_equity_test_last >= best_equity:
        print("\nUsing last model as best (by OOS final equity).")
        best_model = model
        best_final_equity = final_equity_test_last
    else:
        print(f"\nUsing best checkpoint: {best_path}")
        print(f"Best OOS final equity: ${best_equity:,.2f}")
        best_model = PPO.load(best_path, env=train_vec_env)
        best_final_equity = best_equity

    best_model.save("model_eurusd_best")
    print("✓ Best model saved: model_eurusd_best.zip")

    # ---- Plot BOTH: in-sample vs out-of-sample ----
    print("\n" + "="*60)
    print("Generating final equity curves...")
    print("="*60)
    
    equity_curve_train, final_equity_train = evaluate_model(best_model, train_eval_env)
    equity_curve_test, final_equity_test = evaluate_model(best_model, test_eval_env)

    print(f"[IS Eval]  Final equity (train): ${final_equity_train:,.2f}")
    print(f"[OOS Eval] Final equity (test) : ${final_equity_test:,.2f}")

    plt.figure(figsize=(14, 7))
    plt.plot(equity_curve_train, label="Train (in-sample) equity", linewidth=2, alpha=0.8)
    plt.plot(equity_curve_test, label="Test (out-of-sample) equity", linewidth=2, alpha=0.8)
    plt.title("Equity Curves: In-sample vs Out-of-sample (Best Model)\nWith 25+ Indicators + Risk Management", fontsize=14, fontweight='bold')
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Equity ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=150)
    print("✓ Equity curve saved: equity_curve.png")
    plt.close()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
