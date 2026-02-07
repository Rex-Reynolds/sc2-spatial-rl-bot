#!/usr/bin/env python3
"""
Train an RL agent to play StarCraft II using PPO.

Usage:
    python rl/train.py --opponent IdleBot --episodes 1000
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from rl.env import make_env


def main():
    parser = argparse.ArgumentParser(description="Train SC2 RL agent")
    parser.add_argument(
        "--opponent",
        default="IdleBot",
        choices=["IdleBot", "RushBot", "DefenseBot"],
        help="Opponent bot to train against",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes (default: 1000)",
    )
    parser.add_argument(
        "--model-name",
        default="sc2_ppo",
        help="Model name for saving",
    )
    parser.add_argument(
        "--load-model",
        default=None,
        help="Path to existing model to continue training",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("STARCRAFT II RL TRAINING")
    print("=" * 70)
    print(f"Opponent: {args.opponent}")
    print(f"Episodes: {args.episodes}")
    print(f"Model: {args.model_name}")
    print("=" * 70)
    print()

    # Create environment
    env = make_env(opponent=args.opponent, realtime=False)
    env = Monitor(env)  # Wrap for logging

    # Create or load model
    if args.load_model:
        print(f"Loading existing model: {args.load_model}")
        model = PPO.load(args.load_model, env=env)
    else:
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            tensorboard_log=f"./rl/logs/{args.model_name}",
        )

    # Set the environment's policy to use the model
    def policy_fn(obs):
        """Policy function that queries the model."""
        return model.predict(obs, deterministic=False)

    env.unwrapped.policy = policy_fn

    # Callbacks for saving
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./rl/models/{args.model_name}",
        name_prefix="sc2_agent",
    )

    # Train
    print(f"\nStarting training for {args.episodes} episodes...")
    print("This will take a while. Monitor progress with TensorBoard:")
    print(f"  tensorboard --logdir=./rl/logs/{args.model_name}")
    print()

    total_timesteps = args.episodes * 1000  # ~1000 steps per episode
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # Save final model
    model_path = f"./rl/models/{args.model_name}_final"
    model.save(model_path)
    print(f"\nTraining complete! Model saved to: {model_path}")


if __name__ == "__main__":
    main()
