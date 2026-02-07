#!/usr/bin/env python3
"""
Train an RL agent to play StarCraft II using PPO.

Usage:
    python rl/train.py --opponent IdleBot --episodes 1000
"""

import argparse
import signal
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from rl.env import make_env

# Global flag for graceful shutdown
training_interrupted = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global training_interrupted
    print("\n\n⚠️  Training interrupted! Saving model and exiting...")
    training_interrupted = True
    sys.exit(0)


class EpisodeLimitCallback(BaseCallback):
    """Stop training after N episodes."""

    def __init__(self, max_episodes: int):
        super().__init__()
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Count episodes (dones)
        if self.locals.get("dones", [False])[0]:
            self.episode_count += 1
            print(f"\nCompleted episode {self.episode_count}/{self.max_episodes}")

            if self.episode_count >= self.max_episodes:
                print(f"\nReached {self.max_episodes} episodes. Stopping training.")
                return False  # Stop training
        return True  # Continue training


def main():
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

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
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without game window (faster, for background training)",
    )
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Train against itself (self-play mode)",
    )
    parser.add_argument(
        "--opponent-model",
        default=None,
        help="Path to opponent model for self-play (uses same model if not specified)",
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
    # Note: realtime=False makes it fast, but window still shows on macOS
    # True headless requires SC2 to run without display

    # Set up opponent policy for self-play
    opponent_policy = None
    if args.self_play:
        print("Self-play mode enabled!")
        if args.opponent_model:
            print(f"Loading opponent model: {args.opponent_model}")
            opponent_model = PPO.load(args.opponent_model)
            opponent_policy = lambda obs: opponent_model.predict(obs, deterministic=False)
        else:
            print("Opponent will use same model as player 1 (true self-play)")
            # opponent_policy will be set to the same model after it's created

    env = make_env(
        opponent=args.opponent if not args.self_play else "SelfPlay",
        opponent_policy=opponent_policy,
        realtime=False
    )
    env = Monitor(env)  # Wrap for logging

    if args.headless:
        print("Note: Headless mode requested. Game window may still appear on macOS.")

    # Create or load model
    if args.load_model:
        print(f"Loading existing model: {args.load_model}")
        model = PPO.load(args.load_model, env=env)
    else:
        print("Creating new PPO model...")
        ppo_kwargs = {
            "policy": "MlpPolicy",
            "env": env,
            "verbose": 1,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
        }

        # Only add tensorboard_log if tensorboard is enabled
        if not args.no_tensorboard:
            ppo_kwargs["tensorboard_log"] = f"./rl/logs/{args.model_name}"

        model = PPO(**ppo_kwargs)

    # Set the environment's policy to use the model
    def policy_fn(obs):
        """Policy function that queries the model."""
        return model.predict(obs, deterministic=False)

    env.unwrapped.policy = policy_fn

    # For true self-play (same model vs itself), set opponent policy too
    if args.self_play and not args.opponent_model:
        print("Setting up true self-play: same model will play both sides")
        env.unwrapped.opponent_policy = policy_fn

    # Callbacks for saving and episode limiting
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./rl/models/{args.model_name}",
        name_prefix="sc2_agent",
    )
    episode_limit_callback = EpisodeLimitCallback(max_episodes=args.episodes)

    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([checkpoint_callback, episode_limit_callback])

    # Train
    print(f"\nStarting training for {args.episodes} episodes...")
    print("This will take a while. Monitor progress with TensorBoard:")
    print(f"  tensorboard --logdir=./rl/logs/{args.model_name}")
    print()

    # Use generous timesteps estimate - episode callback will stop us
    total_timesteps = args.episodes * 2000

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("Saving current model before exiting...")

    # Save final model (even if interrupted)
    model_path = f"./rl/models/{args.model_name}_final"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    if training_interrupted:
        print("Training was interrupted. Model saved at current state.")
    else:
        print("Training complete!")


if __name__ == "__main__":
    main()
