#!/usr/bin/env python3
"""
Test trained RL agent against opponents.

Usage:
    python rl/test.py --model rl/models/sc2_ppo_final.zip --episodes 10
"""

import argparse
from stable_baselines3 import PPO
import numpy as np

from rl.env import make_env


def main():
    parser = argparse.ArgumentParser(description="Test SC2 RL agent")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--opponent",
        default="IdleBot",
        choices=["IdleBot", "RushBot", "DefenseBot"],
        help="Opponent bot to test against",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of test episodes (default: 10)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run in realtime mode (watchable)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("STARCRAFT II RL TESTING")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Opponent: {args.opponent}")
    print(f"Episodes: {args.episodes}")
    print("=" * 70)
    print()

    # Create environment
    env = make_env(opponent=args.opponent, realtime=args.realtime)

    # Load model
    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    # Test episodes
    wins = 0
    losses = 0
    total_reward = 0.0

    for episode in range(args.episodes):
        print(f"\n--- Episode {episode + 1}/{args.episodes} ---")

        obs, info = env.reset()
        episode_reward = 0.0
        step_count = 0
        done = False

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            done = terminated or truncated

        # Record results
        total_reward += episode_reward
        result = info.get("result")
        print(f"Result: {result}")
        print(f"Episode reward: {episode_reward:.2f}")
        print(f"Steps: {step_count}")

        if result == "Victory":
            wins += 1
        elif result == "Defeat":
            losses += 1

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Episodes: {args.episodes}")
    print(f"Wins: {wins} ({wins / args.episodes * 100:.1f}%)")
    print(f"Losses: {losses} ({losses / args.episodes * 100:.1f}%)")
    print(f"Average reward: {total_reward / args.episodes:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
