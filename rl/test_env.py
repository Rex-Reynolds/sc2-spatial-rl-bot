#!/usr/bin/env python3
"""
Quick test to verify the RL environment works.

This runs a few random actions to check the integration.
"""

import numpy as np
from rl.env import make_env


def main():
    print("=" * 70)
    print("RL ENVIRONMENT TEST")
    print("=" * 70)
    print("Testing environment with random actions...")
    print()

    # Create environment
    env = make_env(opponent="IdleBot", realtime=False)

    # Test one episode with random actions
    print("Resetting environment (starting game)...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print()

    print("Taking 10 random actions...")
    for step in range(10):
        action = env.action_space.sample()  # Random action
        print(f"Step {step + 1}: Action {action}")

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"  Observation: {obs[:5]}...")  # First 5 features
        print(f"  Reward: {reward:.4f}")
        print(f"  Done: {terminated or truncated}")

        if terminated or truncated:
            print(f"\nEpisode ended after {step + 1} steps")
            print(f"Result: {info.get('result')}")
            print(f"Total reward: {info.get('episode_reward', 0):.2f}")
            break

    env.close()
    print("\n" + "=" * 70)
    print("Environment test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
