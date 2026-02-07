#!/usr/bin/env python3
"""
Quick test to verify the RL environment works.

This runs a game with random actions to check the integration.
"""

import numpy as np
from rl.env import make_env


def main():
    print("=" * 70)
    print("RL ENVIRONMENT TEST")
    print("=" * 70)
    print("Testing environment with random policy...")
    print()

    # Create environment
    env = make_env(opponent="IdleBot", realtime=False)

    # Set a random policy
    def random_policy(obs):
        """Random action policy."""
        action = np.random.randint(0, 7)
        return action, None

    env.policy = random_policy

    # Test one episode
    print("Resetting environment (running complete game with random policy)...")
    obs, info = env.reset()
    print(f"Game finished! Trajectory length: {len(env.trajectory)}")
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print()

    # Replay trajectory
    print(f"Replaying {len(env.trajectory)} steps from trajectory...")
    total_reward = 0.0
    step = 0

    while True:
        action = np.random.randint(0, 7)  # Action is ignored in replay mode
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step += 1

        if step <= 5 or terminated or truncated:
            print(f"Step {step}: Reward={reward:.4f}, Done={terminated or truncated}")

        if terminated or truncated:
            print(f"\nEpisode ended after {step} steps")
            print(f"Result: {env.game_result}")
            print(f"Total reward: {total_reward:.2f}")
            break

    env.close()
    print("\n" + "=" * 70)
    print("Environment test complete!")
    print("If you see a game result above, the integration works!")
    print("=" * 70)


if __name__ == "__main__":
    main()
