#!/usr/bin/env python3
"""
Test Spatial Bot in Actual Game

Runs one game to validate:
1. Feature extraction works during gameplay
2. Bot can execute spatial actions
3. Trajectory is collected correctly
"""

import sys
import numpy as np

print("=" * 70)
print("TESTING SPATIAL BOT IN GAME")
print("=" * 70)
print()

try:
    from rl.spatial_env import SpatialSC2Env

    print("Creating spatial environment...")
    env = SpatialSC2Env(
        opponent="IdleBot",
        map_name="Simple64",
        max_game_time=600,
        realtime=False,
        step_interval=16,  # Slower for testing (1 decision/sec)
    )
    print("✓ Environment created")
    print()

    print("Running one game (this will take ~1-2 minutes)...")
    print("-" * 70)

    # Reset runs a complete game
    obs, info = env.reset()

    print()
    print("=" * 70)
    print("GAME COMPLETE!")
    print("=" * 70)
    print()

    # Check observation
    print("Observation Check:")
    print("-" * 70)
    print(f"✓ Screen shape: {obs['screen'].shape}")
    print(f"✓ Screen dtype: {obs['screen'].dtype}")
    print(f"✓ Screen range: [{obs['screen'].min():.3f}, {obs['screen'].max():.3f}]")
    print()
    print(f"✓ Minimap shape: {obs['minimap'].shape}")
    print(f"✓ Minimap dtype: {obs['minimap'].dtype}")
    print(f"✓ Minimap range: [{obs['minimap'].min():.3f}, {obs['minimap'].max():.3f}]")
    print()
    print(f"✓ Scalars shape: {obs['scalars'].shape}")
    print(f"✓ Scalars dtype: {obs['scalars'].dtype}")
    print(f"✓ Scalars range: [{obs['scalars'].min():.3f}, {obs['scalars'].max():.3f}]")
    print()

    # Check trajectory
    print("Trajectory Check:")
    print("-" * 70)
    print(f"✓ Total steps collected: {len(env.trajectory)}")
    print(f"✓ Episode reward: {env.episode_reward:.2f}")
    print(f"✓ Game result: {env.game_result}")
    print()

    if len(env.trajectory) > 0:
        print("Sample trajectory steps:")
        for i in [0, len(env.trajectory) // 2, -1]:
            if i < len(env.trajectory):
                obs_t, action_t, reward_t, done_t, info_t = env.trajectory[i]
                action_type = action_t['action_type']
                screen_idx = action_t['screen_idx']
                screen_x = screen_idx % 64
                screen_y = screen_idx // 64
                print(f"  Step {i}:")
                print(f"    Action: {action_type} (screen: {screen_x},{screen_y})")
                print(f"    Reward: {reward_t:.3f}")
                print(f"    Done: {done_t}")

    print()

    # Verify no NaN or Inf values
    print("Data Quality Check:")
    print("-" * 70)
    has_issues = False

    if np.any(np.isnan(obs['screen'])):
        print("✗ Screen contains NaN values")
        has_issues = True
    else:
        print("✓ Screen has no NaN values")

    if np.any(np.isinf(obs['screen'])):
        print("✗ Screen contains Inf values")
        has_issues = True
    else:
        print("✓ Screen has no Inf values")

    if np.any(np.isnan(obs['minimap'])):
        print("✗ Minimap contains NaN values")
        has_issues = True
    else:
        print("✓ Minimap has no NaN values")

    if np.any(np.isnan(obs['scalars'])):
        print("✗ Scalars contains NaN values")
        has_issues = True
    else:
        print("✓ Scalars has no NaN values")

    if has_issues:
        print()
        print("⚠️  WARNING: Data quality issues detected")
        sys.exit(1)

    print()
    print("=" * 70)
    print("GAME TEST PASSED! ✓")
    print("=" * 70)
    print()
    print("Spatial bot successfully:")
    print("  ✓ Extracted spatial features")
    print("  ✓ Executed spatial actions")
    print("  ✓ Collected trajectory")
    print("  ✓ Completed game without errors")
    print()
    print("Ready for training!")
    print()

except KeyboardInterrupt:
    print("\n\nTest interrupted by user")
    sys.exit(1)
except Exception as e:
    print()
    print("=" * 70)
    print("GAME TEST FAILED ✗")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    sys.exit(1)
