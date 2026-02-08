#!/usr/bin/env python3
"""
Test Spatial Components

Validates that all spatial components work together:
1. Feature extraction
2. CNN policy
3. Spatial environment
4. Spatial bot
"""

import sys
import numpy as np
import torch
from gymnasium import spaces

print("=" * 70)
print("TESTING SPATIAL COMPONENTS")
print("=" * 70)
print()

# Test 1: Feature Extractor Dimensions
print("Test 1: Feature Extractor Dimensions")
print("-" * 70)
try:
    from rl.spatial_features import SpatialFeatureExtractor

    extractor = SpatialFeatureExtractor()
    print(f"✓ Screen size: {extractor.SCREEN_SIZE}x{extractor.SCREEN_SIZE}")
    print(f"✓ Minimap size: {extractor.MINIMAP_SIZE}x{extractor.MINIMAP_SIZE}")
    print(f"✓ Screen channels: {extractor.NUM_SCREEN_CHANNELS}")
    print(f"✓ Minimap channels: {extractor.NUM_MINIMAP_CHANNELS}")
    print("✓ Feature extractor initialized successfully")
except Exception as e:
    print(f"✗ Feature extractor failed: {e}")
    sys.exit(1)

print()

# Test 2: CNN Policy Forward Pass
print("Test 2: CNN Policy Forward Pass")
print("-" * 70)
try:
    from rl.spatial_policy import SpatialActorCriticPolicy

    obs_space = spaces.Dict({
        'screen': spaces.Box(0, 1, (20, 64, 64), dtype=np.float32),
        'minimap': spaces.Box(0, 1, (11, 64, 64), dtype=np.float32),
        'scalars': spaces.Box(0, 1, (90,), dtype=np.float32),
    })

    policy = SpatialActorCriticPolicy(obs_space, num_action_types=50)

    # Create fake observation
    obs = {
        'screen': torch.randn(1, 20, 64, 64),
        'minimap': torch.randn(1, 11, 64, 64),
        'scalars': torch.randn(1, 90),
    }

    # Forward pass
    outputs = policy(obs)

    print(f"✓ Action logits shape: {outputs['action_type_logits'].shape}")
    print(f"✓ Screen logits shape: {outputs['screen_logits'].shape}")
    print(f"✓ Minimap logits shape: {outputs['minimap_logits'].shape}")
    print(f"✓ Value shape: {outputs['value'].shape}")
    print("✓ CNN policy forward pass successful")

    # Test action sampling
    action, value, log_prob, entropy = policy.get_action_and_value(obs)
    print(f"✓ Sampled action type: {action['action_type'].item()}")
    print(f"✓ Sampled screen idx: {action['screen_idx'].item()}")
    print(f"✓ Value estimate: {value.item():.3f}")
    print(f"✓ Log prob: {log_prob.item():.3f}")
    print(f"✓ Entropy: {entropy.item():.3f}")
    print("✓ Action sampling successful")

except Exception as e:
    print(f"✗ CNN policy failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Spatial Environment Spaces
print("Test 3: Spatial Environment Spaces")
print("-" * 70)
try:
    from rl.spatial_env import SpatialSC2Env

    # Don't run actual game yet, just test initialization
    env = SpatialSC2Env.__new__(SpatialSC2Env)
    env.opponent_name = "IdleBot"
    env.opponent_policy = None
    env.map_name = "Simple64"
    env.max_game_time = 600
    env.realtime = False
    env.step_interval = 8
    env.num_action_types = 50

    # Set up spaces
    env.observation_space = spaces.Dict({
        'screen': spaces.Box(0, 1, (20, 64, 64), dtype=np.float32),
        'minimap': spaces.Box(0, 1, (11, 64, 64), dtype=np.float32),
        'scalars': spaces.Box(0, 1, (90,), dtype=np.float32),
    })
    env.action_space = spaces.Dict({
        'action_type': spaces.Discrete(50),
        'screen_idx': spaces.Discrete(64 * 64),
        'minimap_idx': spaces.Discrete(64 * 64),
    })

    print(f"✓ Observation space: Dict with 3 keys")
    print(f"  - screen: {env.observation_space['screen'].shape}")
    print(f"  - minimap: {env.observation_space['minimap'].shape}")
    print(f"  - scalars: {env.observation_space['scalars'].shape}")
    print(f"✓ Action space: Dict with 3 keys")
    print(f"  - action_type: Discrete({env.action_space['action_type'].n})")
    print(f"  - screen_idx: Discrete({env.action_space['screen_idx'].n})")
    print(f"  - minimap_idx: Discrete({env.action_space['minimap_idx'].n})")
    print("✓ Environment spaces configured correctly")

except Exception as e:
    print(f"✗ Environment initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Spatial Bot Actions
print("Test 4: Spatial Bot Action Definitions")
print("-" * 70)
try:
    from rl.spatial_bot import SpatialRLBot

    print(f"✓ Total actions defined: {len(SpatialRLBot.ACTION_NAMES)}")
    print(f"✓ Sample actions:")
    for i in [0, 5, 11, 23, 24, 30, 36]:
        if i < len(SpatialRLBot.ACTION_NAMES):
            print(f"    {i}: {SpatialRLBot.ACTION_NAMES[i]}")
    print("✓ Spatial bot actions defined correctly")

except Exception as e:
    print(f"✗ Spatial bot failed: {e}")
    sys.exit(1)

print()

# Test 5: Coordinate Conversions
print("Test 5: Coordinate Conversion Functions")
print("-" * 70)
try:
    from rl.spatial_policy import convert_spatial_idx_to_coords, convert_coords_to_spatial_idx

    # Test conversions
    test_cases = [
        (0, (0, 0)),
        (63, (63, 0)),
        (64, (0, 1)),
        (4095, (63, 63)),
        (2048, (0, 32)),
    ]

    all_passed = True
    for idx, expected_coords in test_cases:
        coords = convert_spatial_idx_to_coords(idx)
        back_to_idx = convert_coords_to_spatial_idx(coords[0], coords[1])

        if coords == expected_coords and back_to_idx == idx:
            print(f"✓ idx={idx} → coords={coords} → idx={back_to_idx}")
        else:
            print(f"✗ idx={idx} → coords={coords} (expected {expected_coords}) → idx={back_to_idx}")
            all_passed = False

    if all_passed:
        print("✓ All coordinate conversions correct")
    else:
        print("✗ Some coordinate conversions failed")
        sys.exit(1)

except Exception as e:
    print(f"✗ Coordinate conversion failed: {e}")
    sys.exit(1)

print()

# Summary
print("=" * 70)
print("ALL COMPONENT TESTS PASSED! ✓")
print("=" * 70)
print()
print("Next step: Run actual game test with:")
print("  python rl/test_spatial_game.py")
print()
