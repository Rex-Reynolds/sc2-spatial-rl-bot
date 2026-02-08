"""
High Resolution Configuration (128×128)

Increases spatial resolution for better fine-grained control.
Requires more compute but provides better spatial reasoning.

Usage:
    In spatial_features.py, change:
    SCREEN_SIZE = 128  # was 64
    MINIMAP_SIZE = 128  # was 64

    Update observation space in spatial_env.py:
    'screen': (20, 128, 128)
    'minimap': (11, 128, 128)

    Update action space:
    'screen_idx': Discrete(128 * 128)  # 16384 instead of 4096
    'minimap_idx': Discrete(128 * 128)
"""

# High-res configuration
SCREEN_SIZE_HIGH_RES = 128
MINIMAP_SIZE_HIGH_RES = 128

# Computational cost comparison
COST_64x64 = {
    'screen_features': 64 * 64 * 20,  # 81,920
    'minimap_features': 64 * 64 * 11,  # 45,056
    'screen_actions': 64 * 64,  # 4,096
    'total_obs': 81920 + 45056 + 90,  # ~127K
    'total_actions': 50 * 4096,  # ~205K
}

COST_128x128 = {
    'screen_features': 128 * 128 * 20,  # 327,680 (4x more)
    'minimap_features': 128 * 128 * 11,  # 180,224 (4x more)
    'screen_actions': 128 * 128,  # 16,384 (4x more)
    'total_obs': 327680 + 180224 + 90,  # ~508K
    'total_actions': 50 * 16384,  # ~819K
}

print("Resolution Comparison:")
print(f"64×64:  {COST_64x64['total_obs']:,} obs features, {COST_64x64['total_actions']:,} action combinations")
print(f"128×128: {COST_128x128['total_obs']:,} obs features, {COST_128x128['total_actions']:,} action combinations")
print(f"Increase: {COST_128x128['total_obs'] / COST_64x64['total_obs']:.1f}x obs, {COST_128x128['total_actions'] / COST_64x64['total_actions']:.1f}x actions")

# Recommended settings for high-res
HIGH_RES_SETTINGS = {
    'learning_rate': 1e-4,  # Lower LR for stability
    'batch_size': 32,  # Smaller batch due to memory
    'ppo_epochs': 2,  # Fewer epochs to save time
    'step_interval': 16,  # Keep same decision frequency
    'use_cuda': True,  # GPU strongly recommended
}

print("\nRecommended high-res settings:")
for k, v in HIGH_RES_SETTINGS.items():
    print(f"  {k}: {v}")

print("\n⚠️  WARNING: High-res training requires:")
print("  - GPU with 8GB+ VRAM")
print("  - 2-3x more training time per episode")
print("  - Start with 64×64, upgrade to 128×128 after validation")
