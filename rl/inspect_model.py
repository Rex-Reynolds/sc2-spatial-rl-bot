#!/usr/bin/env python3
"""
Inspect a trained RL model to see what it learned.

Shows:
- Model architecture
- Action preferences in different game states
- Policy behavior patterns
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
import torch


def inspect_model(model_path, bot_type="basic"):
    """Load and inspect a trained model."""
    print("=" * 70)
    print(f"INSPECTING MODEL: {model_path}")
    print("=" * 70)

    # Load model
    print("\n1. Loading model...")
    model = PPO.load(model_path)

    # Model architecture
    print("\n2. MODEL ARCHITECTURE")
    print("-" * 70)
    print(f"Policy type: {type(model.policy).__name__}")
    print(f"Observation space: {model.observation_space}")
    print(f"Action space: {model.action_space}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Gamma (discount): {model.gamma}")

    # Network structure
    print("\n3. NEURAL NETWORK STRUCTURE")
    print("-" * 70)
    print(model.policy)

    # Count parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test action preferences in different scenarios
    print("\n4. ACTION PREFERENCES IN DIFFERENT SCENARIOS")
    print("-" * 70)

    if bot_type == "basic":
        action_names = [
            "train_scv", "build_supply_depot", "build_barracks",
            "train_marine", "attack", "defend", "no_op"
        ]
        scenarios = get_basic_scenarios()
    else:
        action_names = [
            "train_scv", "build_supply_depot", "build_refinery",
            "build_barracks", "build_factory", "build_starport",
            "build_tech_lab_barracks", "build_reactor_barracks", "build_tech_lab_factory",
            "train_marine", "train_marauder", "train_tank", "train_hellion", "train_medivac",
            "research_stim", "research_combat_shields", "research_concussive_shells",
            "upgrade_infantry_weapons", "upgrade_infantry_armor",
            "attack", "defend", "expand", "no_op"
        ]
        scenarios = get_advanced_scenarios()

    for scenario_name, obs in scenarios:
        print(f"\n{scenario_name}:")
        print(f"  Observation: {obs}")

        # Get action probabilities
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.numpy()[0]

        # Get deterministic action
        action, _ = model.predict(obs, deterministic=True)

        # Show top 3 actions
        top_3_indices = np.argsort(probs)[-3:][::-1]
        print(f"  Chosen action: {action} ({action_names[action]})")
        print(f"  Top 3 actions:")
        for idx in top_3_indices:
            print(f"    - {action_names[idx]}: {probs[idx]*100:.1f}%")


def get_basic_scenarios():
    """Get test scenarios for basic bot (11 obs)."""
    return [
        ("Early game (high minerals, no army)", np.array([
            0.8,   # minerals (high)
            0.0,   # gas
            0.3,   # supply_used
            0.5,   # supply_cap
            0.3,   # scv_count
            0.0,   # marine_count (none yet)
            0.0,   # barracks_count (none yet)
            0.0,   # enemy_units
            1.0,   # enemy_structures (they exist)
            0.1,   # game_time (early)
            0.3,   # army_strength
        ], dtype=np.float32)),

        ("Mid game (army ready, enemy visible)", np.array([
            0.3,   # minerals (medium)
            0.1,   # gas
            0.7,   # supply_used
            0.8,   # supply_cap
            0.5,   # scv_count
            0.4,   # marine_count (20 marines)
            0.3,   # barracks_count (3 rax)
            0.3,   # enemy_units (some visible)
            1.0,   # enemy_structures
            0.5,   # game_time (mid)
            0.6,   # army_strength (we're ahead)
        ], dtype=np.float32)),

        ("Supply blocked", np.array([
            0.9,   # minerals (very high)
            0.0,   # gas
            0.95,  # supply_used (almost capped!)
            0.95,  # supply_cap
            0.5,   # scv_count
            0.2,   # marine_count
            0.2,   # barracks_count
            0.0,   # enemy_units
            1.0,   # enemy_structures
            0.3,   # game_time
            0.4,   # army_strength
        ], dtype=np.float32)),

        ("Low minerals, need economy", np.array([
            0.05,  # minerals (very low!)
            0.0,   # gas
            0.4,   # supply_used
            0.6,   # supply_cap
            0.2,   # scv_count (too few workers)
            0.1,   # marine_count
            0.1,   # barracks_count
            0.0,   # enemy_units
            1.0,   # enemy_structures
            0.4,   # game_time
            0.4,   # army_strength
        ], dtype=np.float32)),
    ]


def get_advanced_scenarios():
    """Get test scenarios for advanced bot (25 obs)."""
    return [
        ("Early game", np.array([
            0.8, 0.0, 0.3, 0.5,  # minerals, gas, supply
            0.3, 0.0, 0.0, 0.0, 0.0, 0.0,  # units (just SCVs)
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # buildings (just CC)
            0.0, 0.0, 0.0,  # no upgrades
            0.0, 0.0,  # weapon/armor levels
            0.0, 1.0,  # enemy counts
            0.1, 0.1,  # game_time, army_supply
        ], dtype=np.float32)),

        ("Mid game army", np.array([
            0.4, 0.2, 0.7, 0.8,  # minerals, gas, supply
            0.5, 0.3, 0.2, 0.0, 0.0, 0.0,  # marines + marauders
            1.0, 0.3, 0.2, 0.0, 0.2, 0.2, 0.1,  # buildings
            1.0, 1.0, 0.0,  # stim + shields researched
            1.0, 1.0,  # weapon/armor +1
            0.3, 1.0,  # enemy present
            0.5, 0.4,  # mid game, decent army
        ], dtype=np.float32)),
    ]


def main():
    parser = argparse.ArgumentParser(description="Inspect trained RL model")
    parser.add_argument("model", help="Path to model (.zip file)")
    parser.add_argument(
        "--bot-type",
        choices=["basic", "advanced"],
        default="basic",
        help="Bot type (basic=7 actions, advanced=23 actions)"
    )

    args = parser.parse_args()

    inspect_model(args.model, args.bot_type)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
