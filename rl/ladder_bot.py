#!/usr/bin/env python3
"""
Ladder Bot for Tournament Deployment
Compatible with SC2 Ladder Manager and AI Arena
"""

import sys
import os
import torch

from sc2 import maps, Race
from sc2.main import run_game
from sc2.player import Bot

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.play_vs_bot import PlayableRLBot, load_trained_model


def main():
    """
    Main entry point for ladder games.

    Ladder managers typically call this script with:
    - sys.argv[1]: Map name
    - Environment variables for opponent configuration
    """

    # Default configuration
    model_path = os.environ.get(
        "MODEL_PATH",
        "rl/models/curriculum_stage4_selfplay/final_model.pt"
    )

    # Get map name from command line or environment
    if len(sys.argv) > 1:
        map_name = sys.argv[1]
    else:
        map_name = os.environ.get("MAP_NAME", "Simple64")

    # Load trained model
    print(f"[SpatialRLBot] Loading model: {model_path}")
    try:
        policy = load_trained_model(model_path)
        bot = PlayableRLBot(policy)
        print(f"[SpatialRLBot] ✓ Model loaded successfully")
    except Exception as e:
        print(f"[SpatialRLBot] ✗ Failed to load model: {e}")
        sys.exit(1)

    # Get replay save path
    replay_path = os.environ.get("REPLAY_PATH")
    if replay_path:
        print(f"[SpatialRLBot] Will save replay to: {replay_path}")

    # Run game
    # Note: Ladder manager handles opponent configuration
    # This script only needs to provide our bot
    print(f"[SpatialRLBot] Starting game on {map_name}")

    try:
        result = run_game(
            maps.get(map_name),
            [Bot(Race.Terran, bot, name="SpatialRLBot")],
            realtime=False,
            save_replay_as=replay_path
        )

        # Log result
        result_str = "WIN" if result[0] == 1 else "LOSS"
        print(f"[SpatialRLBot] Game finished: {result_str}")

    except Exception as e:
        print(f"[SpatialRLBot] ✗ Game error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
