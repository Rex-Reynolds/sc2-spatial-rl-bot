#!/usr/bin/env python3
"""
Self-Play Match: Watch your bot play against itself!
"""

import sys
import torch
from sc2 import maps
from sc2.main import run_game
from sc2.player import Bot
from sc2.data import Race

from rl.play_vs_bot import PlayableRLBot, load_trained_model


def main():
    if len(sys.argv) < 2:
        print("Usage: python rl/self_play_match.py <model_path>")
        print("Example: python rl/self_play_match.py rl/models/hyperparam_agent1_baseline/checkpoint_ep5.pt")
        sys.exit(1)

    model_path = sys.argv[1]

    print("=" * 70)
    print("SELF-PLAY MATCH")
    print("=" * 70)
    print(f"Model: {model_path}")
    print("")
    print("Loading model for both players...")

    # Load model once
    policy = load_trained_model(model_path)

    # Create two separate bot instances with the same policy
    bot1 = PlayableRLBot(policy)
    bot2 = PlayableRLBot(policy)

    print("✓ Model loaded successfully!")
    print("")
    print("Starting self-play match...")
    print("Player 1 (Blue): SpatialRLBot")
    print("Player 2 (Red):  SpatialRLBot")
    print("")
    print("Watch them battle it out! 🤖 vs 🤖")
    print("=" * 70)
    print("")

    # Run game: Bot vs itself
    result = run_game(
        maps.get("Simple64"),
        [
            Bot(Race.Terran, bot1, name="SpatialRLBot_Blue"),
            Bot(Race.Terran, bot2, name="SpatialRLBot_Red"),
        ],
        realtime=False,
        save_replay_as="replays/self_play_match.SC2Replay"
    )

    print("")
    print("=" * 70)
    print("SELF-PLAY MATCH COMPLETE")
    print("=" * 70)

    if result[0] == 1:
        print("Winner: Player 1 (Blue) 🔵")
    elif result[1] == 1:
        print("Winner: Player 2 (Red) 🔴")
    else:
        print("Result: Tie")

    print("")
    print("Replay saved: replays/self_play_match.SC2Replay")
    print("Open in StarCraft II to watch the match!")
    print("=" * 70)


if __name__ == "__main__":
    main()
