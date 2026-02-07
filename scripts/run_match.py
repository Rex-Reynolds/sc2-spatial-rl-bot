#!/usr/bin/env python3
"""Run a single StarCraft II match between two bots."""

import argparse
import sys
from pathlib import Path

from sc2 import maps, run_game
from sc2.player import Bot
from sc2.data import Race, Result

# Add parent directory to path to import bots
sys.path.insert(0, str(Path(__file__).parent.parent))

from bots.rush_bot import RushBot
from bots.idle_bot import IdleBot


def main():
    parser = argparse.ArgumentParser(description="Run a single SC2 bot match")
    parser.add_argument(
        "--map",
        default="Simple64",
        help="Map name (default: Simple64)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run in realtime mode (default: off for fast simulation)",
    )
    parser.add_argument(
        "--replay",
        default="replays/match.SC2Replay",
        help="Path to save replay (default: replays/match.SC2Replay)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="Game time limit in seconds (default: 300)",
    )

    args = parser.parse_args()

    # Ensure replay directory exists
    replay_path = Path(args.replay)
    replay_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting match on {args.map}...")
    print(f"RushBot vs IdleBot")
    print(f"Realtime: {args.realtime}")
    print(f"Time limit: {args.time_limit}s")
    print(f"Replay: {replay_path}")
    print("-" * 50)

    # Run the game
    result = run_game(
        maps.get(args.map),
        [
            Bot(Race.Terran, RushBot()),
            Bot(Race.Terran, IdleBot()),
        ],
        realtime=args.realtime,
        save_replay_as=str(replay_path),
        game_time_limit=args.time_limit,
    )

    print("-" * 50)
    print(f"Match complete!")
    print(f"Result: {result}")

    if result[0] == Result.Victory:
        print("Winner: RushBot")
    elif result[1] == Result.Victory:
        print("Winner: IdleBot")
    else:
        print("Result: Tie")

    print(f"\nReplay saved to: {replay_path}")


if __name__ == "__main__":
    main()
