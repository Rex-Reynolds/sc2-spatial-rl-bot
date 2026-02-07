#!/usr/bin/env python3
"""Run a custom match between any two bots."""

import argparse
import sys
from pathlib import Path

from sc2 import maps
from sc2.main import run_game
from sc2.player import Bot
from sc2.data import Race, Result

# Add parent directory to path to import bots
sys.path.insert(0, str(Path(__file__).parent.parent))

from bots import (
    IdleBot,
    RushBot,
    DefenseBot,
    EconomyBot,
    ProxyBot,
    StimBot,
    TankBot,
    BioBallBot,
    MechBot,
)

# Bot registry
BOTS = {
    "IdleBot": IdleBot,
    "RushBot": RushBot,
    "DefenseBot": DefenseBot,
    "EconomyBot": EconomyBot,
    "ProxyBot": ProxyBot,
    "StimBot": StimBot,
    "TankBot": TankBot,
    "BioBallBot": BioBallBot,
    "MechBot": MechBot,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run a custom match between any two bots"
    )
    parser.add_argument(
        "bot1",
        choices=list(BOTS.keys()),
        help="First bot",
    )
    parser.add_argument(
        "bot2",
        choices=list(BOTS.keys()),
        help="Second bot",
    )
    parser.add_argument(
        "--map",
        default="Simple64",
        help="Map name (default: Simple64)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run in realtime mode (watch the game!)",
    )
    parser.add_argument(
        "--replay",
        default=None,
        help="Path to save replay",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="Game time limit in seconds (default: 600)",
    )

    args = parser.parse_args()

    # Prepare replay path
    replay_path = args.replay
    if replay_path:
        replay_path = Path(replay_path)
        replay_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"CUSTOM MATCH")
    print(f"{'='*60}")
    print(f"{args.bot1} vs {args.bot2}")
    print(f"Map: {args.map}")
    print(f"Realtime: {args.realtime}")
    print(f"Time limit: {args.time_limit}s")
    if replay_path:
        print(f"Replay: {replay_path}")
    print(f"{'='*60}")
    print()

    # Run the game
    result = run_game(
        maps.get(args.map),
        [
            Bot(Race.Terran, BOTS[args.bot1](), name=args.bot1),
            Bot(Race.Terran, BOTS[args.bot2](), name=args.bot2),
        ],
        realtime=args.realtime,
        save_replay_as=str(replay_path) if replay_path else None,
        game_time_limit=args.time_limit,
    )

    print()
    print(f"{'='*60}")
    print(f"MATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Result: {result}")

    if result[0] == Result.Victory:
        print(f"Winner: {args.bot1} 🏆")
    elif result[1] == Result.Victory:
        print(f"Winner: {args.bot2} 🏆")
    else:
        print("Result: Tie")

    if replay_path:
        print(f"\nReplay saved to: {replay_path}")


if __name__ == "__main__":
    main()
