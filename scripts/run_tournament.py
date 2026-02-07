#!/usr/bin/env python3
"""Run a tournament of N matches between two bots and track statistics."""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from sc2 import maps
from sc2.main import run_game
from sc2.player import Bot
from sc2.data import Race, Result

# Add parent directory to path to import bots
sys.path.insert(0, str(Path(__file__).parent.parent))

from bots.rush_bot import RushBot
from bots.idle_bot import IdleBot


def main():
    parser = argparse.ArgumentParser(description="Run a SC2 bot tournament")
    parser.add_argument(
        "-n",
        "--num-matches",
        type=int,
        default=10,
        help="Number of matches to run (default: 10)",
    )
    parser.add_argument(
        "--map",
        default="Simple64",
        help="Map name (default: Simple64)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="Game time limit in seconds (default: 300)",
    )
    parser.add_argument(
        "--save-replays",
        action="store_true",
        help="Save replays for each match",
    )

    args = parser.parse_args()

    # Statistics
    stats = {
        "rushbot_wins": 0,
        "idlebot_wins": 0,
        "ties": 0,
        "errors": 0,
    }

    print("=" * 60)
    print(f"STARCRAFT II TOURNAMENT")
    print(f"RushBot vs IdleBot")
    print(f"Matches: {args.num_matches}")
    print(f"Map: {args.map}")
    print(f"Time limit: {args.time_limit}s")
    print("=" * 60)
    print()

    # Ensure replay directory exists if saving replays
    if args.save_replays:
        replay_dir = Path("replays")
        replay_dir.mkdir(exist_ok=True)

    # Run matches sequentially
    for match_num in range(1, args.num_matches + 1):
        print(f"Match {match_num}/{args.num_matches}...", end=" ", flush=True)

        # Prepare replay path if saving
        replay_path = None
        if args.save_replays:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            replay_path = f"replays/tournament_{timestamp}_match{match_num}.SC2Replay"

        try:
            # Create fresh bot instances for each match
            result = run_game(
                maps.get(args.map),
                [
                    Bot(Race.Terran, RushBot()),
                    Bot(Race.Terran, IdleBot()),
                ],
                realtime=False,
                save_replay_as=replay_path,
                game_time_limit=args.time_limit,
            )

            # Track results
            if result[0] == Result.Victory:
                stats["rushbot_wins"] += 1
                print("RushBot wins")
            elif result[1] == Result.Victory:
                stats["idlebot_wins"] += 1
                print("IdleBot wins")
            else:
                stats["ties"] += 1
                print("Tie")

        except Exception as e:
            stats["errors"] += 1
            print(f"ERROR: {e}")
            # Continue through errors

    # Print final statistics
    print()
    print("=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    print(f"Total matches: {args.num_matches}")
    print(f"RushBot wins: {stats['rushbot_wins']} ({stats['rushbot_wins']/args.num_matches*100:.1f}%)")
    print(f"IdleBot wins: {stats['idlebot_wins']} ({stats['idlebot_wins']/args.num_matches*100:.1f}%)")
    print(f"Ties: {stats['ties']} ({stats['ties']/args.num_matches*100:.1f}%)")
    print(f"Errors: {stats['errors']} ({stats['errors']/args.num_matches*100:.1f}%)")
    print("=" * 60)

    # Determine overall winner
    if stats['rushbot_wins'] > stats['idlebot_wins']:
        print(f"\nTOURNAMENT WINNER: RushBot")
    elif stats['idlebot_wins'] > stats['rushbot_wins']:
        print(f"\nTOURNAMENT WINNER: IdleBot")
    else:
        print(f"\nTOURNAMENT RESULT: Tie")


if __name__ == "__main__":
    main()
