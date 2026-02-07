#!/usr/bin/env python3
"""Run a round-robin tournament where all bots play against each other."""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from itertools import combinations

from sc2 import maps
from sc2.main import run_game
from sc2.player import Bot
from sc2.data import Race, Result

# Add parent directory to path to import bots
sys.path.insert(0, str(Path(__file__).parent.parent))

from bots.rush_bot import RushBot
from bots.idle_bot import IdleBot
from bots.defense_bot import DefenseBot
from bots.economy_bot import EconomyBot
from bots.proxy_bot import ProxyBot


# Bot registry
BOTS = {
    "RushBot": RushBot,
    "IdleBot": IdleBot,
    "DefenseBot": DefenseBot,
    "EconomyBot": EconomyBot,
    "ProxyBot": ProxyBot,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run a round-robin tournament with all bots"
    )
    parser.add_argument(
        "-n",
        "--matches-per-pairing",
        type=int,
        default=5,
        help="Number of matches per bot pairing (default: 5)",
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
        "--bots",
        nargs="+",
        default=None,
        help="Specific bots to include (default: all bots)",
    )

    args = parser.parse_args()

    # Determine which bots to use
    if args.bots:
        selected_bots = {name: BOTS[name] for name in args.bots if name in BOTS}
        if not selected_bots:
            print(f"Error: No valid bots selected from {args.bots}")
            print(f"Available bots: {list(BOTS.keys())}")
            return
    else:
        selected_bots = BOTS

    # Initialize stats
    stats = {name: {"wins": 0, "losses": 0, "ties": 0} for name in selected_bots}

    print("=" * 70)
    print("STARCRAFT II ROUND-ROBIN TOURNAMENT")
    print("=" * 70)
    print(f"Bots: {', '.join(selected_bots.keys())}")
    print(f"Matches per pairing: {args.matches_per_pairing}")
    print(f"Map: {args.map}")
    print(f"Time limit: {args.time_limit}s")
    print("=" * 70)
    print()

    # Generate all pairings
    pairings = list(combinations(selected_bots.keys(), 2))
    total_matches = len(pairings) * args.matches_per_pairing

    print(f"Total pairings: {len(pairings)}")
    print(f"Total matches: {total_matches}")
    print()

    match_count = 0

    # Run round-robin tournament
    for bot1_name, bot2_name in pairings:
        print(f"\n{'='*70}")
        print(f"{bot1_name} vs {bot2_name}")
        print(f"{'='*70}")

        pairing_wins = {bot1_name: 0, bot2_name: 0, "ties": 0}

        for match_num in range(1, args.matches_per_pairing + 1):
            match_count += 1
            print(
                f"Match {match_num}/{args.matches_per_pairing} "
                f"({match_count}/{total_matches})...",
                end=" ",
                flush=True,
            )

            try:
                # Create fresh bot instances
                result = run_game(
                    maps.get(args.map),
                    [
                        Bot(Race.Terran, selected_bots[bot1_name](), name=bot1_name),
                        Bot(Race.Terran, selected_bots[bot2_name](), name=bot2_name),
                    ],
                    realtime=False,
                    game_time_limit=args.time_limit,
                )

                # Track results
                if result[0] == Result.Victory:
                    stats[bot1_name]["wins"] += 1
                    stats[bot2_name]["losses"] += 1
                    pairing_wins[bot1_name] += 1
                    print(f"{bot1_name} wins")
                elif result[1] == Result.Victory:
                    stats[bot2_name]["wins"] += 1
                    stats[bot1_name]["losses"] += 1
                    pairing_wins[bot2_name] += 1
                    print(f"{bot2_name} wins")
                else:
                    stats[bot1_name]["ties"] += 1
                    stats[bot2_name]["ties"] += 1
                    pairing_wins["ties"] += 1
                    print("Tie")

            except Exception as e:
                print(f"ERROR: {e}")

        # Print pairing summary
        print(f"\nPairing summary:")
        print(f"  {bot1_name}: {pairing_wins[bot1_name]} wins")
        print(f"  {bot2_name}: {pairing_wins[bot2_name]} wins")
        print(f"  Ties: {pairing_wins['ties']}")

    # Calculate final standings
    print("\n" + "=" * 70)
    print("FINAL STANDINGS")
    print("=" * 70)

    # Sort by wins, then by win rate
    standings = []
    for bot_name in selected_bots:
        wins = stats[bot_name]["wins"]
        losses = stats[bot_name]["losses"]
        ties = stats[bot_name]["ties"]
        total = wins + losses + ties
        win_rate = (wins / total * 100) if total > 0 else 0

        standings.append(
            {
                "name": bot_name,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "total": total,
                "win_rate": win_rate,
            }
        )

    # Sort by win rate descending
    standings.sort(key=lambda x: x["win_rate"], reverse=True)

    # Print standings table
    print(f"{'Rank':<6} {'Bot':<15} {'Wins':<6} {'Losses':<8} {'Ties':<6} {'Win Rate':<10}")
    print("-" * 70)

    for rank, bot_stats in enumerate(standings, 1):
        print(
            f"{rank:<6} {bot_stats['name']:<15} "
            f"{bot_stats['wins']:<6} {bot_stats['losses']:<8} "
            f"{bot_stats['ties']:<6} {bot_stats['win_rate']:.1f}%"
        )

    print("=" * 70)
    print(f"\nCHAMPION: {standings[0]['name']} ({standings[0]['win_rate']:.1f}% win rate)")
    print()


if __name__ == "__main__":
    main()
