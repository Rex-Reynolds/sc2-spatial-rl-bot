#!/usr/bin/env python3
"""
Record demonstration games for showcasing the bot
"""

import sys
import os
import torch
from datetime import datetime
from pathlib import Path

from sc2 import maps, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.data import Difficulty

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.play_vs_bot import PlayableRLBot, load_trained_model


def record_demo_games(model_path: str, num_games: int = 5, output_dir: str = "replays"):
    """
    Record demonstration games against various opponents.

    Args:
        model_path: Path to trained model
        num_games: Number of games per difficulty level
        output_dir: Directory to save replays
    """

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Load trained model
    print("=" * 70)
    print("SC2 DEMO GAME RECORDER")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}/")
    print(f"Games per difficulty: {num_games}")
    print("=" * 70)
    print("")

    policy = load_trained_model(model_path)
    bot = PlayableRLBot(policy)

    # Different opponent difficulties
    opponents = [
        ("Easy", Difficulty.Easy),
        ("Medium", Difficulty.Medium),
        ("MediumHard", Difficulty.MediumHard),
        ("Hard", Difficulty.Hard),
        ("VeryHard", Difficulty.VeryHard),
    ]

    total_games = len(opponents) * num_games
    game_count = 0
    results_by_difficulty = {name: {"wins": 0, "losses": 0} for name, _ in opponents}

    for difficulty_name, difficulty in opponents:
        print(f"\n{'='*70}")
        print(f"Playing vs {difficulty_name} AI")
        print(f"{'='*70}\n")

        for i in range(num_games):
            game_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            replay_name = f"demo_{difficulty_name}_{i+1}_{timestamp}.SC2Replay"
            replay_path = os.path.join(output_dir, replay_name)

            print(f"[{game_count}/{total_games}] Recording: {replay_name}...")

            try:
                result = run_game(
                    maps.get("Simple64"),
                    [
                        Bot(Race.Terran, bot, name="SpatialRLBot"),
                        Computer(Race.Random, difficulty)
                    ],
                    realtime=False,
                    save_replay_as=replay_path
                )

                is_win = result[0] == 1
                result_str = "WIN" if is_win else "LOSS"

                if is_win:
                    results_by_difficulty[difficulty_name]["wins"] += 1
                else:
                    results_by_difficulty[difficulty_name]["losses"] += 1

                print(f"  ✓ Result: {result_str}")
                print(f"  Saved: {replay_path}")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                results_by_difficulty[difficulty_name]["losses"] += 1

    # Print summary
    print("\n" + "=" * 70)
    print("RECORDING COMPLETE - SUMMARY")
    print("=" * 70)

    total_wins = 0
    total_losses = 0

    for difficulty_name, stats in results_by_difficulty.items():
        wins = stats["wins"]
        losses = stats["losses"]
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0

        total_wins += wins
        total_losses += losses

        print(f"{difficulty_name:12s}: {wins}/{total} wins ({win_rate:.1f}%)")

    total = total_wins + total_losses
    overall_rate = (total_wins / total * 100) if total > 0 else 0

    print("-" * 70)
    print(f"{'OVERALL':12s}: {total_wins}/{total} wins ({overall_rate:.1f}%)")
    print("=" * 70)
    print(f"\nReplays saved to: {output_dir}/")
    print(f"Total replays: {game_count}")
    print("")


def record_specific_matchup(
    model_path: str,
    opponent: str = "Medium",
    map_name: str = "Simple64",
    num_games: int = 1,
    output_dir: str = "replays"
):
    """
    Record games against a specific opponent.

    Args:
        model_path: Path to trained model
        opponent: Opponent difficulty (Easy, Medium, Hard, VeryHard, etc.)
        map_name: Map to play on
        num_games: Number of games to record
        output_dir: Directory to save replays
    """

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Load model
    print(f"Loading model: {model_path}")
    policy = load_trained_model(model_path)
    bot = PlayableRLBot(policy)

    # Map opponent string to difficulty
    difficulty_map = {
        "VeryEasy": Difficulty.VeryEasy,
        "Easy": Difficulty.Easy,
        "Medium": Difficulty.Medium,
        "MediumHard": Difficulty.MediumHard,
        "Hard": Difficulty.Hard,
        "Harder": Difficulty.Harder,
        "VeryHard": Difficulty.VeryHard,
    }

    if opponent not in difficulty_map:
        print(f"Error: Invalid opponent '{opponent}'")
        print(f"Valid options: {', '.join(difficulty_map.keys())}")
        return

    difficulty = difficulty_map[opponent]

    print(f"\nRecording {num_games} game(s) vs {opponent} on {map_name}...\n")

    wins = 0
    for i in range(num_games):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        replay_name = f"{map_name}_{opponent}_{i+1}_{timestamp}.SC2Replay"
        replay_path = os.path.join(output_dir, replay_name)

        print(f"[{i+1}/{num_games}] Recording: {replay_name}...")

        try:
            result = run_game(
                maps.get(map_name),
                [
                    Bot(Race.Terran, bot, name="SpatialRLBot"),
                    Computer(Race.Random, difficulty)
                ],
                realtime=False,
                save_replay_as=replay_path
            )

            is_win = result[0] == 1
            if is_win:
                wins += 1

            result_str = "WIN" if is_win else "LOSS"
            print(f"  ✓ Result: {result_str}")
            print(f"  Saved: {replay_path}\n")

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    win_rate = (wins / num_games * 100) if num_games > 0 else 0
    print(f"Summary: {wins}/{num_games} wins ({win_rate:.1f}%)")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python rl/record_demo_games.py <model_path> [num_games] [output_dir]")
        print("")
        print("Examples:")
        print("  # Record 5 games per difficulty (default)")
        print("  python rl/record_demo_games.py rl/models/final_model.pt")
        print("")
        print("  # Record 10 games per difficulty")
        print("  python rl/record_demo_games.py rl/models/final_model.pt 10")
        print("")
        print("  # Custom output directory")
        print("  python rl/record_demo_games.py rl/models/final_model.pt 5 my_replays")
        print("")
        print("Specific matchup:")
        print("  python rl/record_demo_games.py rl/models/final_model.pt --opponent Hard --map Simple96 --games 3")
        sys.exit(1)

    model_path = sys.argv[1]

    # Check for specific matchup flags
    if "--opponent" in sys.argv:
        opponent_idx = sys.argv.index("--opponent") + 1
        opponent = sys.argv[opponent_idx] if opponent_idx < len(sys.argv) else "Medium"

        map_name = "Simple64"
        if "--map" in sys.argv:
            map_idx = sys.argv.index("--map") + 1
            map_name = sys.argv[map_idx] if map_idx < len(sys.argv) else "Simple64"

        num_games = 1
        if "--games" in sys.argv:
            games_idx = sys.argv.index("--games") + 1
            num_games = int(sys.argv[games_idx]) if games_idx < len(sys.argv) else 1

        output_dir = "replays"
        if "--output" in sys.argv:
            output_idx = sys.argv.index("--output") + 1
            output_dir = sys.argv[output_idx] if output_idx < len(sys.argv) else "replays"

        record_specific_matchup(model_path, opponent, map_name, num_games, output_dir)

    else:
        # Record full suite of demo games
        num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "replays"

        record_demo_games(model_path, num_games, output_dir)


if __name__ == "__main__":
    main()
