#!/usr/bin/env python3
"""
Simplified test to verify basic game execution.

This tests if we can run a game with a minimal bot.
"""

from sc2 import maps
from sc2.main import run_game
from sc2.player import Bot
from sc2.data import Race
from sc2.bot_ai import BotAI

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bots import IdleBot


class MinimalRLBot(BotAI):
    """Minimal bot for testing - just does nothing."""

    async def on_start(self):
        print("MinimalRLBot: on_start called")

    async def on_step(self, iteration: int):
        print(f"MinimalRLBot: on_step {iteration}")
        if iteration > 100:  # End after 100 steps
            print("Ending game early")
            # Game will end naturally


def main():
    print("=" * 70)
    print("SIMPLE GAME TEST")
    print("=" * 70)
    print("Running a basic game with minimal bot...")
    print()

    try:
        result = run_game(
            maps.get("Simple64"),
            [
                Bot(Race.Terran, MinimalRLBot(), name="MinimalBot"),
                Bot(Race.Terran, IdleBot(), name="IdleBot"),
            ],
            realtime=False,
        )

        print(f"\nGame completed!")
        print(f"Result type: {type(result)}")
        print(f"Result value: {result}")
        print(f"Result[0]: {result[0] if result else 'None'}")

    except Exception as e:
        print(f"\nError running game: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
