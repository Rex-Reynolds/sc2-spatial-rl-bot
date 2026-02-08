#!/usr/bin/env python3
"""
Play Against Your Trained Bot

Run this to play a game as a human against your trained spatial bot.
"""

import sys
import torch
from sc2 import maps
from sc2.main import run_game
from sc2.player import Bot, Human
from sc2.data import Race, Difficulty

from rl.spatial_policy import SpatialActorCriticPolicy
from rl.spatial_bot import SpatialRLBot
from rl.spatial_env import SpatialSC2Env
from gymnasium import spaces
import numpy as np


def load_trained_model(model_path: str):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")

    # Create observation space (needed for policy)
    obs_space = spaces.Dict({
        'screen': spaces.Box(0, 1, (20, 64, 64), dtype=np.float32),
        'minimap': spaces.Box(0, 1, (11, 64, 64), dtype=np.float32),
        'scalars': spaces.Box(0, 1, (90,), dtype=np.float32),
    })

    # Create policy
    policy = SpatialActorCriticPolicy(obs_space, num_action_types=50, use_lstm=True)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()

    print("✓ Model loaded successfully!")
    return policy


class PlayableRLBot(SpatialRLBot):
    """Standalone version of spatial bot for playing games."""

    def __init__(self, policy):
        # Create minimal env for compatibility
        class FakeEnv:
            def __init__(self):
                self.policy = None
                self.episode_reward = 0
                self.trajectory = []

            def add_step_to_trajectory(self, *args):
                pass

        super().__init__(FakeEnv(), player_id=1)
        self.trained_policy = policy
        self.device = 'cpu'

    async def on_step(self, iteration: int):
        """Override to use trained policy."""
        if iteration % 16 != 0:
            await self.distribute_workers()
            return

        # Get observation
        obs = self._get_spatial_observation()

        # Convert to torch
        obs_torch = {
            'screen': torch.from_numpy(obs['screen']).unsqueeze(0),
            'minimap': torch.from_numpy(obs['minimap']).unsqueeze(0),
            'scalars': torch.from_numpy(obs['scalars']).unsqueeze(0),
        }

        # Get action from trained policy
        with torch.no_grad():
            action, _, _, _ = self.trained_policy.get_action_and_value(
                obs_torch, deterministic=False
            )

        # Convert to dict
        action_dict = {
            'action_type': action['action_type'].cpu().numpy()[0],
            'screen_idx': action['screen_idx'].cpu().numpy()[0],
            'minimap_idx': action['minimap_idx'].cpu().numpy()[0],
        }

        # Execute action
        await self._execute_spatial_action(action_dict)
        await self.distribute_workers()


def main():
    if len(sys.argv) < 2:
        print("Usage: python rl/play_vs_bot.py <model_path>")
        print("Example: python rl/play_vs_bot.py rl/models/spatial_100ep_extended/final_model.pt")
        sys.exit(1)

    model_path = sys.argv[1]

    # Load trained model
    policy = load_trained_model(model_path)

    # Create bot
    rl_bot = PlayableRLBot(policy)

    print("")
    print("=" * 70)
    print("PLAY VS YOUR TRAINED BOT")
    print("=" * 70)
    print("")
    print("Starting game...")
    print("You are Player 1 (Terran)")
    print("AI Bot is Player 2 (Terran)")
    print("")
    print("Good luck! 🎮")
    print("")

    # Run game: Human vs AI
    result = run_game(
        maps.get("Simple64"),
        [
            Human(Race.Terran),
            Bot(Race.Terran, rl_bot, name="SpatialRLBot"),
        ],
        realtime=True,  # Real-time for human play
    )

    print("")
    print("=" * 70)
    print("GAME OVER")
    print("=" * 70)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
