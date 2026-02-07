"""
Gymnasium environment wrapper for StarCraft II bot training.

Bridges async burnysc2 API with synchronous Gymnasium interface.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import asyncio
from typing import Optional, Tuple, Dict, Any
from threading import Thread, Event
from queue import Queue

from sc2 import maps
from sc2.main import run_game
from sc2.player import Bot
from sc2.data import Race, Result

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bots import IdleBot, RushBot, DefenseBot


class SC2Env(gym.Env):
    """
    StarCraft II Gymnasium environment for RL training.

    Observation Space (11 features):
    - minerals (normalized 0-1)
    - vespene gas (normalized 0-1)
    - supply_used (normalized 0-1)
    - supply_cap (normalized 0-1)
    - scv_count (normalized 0-1)
    - marine_count (normalized 0-1)
    - barracks_count (normalized 0-1)
    - enemy_visible_units (normalized 0-1)
    - enemy_visible_structures (normalized 0-1)
    - game_time (normalized 0-1, max 10 min)
    - army_vs_enemy_strength (normalized -1 to 1)

    Action Space (7 discrete actions):
    0: train_scv
    1: build_supply_depot
    2: build_barracks
    3: train_marine
    4: attack
    5: defend
    6: no_op
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        opponent="IdleBot",
        map_name="Simple64",
        max_game_time=600,
        realtime=False,
        step_interval=16,  # Game frames between RL decisions
    ):
        super().__init__()

        self.opponent_name = opponent
        self.opponent_class = self._get_opponent_class(opponent)
        self.map_name = map_name
        self.max_game_time = max_game_time
        self.realtime = realtime
        self.step_interval = step_interval

        # Observation space: 11 continuous features [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(11,),
            dtype=np.float32,
        )

        # Action space: 7 discrete actions
        self.action_space = spaces.Discrete(7)

        # State tracking
        self.rl_bot = None
        self.current_obs = None
        self.game_result = None
        self.episode_reward = 0.0
        self.step_count = 0

        # Game loop management
        self.game_thread = None
        self.action_queue = Queue()
        self.obs_queue = Queue()
        self.game_done = Event()
        self._episode_active = False

    def _get_opponent_class(self, opponent: str):
        """Get opponent bot class by name."""
        opponents = {
            "IdleBot": IdleBot,
            "RushBot": RushBot,
            "DefenseBot": DefenseBot,
        }
        return opponents.get(opponent, IdleBot)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Returns:
            observation: Initial observation
            info: Additional info dict
        """
        super().reset(seed=seed)

        # Stop previous game if running
        if self._episode_active:
            self.game_done.set()
            if self.game_thread and self.game_thread.is_alive():
                self.game_thread.join(timeout=5.0)

        # Reset tracking
        self.episode_reward = 0.0
        self.step_count = 0
        self.game_result = None
        self.game_done.clear()

        # Clear queues
        while not self.action_queue.empty():
            self.action_queue.get()
        while not self.obs_queue.empty():
            self.obs_queue.get()

        # Create RL bot instance (will be controlled by agent)
        from rl.rl_bot import RLBot
        self.rl_bot = RLBot(self)

        # Start game in background thread
        self._episode_active = True
        self.game_thread = Thread(target=self._run_game_async, daemon=True)
        self.game_thread.start()

        # Wait for initial observation from game
        self.current_obs = self.obs_queue.get(timeout=30.0)

        return self.current_obs, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.

        Args:
            action: Integer action [0-6]

        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode is done (win/loss)
            truncated: Whether episode was truncated (time limit)
            info: Additional info
        """
        self.step_count += 1

        # Send action to game thread
        self.action_queue.put(action)

        # Wait for next observation from game
        try:
            self.current_obs = self.obs_queue.get(timeout=30.0)
        except:
            # Game ended or timeout
            self.current_obs = self._get_default_observation()

        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward

        # Check if episode is done
        terminated = self.game_result is not None or self.game_done.is_set()
        truncated = self.step_count >= 1000  # Max steps per episode

        if terminated or truncated:
            self._episode_active = False

        info = {
            "episode_reward": self.episode_reward,
            "step_count": self.step_count,
            "result": self.game_result,
        }

        return self.current_obs, reward, terminated, truncated, info

    def _get_default_observation(self) -> np.ndarray:
        """Get default observation at start."""
        return np.array([0.0] * 11, dtype=np.float32)

    def _calculate_reward(self) -> float:
        """
        Calculate reward for current step.

        Reward components:
        - Win: +10
        - Loss: -10
        - Enemy unit killed: +0.1 per unit
        - Own unit lost: -0.05 per unit
        - Economy growth: +0.01 per mineral/gas collected
        """
        if not self.rl_bot:
            return 0.0

        bot = self.rl_bot
        reward = 0.0

        # Win/loss reward
        if self.game_result == Result.Victory:
            reward += 10.0
        elif self.game_result == Result.Defeat:
            reward -= 10.0

        # Unit kill/loss reward
        if hasattr(bot, "reward_tracker"):
            tracker = bot.reward_tracker
            reward += tracker.get("enemy_units_killed", 0) * 0.1
            reward -= tracker.get("own_units_lost", 0) * 0.05
            reward += tracker.get("minerals_collected", 0) * 0.0001
            reward += tracker.get("gas_collected", 0) * 0.0002

        return reward

    def render(self):
        """Render the environment (game is already rendered if realtime=True)."""
        pass

    def close(self):
        """Clean up resources."""
        if self._episode_active:
            self.game_done.set()
            if self.game_thread and self.game_thread.is_alive():
                self.game_thread.join(timeout=5.0)
        self._episode_active = False

    def _run_game_async(self):
        """Run SC2 game in background thread."""
        try:
            # Run the game synchronously in this thread
            asyncio.run(self._run_game())
        except Exception as e:
            print(f"Game error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.game_done.set()
            self._episode_active = False

    async def _run_game(self):
        """Actually run the SC2 game."""
        result = await run_game(
            maps.get(self.map_name),
            [
                Bot(Race.Terran, self.rl_bot, name="RLAgent"),
                Bot(Race.Terran, self.opponent_class(), name=self.opponent_name),
            ],
            realtime=self.realtime,
        )

        # Store final result
        self.game_result = result[0]  # Result for our bot


def make_env(opponent="IdleBot", **kwargs):
    """Factory function to create SC2 environment."""
    return SC2Env(opponent=opponent, **kwargs)
