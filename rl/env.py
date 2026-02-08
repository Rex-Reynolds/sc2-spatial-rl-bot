"""
Gymnasium environment wrapper for StarCraft II bot training.

Simplified architecture: Run complete games and collect trajectories.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
import time

from sc2 import maps
from sc2.main import run_game
from sc2.player import Bot
from sc2.data import Race, Result

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bots import IdleBot, RushBot, DefenseBot, MarineMedivacBot


class SC2Env(gym.Env):
    """
    StarCraft II Gymnasium environment for RL training.

    Simplified approach: Each reset() runs a complete game, collecting
    a trajectory. step() replays the trajectory for compatibility with SB3.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        opponent="IdleBot",
        opponent_policy=None,  # For self-play: pass a policy function
        map_name="Simple64",
        max_game_time=600,
        realtime=False,
        step_interval=16,
        advanced=False,  # Use AdvancedRLBot with expanded action/obs space
        use_improved_rewards=False,  # Use ImprovedRLBot with better reward shaping
    ):
        super().__init__()

        self.opponent_name = opponent
        self.opponent_class = self._get_opponent_class(opponent)
        self.opponent_policy = opponent_policy  # Policy for opponent (if RL)
        self.map_name = map_name
        self.max_game_time = max_game_time
        self.realtime = realtime
        self.step_interval = step_interval
        self.advanced = advanced
        self.use_improved_rewards = use_improved_rewards

        # Observation and action space depend on bot type
        if advanced:
            # Advanced bot: 26 observations, 23 actions
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(26,),
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(23)
        else:
            # Basic bot: 11 observations, 7 actions
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(11,),
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(7)

        # Episode data (trajectory)
        self.trajectory: List[Tuple] = []  # (obs, action, reward, done, info)
        self.trajectory_idx = 0
        self.game_result = None
        self.episode_reward = 0.0  # Track total episode reward

        # Policy function (set by training loop) - for player 1
        self.policy = None

    def _get_opponent_class(self, opponent: str):
        """Get opponent bot class by name."""
        opponents = {
            "IdleBot": IdleBot,
            "RushBot": RushBot,
            "DefenseBot": DefenseBot,
            "MarineMedivacBot": MarineMedivacBot,
        }
        return opponents.get(opponent, IdleBot)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment by running a complete game.

        The bot will use self.policy to choose actions during the game.
        """
        super().reset(seed=seed)

        # Clear previous trajectory
        self.trajectory = []
        self.trajectory_idx = 0
        self.game_result = None
        self.episode_reward = 0.0

        # Run complete game and collect trajectory
        print(f"\nRunning game: RLAgent vs {self.opponent_name}")
        try:
            self._run_complete_game()
        except KeyboardInterrupt:
            # Propagate KeyboardInterrupt to stop training
            print("Training interrupted!")
            raise

        # Return first observation
        if self.trajectory:
            obs, _, _, _, info = self.trajectory[0]
            return obs, info
        else:
            # Game failed, return default
            return self._get_default_observation(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step by replaying from the trajectory.

        The action parameter is ignored - actions were already chosen
        during the game run in reset().
        """
        if self.trajectory_idx >= len(self.trajectory):
            # Trajectory exhausted
            obs = self._get_default_observation()
            return obs, 0.0, True, False, {"result": self.game_result}

        # Get next step from trajectory
        obs, recorded_action, reward, done, info = self.trajectory[self.trajectory_idx]
        self.trajectory_idx += 1

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def _run_complete_game(self):
        """Run a complete game and populate the trajectory."""
        try:
            # Choose bot class based on flags
            if self.advanced:
                if self.use_improved_rewards:
                    from rl.improved_rl_bot import ImprovedRLBot
                    BotClass = ImprovedRLBot
                    bot_name = "ImprovedRLAgent"
                else:
                    from rl.advanced_rl_bot import AdvancedRLBot
                    BotClass = AdvancedRLBot
                    bot_name = "AdvancedRLAgent"
            else:
                from rl.rl_bot import RLBot
                BotClass = RLBot
                bot_name = "RLAgent"

            # Create bot that will collect the trajectory
            rl_bot = BotClass(self, player_id=1)

            # Create opponent - either RL or scripted
            if self.opponent_policy is not None:
                # Self-play mode: opponent is also an RL agent
                print(f"Running self-play: {bot_name} vs {bot_name}2")
                opponent_bot = BotClass(self, player_id=2, policy=self.opponent_policy)
                opponent_name = f"{bot_name}2"
            else:
                # Standard mode: opponent is scripted bot
                opponent_bot = self.opponent_class()
                opponent_name = self.opponent_name

            # Run the game
            result = run_game(
                maps.get(self.map_name),
                [
                    Bot(Race.Terran, rl_bot, name=bot_name),
                    Bot(Race.Terran, opponent_bot, name=opponent_name),
                ],
                realtime=self.realtime,
            )

            if result and len(result) > 0:
                self.game_result = result[0]
                print(f"Game result: {self.game_result}")
            else:
                print(f"Warning: Invalid game result")
                self.game_result = Result.Defeat

        except KeyboardInterrupt:
            # Don't catch KeyboardInterrupt - let it propagate!
            print("\n\nTraining interrupted by user (Ctrl+C)")
            self.game_result = Result.Defeat
            raise  # Re-raise to stop training
        except Exception as e:
            print(f"Game error: {e}")
            import traceback
            traceback.print_exc()
            self.game_result = Result.Defeat

    def add_step_to_trajectory(
        self, obs: np.ndarray, action: int, reward: float, done: bool, info: Dict
    ):
        """Called by bot to add a step to the trajectory."""
        self.episode_reward += reward
        self.trajectory.append((obs, action, reward, done, info))

    def _get_default_observation(self) -> np.ndarray:
        """Get default observation."""
        size = 26 if self.advanced else 11
        return np.array([0.0] * size, dtype=np.float32)

    def render(self):
        """Render the environment."""
        pass

    def close(self):
        """Clean up resources."""
        pass


def make_env(opponent="IdleBot", opponent_policy=None, advanced=False, use_improved_rewards=False, **kwargs):
    """Factory function to create SC2 environment."""
    return SC2Env(
        opponent=opponent,
        opponent_policy=opponent_policy,
        advanced=advanced,
        use_improved_rewards=use_improved_rewards,
        **kwargs
    )
