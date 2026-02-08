"""
Spatial SC2 Environment for RL Training

Uses spatial observations (feature maps) and spatial actions.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

from sc2 import maps
from sc2.main import run_game
from sc2.player import Bot
from sc2.data import Race, Result

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.spatial_features import SpatialFeatureExtractor


class SpatialSC2Env(gym.Env):
    """
    Spatial SC2 environment with feature maps.

    Observation Space:
      - screen: (20, 64, 64) float32
      - minimap: (11, 64, 64) float32
      - scalars: (90,) float32

    Action Space:
      Dict with:
        - action_type: Discrete(50)
        - screen_idx: Discrete(64*64)
        - minimap_idx: Discrete(64*64)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        opponent="IdleBot",
        opponent_policy=None,
        map_name="Simple64",
        max_game_time=600,
        realtime=False,
        step_interval=8,  # Faster for micro
        num_action_types=50,
    ):
        super().__init__()

        self.opponent_name = opponent
        self.opponent_policy = opponent_policy
        self.map_name = map_name
        self.max_game_time = max_game_time
        self.realtime = realtime
        self.step_interval = step_interval
        self.num_action_types = num_action_types

        # Spatial observation space
        self.observation_space = spaces.Dict({
            'screen': spaces.Box(
                low=0.0, high=1.0,
                shape=(20, 64, 64),
                dtype=np.float32
            ),
            'minimap': spaces.Box(
                low=0.0, high=1.0,
                shape=(11, 64, 64),
                dtype=np.float32
            ),
            'scalars': spaces.Box(
                low=0.0, high=1.0,
                shape=(90,),
                dtype=np.float32
            ),
        })

        # Spatial action space
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(num_action_types),
            'screen_idx': spaces.Discrete(64 * 64),  # Flattened screen position
            'minimap_idx': spaces.Discrete(64 * 64),  # Flattened minimap position
        })

        # Episode data
        self.trajectory: List[Tuple] = []
        self.trajectory_idx = 0
        self.game_result = None
        self.episode_reward = 0.0

        # Policy function (set by training loop)
        self.policy = None

        # Feature extractor
        self.feature_extractor = SpatialFeatureExtractor()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment by running a complete game."""
        super().reset(seed=seed)

        # Clear previous trajectory
        self.trajectory = []
        self.trajectory_idx = 0
        self.game_result = None
        self.episode_reward = 0.0

        # Run complete game
        print(f"\nRunning spatial game: RLAgent vs {self.opponent_name}")
        try:
            self._run_complete_game()
        except KeyboardInterrupt:
            print("Training interrupted!")
            raise

        # Return first observation
        if self.trajectory:
            obs, _, _, _, info = self.trajectory[0]
            return obs, info
        else:
            return self._get_default_observation(), {}

    def step(
        self, action: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step by replaying from trajectory.

        Action is ignored - actions were chosen during game run.
        """
        if self.trajectory_idx >= len(self.trajectory):
            obs = self._get_default_observation()
            return obs, 0.0, True, False, {"result": self.game_result}

        obs, recorded_action, reward, done, info = self.trajectory[self.trajectory_idx]
        self.trajectory_idx += 1

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def _run_complete_game(self):
        """Run a complete game and populate trajectory."""
        try:
            from rl.spatial_bot import SpatialRLBot

            # Create RL bot
            rl_bot = SpatialRLBot(self, player_id=1)

            # Create opponent
            if self.opponent_policy is not None:
                # Self-play mode
                print(f"Running self-play: SpatialRLAgent vs SpatialRLAgent2")
                opponent_bot = SpatialRLBot(self, player_id=2, policy=self.opponent_policy)
                opponent_name = "SpatialRLAgent2"
            else:
                # Scripted opponent
                from bots import IdleBot, RushBot, DefenseBot
                opponent_classes = {
                    "IdleBot": IdleBot,
                    "RushBot": RushBot,
                    "DefenseBot": DefenseBot,
                }
                opponent_class = opponent_classes.get(self.opponent_name, IdleBot)
                opponent_bot = opponent_class()
                opponent_name = self.opponent_name

            # Run game
            result = run_game(
                maps.get(self.map_name),
                [
                    Bot(Race.Terran, rl_bot, name="SpatialRLAgent"),
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
            print("\n\nTraining interrupted by user (Ctrl+C)")
            self.game_result = Result.Defeat
            raise
        except Exception as e:
            print(f"Game error: {e}")
            import traceback
            traceback.print_exc()
            self.game_result = Result.Defeat

    def add_step_to_trajectory(
        self,
        obs: Dict[str, np.ndarray],
        action: Dict[str, int],
        reward: float,
        done: bool,
        info: Dict
    ):
        """Called by bot to add step to trajectory."""
        self.episode_reward += reward
        self.trajectory.append((obs, action, reward, done, info))

    def _get_default_observation(self) -> Dict[str, np.ndarray]:
        """Get default observation (zeros)."""
        return {
            'screen': np.zeros((20, 64, 64), dtype=np.float32),
            'minimap': np.zeros((11, 64, 64), dtype=np.float32),
            'scalars': np.zeros((90,), dtype=np.float32),
        }

    def render(self):
        """Render environment."""
        pass

    def close(self):
        """Clean up resources."""
        pass


def make_spatial_env(opponent="IdleBot", opponent_policy=None, **kwargs):
    """Factory function to create spatial SC2 environment."""
    return SpatialSC2Env(opponent=opponent, opponent_policy=opponent_policy, **kwargs)
