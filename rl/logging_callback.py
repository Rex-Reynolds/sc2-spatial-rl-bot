"""
Custom logging callback for TensorBoard metrics.

Fixes the issue where PPO doesn't write metrics properly with our
trajectory-based environment.
"""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class TensorBoardLoggingCallback(BaseCallback):
    """
    Custom callback to log training metrics to TensorBoard.

    Logs:
    - Episode rewards
    - Episode lengths
    - Win rate
    - Average reward per episode
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_results = []  # Victory/Defeat
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        """Called at each step."""
        # Accumulate reward
        if "rewards" in self.locals:
            reward = self.locals["rewards"][0] if len(self.locals["rewards"]) > 0 else 0
            self.current_episode_reward += reward
            self.current_episode_length += 1

        # Check if episode is done
        if "dones" in self.locals and len(self.locals["dones"]) > 0:
            done = self.locals["dones"][0]

            if done:
                # Log episode metrics
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)

                # Try to get result from info
                if "infos" in self.locals and len(self.locals["infos"]) > 0:
                    info = self.locals["infos"][0]
                    result = info.get("result")
                    if result:
                        from sc2.data import Result
                        is_win = (result == Result.Victory)
                        self.episode_results.append(1.0 if is_win else 0.0)

                # Log to TensorBoard
                self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards[-100:]))
                self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths[-100:]))

                if len(self.episode_results) > 0:
                    win_rate = np.mean(self.episode_results[-100:]) * 100
                    self.logger.record("rollout/win_rate", win_rate)

                self.logger.record("rollout/episodes", len(self.episode_rewards))

                if self.verbose > 0:
                    print(f"\nEpisode {len(self.episode_rewards)}: "
                          f"Reward={self.current_episode_reward:.2f}, "
                          f"Length={self.current_episode_length}")

                # Reset for next episode
                self.current_episode_reward = 0
                self.current_episode_length = 0

        return True

    def _on_training_start(self) -> None:
        """Called before training starts."""
        if self.verbose > 0:
            print("Starting training with TensorBoard logging...")

    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.verbose > 0:
            print(f"\nTraining complete!")
            print(f"Total episodes: {len(self.episode_rewards)}")
            if len(self.episode_rewards) > 0:
                print(f"Average reward: {np.mean(self.episode_rewards):.2f}")
            if len(self.episode_results) > 0:
                print(f"Win rate: {np.mean(self.episode_results) * 100:.1f}%")
