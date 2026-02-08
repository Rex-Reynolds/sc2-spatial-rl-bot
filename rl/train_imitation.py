#!/usr/bin/env python3
"""
Train a bot using imitation learning from professional replays.

Uses behavioral cloning to learn from expert demonstrations.
"""

import argparse
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rl.replay_parser import ReplayParser
from rl.env import make_env


class ImitationLearner:
    """Train a policy using behavioral cloning from expert data."""

    def __init__(self, observation_dim=26, action_dim=23, lr=0.001):
        """
        Initialize imitation learner.

        Args:
            observation_dim: Size of observation vector
            action_dim: Number of discrete actions
            lr: Learning rate
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.lr = lr

        # Create a dummy environment just to get the spaces
        env = make_env(advanced=True)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Create policy network
        self.policy = ActorCriticPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=lambda _: lr,
        )

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def train(self, expert_obs: np.ndarray, expert_actions: np.ndarray,
              batch_size=64, n_epochs=50, validation_split=0.1):
        """
        Train policy using behavioral cloning.

        Args:
            expert_obs: Expert observations (N, obs_dim)
            expert_actions: Expert actions (N,)
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            validation_split: Fraction of data for validation
        """
        print("=" * 70)
        print("IMITATION LEARNING TRAINING")
        print("=" * 70)
        print(f"Training samples: {len(expert_obs)}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {n_epochs}")
        print("=" * 70)

        # Split into train/validation
        n_val = int(len(expert_obs) * validation_split)
        indices = np.random.permutation(len(expert_obs))

        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        train_obs = expert_obs[train_indices]
        train_actions = expert_actions[train_indices]
        val_obs = expert_obs[val_indices]
        val_actions = expert_actions[val_indices]

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_obs),
            torch.LongTensor(train_actions)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.FloatTensor(val_obs),
            torch.LongTensor(val_actions)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Training loop
        best_val_acc = 0.0

        for epoch in range(n_epochs):
            # Train
            self.policy.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_obs, batch_actions in train_loader:
                self.optimizer.zero_grad()

                # Forward pass
                distribution = self.policy.get_distribution(batch_obs)
                action_log_probs = distribution.distribution.logits

                # Cross-entropy loss
                loss = F.cross_entropy(action_log_probs, batch_actions)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Track metrics
                train_loss += loss.item()
                predictions = action_log_probs.argmax(dim=1)
                train_correct += (predictions == batch_actions).sum().item()
                train_total += len(batch_actions)

            # Validation
            self.policy.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_obs, batch_actions in val_loader:
                    distribution = self.policy.get_distribution(batch_obs)
                    action_log_probs = distribution.distribution.logits

                    loss = F.cross_entropy(action_log_probs, batch_actions)

                    val_loss += loss.item()
                    predictions = action_log_probs.argmax(dim=1)
                    val_correct += (predictions == batch_actions).sum().item()
                    val_total += len(batch_actions)

            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {avg_val_loss:.4f}  |  Val Acc:   {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"  ✓ New best validation accuracy: {val_acc:.2f}%")

        print("\n" + "=" * 70)
        print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
        print("=" * 70)

    def save(self, path: str):
        """Save the trained policy."""
        # Save using SB3 format for compatibility
        # We'll wrap it in a PPO object
        env = make_env(advanced=True)
        model = PPO("MlpPolicy", env)

        # Replace the policy with our trained one
        model.policy = self.policy

        model.save(path)
        print(f"✓ Saved imitation model to {path}")

    @staticmethod
    def load(path: str):
        """Load a trained imitation model."""
        model = PPO.load(path)
        return model.policy


def main():
    parser = argparse.ArgumentParser(description="Train bot using imitation learning")
    parser.add_argument(
        "--data",
        default="rl/data/pro_replays.pkl",
        help="Path to parsed replay data (.pkl file)"
    )
    parser.add_argument(
        "--output",
        default="rl/models/pro_imitation",
        help="Output model path"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )

    args = parser.parse_args()

    # Check if data exists
    if not Path(args.data).exists():
        print(f"Error: Data file not found: {args.data}")
        print("\nFirst parse replays using:")
        print(f"  python rl/replay_parser.py <replay_directory> --output {args.data}")
        return

    # Load expert data
    print(f"Loading expert data from {args.data}...")
    expert_obs, expert_actions = ReplayParser.load_parsed_data(args.data)

    print(f"✓ Loaded {len(expert_obs)} expert demonstrations")
    print(f"  Observation shape: {expert_obs.shape}")
    print(f"  Action shape: {expert_actions.shape}")

    # Train imitation model
    learner = ImitationLearner(
        observation_dim=expert_obs.shape[1],
        action_dim=23,
        lr=args.lr
    )

    learner.train(
        expert_obs,
        expert_actions,
        batch_size=args.batch_size,
        n_epochs=args.epochs
    )

    # Save model
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    learner.save(args.output)

    print("\n✓ Imitation learning complete!")
    print(f"\nTo train RL agent against this bot:")
    print(f"  python rl/train.py --advanced --self-play --opponent-model {args.output} --episodes 100")


if __name__ == "__main__":
    main()
