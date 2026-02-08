#!/usr/bin/env python3
"""
Train Spatial RL Bot

Custom training loop for spatial CNN policy with multi-headed actions.

Based on PPO algorithm but adapted for:
- Dict observation space (screen + minimap + scalars)
- Dict action space (action_type + screen_idx + minimap_idx)
- Multi-headed policy (3 separate action heads)
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from rl.spatial_env import SpatialSC2Env
from rl.spatial_policy import SpatialActorCriticPolicy


def convert_obs_to_torch(obs_dict, device):
    """Convert numpy dict observation to torch tensors."""
    return {
        'screen': torch.from_numpy(obs_dict['screen']).unsqueeze(0).to(device),
        'minimap': torch.from_numpy(obs_dict['minimap']).unsqueeze(0).to(device),
        'scalars': torch.from_numpy(obs_dict['scalars']).unsqueeze(0).to(device),
    }


def train_spatial_bot(args):
    """Train spatial bot with custom PPO loop."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Create directories
    model_dir = Path(f"rl/models/{args.model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(f"rl/logs/{args.model_name}")
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir) if not args.no_tensorboard else None

    # Create environment
    print(f"Creating environment (opponent: {args.opponent})...")
    env = SpatialSC2Env(
        opponent=args.opponent,
        opponent_policy=None,  # TODO: Add self-play support
        map_name="Simple64",
        realtime=False,
        step_interval=args.step_interval,
    )

    # Create policy
    print("Creating spatial CNN policy...")
    policy = SpatialActorCriticPolicy(
        observation_space=env.observation_space,
        num_action_types=50,
        use_lstm=args.use_lstm,
    ).to(device)

    # Load existing model if specified
    if args.load_model:
        print(f"Loading model from: {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print("✓ Model loaded")

    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    # Set policy function for environment
    def policy_fn(obs_dict):
        """Policy function for environment to use during games."""
        with torch.no_grad():
            obs_torch = convert_obs_to_torch(obs_dict, device)
            action, value, log_prob, entropy = policy.get_action_and_value(
                obs_torch, deterministic=False
            )
            # Convert to dict with numpy values
            return {
                'action_type': action['action_type'].cpu().numpy()[0],
                'screen_idx': action['screen_idx'].cpu().numpy()[0],
                'minimap_idx': action['minimap_idx'].cpu().numpy()[0],
            }

    env.policy = policy_fn

    # Training loop
    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Episodes: {args.episodes}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Opponent: {args.opponent}")
    print("=" * 70)
    print()

    global_step = 0
    episode_rewards = []

    for episode in range(args.episodes):
        episode_start_time = time.time()

        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{args.episodes}")
        print(f"{'='*70}")

        # Run one game (collect trajectory)
        obs, info = env.reset()

        # Check if trajectory was collected
        if len(env.trajectory) == 0:
            print("⚠️  No trajectory collected, skipping episode")
            continue

        episode_reward = env.episode_reward
        episode_length = len(env.trajectory)
        episode_rewards.append(episode_reward)

        print(f"Game complete: {env.game_result}")
        print(f"Episode reward: {episode_reward:.2f}")
        print(f"Episode length: {episode_length} steps")

        # Extract trajectory data
        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []

        print("Processing trajectory...")
        for obs_t, action_t, reward_t, done_t, info_t in env.trajectory:
            obs_torch = convert_obs_to_torch(obs_t, device)

            # Get value and log prob for this transition
            with torch.no_grad():
                action_dict = {
                    'action_type': torch.tensor([action_t['action_type']]).to(device),
                    'screen_idx': torch.tensor([action_t['screen_idx']]).to(device),
                    'minimap_idx': torch.tensor([action_t['minimap_idx']]).to(device),
                }
                _, value, log_prob, _ = policy.get_action_and_value(
                    obs_torch, action=action_dict
                )

            observations.append(obs_torch)
            actions.append(action_dict)
            rewards.append(reward_t)
            values.append(value.item())
            log_probs.append(log_prob.item())

        # Calculate returns and advantages
        returns = []
        advantages = []
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            next_value = rewards[t] + args.gamma * next_value
            returns.insert(0, next_value)
            advantages.insert(0, next_value - values[t])

        # Normalize advantages
        advantages = np.array(advantages)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training update
        print("Performing PPO update...")
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)

        # PPO epochs
        for epoch in range(args.ppo_epochs):
            # Forward pass through policy with all transitions
            all_log_probs = []
            all_values = []
            all_entropies = []

            for i in range(len(observations)):
                _, value, log_prob, entropy = policy.get_action_and_value(
                    observations[i], action=actions[i]
                )
                all_log_probs.append(log_prob)
                all_values.append(value.squeeze())
                all_entropies.append(entropy)

            # Stack tensors
            new_log_probs = torch.stack(all_log_probs)
            new_values = torch.stack(all_values)
            new_entropies = torch.stack(all_entropies)

            old_log_probs = torch.tensor(log_probs, dtype=torch.float32).to(device)

            # PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)

            policy_loss = -torch.min(
                ratio * advantages_tensor,
                clipped_ratio * advantages_tensor
            ).mean()

            value_loss = 0.5 * ((new_values - returns_tensor) ** 2).mean()

            entropy_loss = -new_entropies.mean()

            loss = policy_loss + args.vf_coef * value_loss + args.ent_coef * entropy_loss

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

        global_step += episode_length

        # Logging
        elapsed_time = time.time() - episode_start_time
        print(f"\nUpdate complete:")
        print(f"  Policy loss: {policy_loss.item():.4f}")
        print(f"  Value loss: {value_loss.item():.4f}")
        print(f"  Entropy: {-entropy_loss.item():.4f}")
        print(f"  Episode time: {elapsed_time:.1f}s")

        if writer:
            writer.add_scalar("charts/episode_reward", episode_reward, episode)
            writer.add_scalar("charts/episode_length", episode_length, episode)
            writer.add_scalar("losses/policy_loss", policy_loss.item(), episode)
            writer.add_scalar("losses/value_loss", value_loss.item(), episode)
            writer.add_scalar("losses/entropy", -entropy_loss.item(), episode)
            writer.add_scalar("charts/global_step", global_step, episode)

        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = model_dir / f"checkpoint_ep{episode+1}.pt"
            torch.save({
                'episode': episode + 1,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_reward': episode_reward,
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_path = model_dir / "final_model.pt"
    torch.save({
        'episode': args.episodes,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"\n✓ Final model saved: {final_path}")

    if writer:
        writer.close()

    # Summary
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total episodes: {args.episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Final model: {final_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train spatial SC2 RL bot")
    parser.add_argument("--opponent", default="IdleBot", help="Opponent bot")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--model-name", default="spatial_test", help="Model name")
    parser.add_argument("--load-model", default=None, help="Load existing model")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--step-interval", type=int, default=16, help="Frames between decisions")
    parser.add_argument("--save-freq", type=int, default=5, help="Save checkpoint every N episodes")
    parser.add_argument("--use-lstm", action="store_true", help="Use LSTM in policy")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard")

    args = parser.parse_args()

    train_spatial_bot(args)


if __name__ == "__main__":
    main()
