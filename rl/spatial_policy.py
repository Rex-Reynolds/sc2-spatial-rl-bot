"""
Spatial CNN Policy for SC2 RL

AlphaStar-inspired architecture with:
- CNN encoders for screen and minimap
- MLP encoder for scalars
- LSTM for temporal reasoning
- Multiple action heads (action type + spatial locations)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class SpatialFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for spatial observations.

    Processes screen, minimap, and scalar features.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        # Calculate combined feature dimension
        super().__init__(observation_space, features_dim)

        # Extract dimensions from observation space
        screen_channels = observation_space['screen'].shape[0]  # 20
        minimap_channels = observation_space['minimap'].shape[0]  # 11
        scalar_dim = observation_space['scalars'].shape[0]  # ~100

        # Screen encoder (20, 64, 64) → 256
        self.screen_encoder = nn.Sequential(
            nn.Conv2d(screen_channels, 32, kernel_size=8, stride=4, padding=2),  # → 32x16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # → 64x8x8
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # → 64x8x8
            nn.ReLU(),
            nn.Flatten(),  # → 4096
            nn.Linear(4096, 256),
            nn.ReLU(),
        )

        # Minimap encoder (11, 64, 64) → 128
        self.minimap_encoder = nn.Sequential(
            nn.Conv2d(minimap_channels, 16, kernel_size=8, stride=4, padding=2),  # → 16x16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # → 32x8x8
            nn.ReLU(),
            nn.Flatten(),  # → 2048
            nn.Linear(2048, 128),
            nn.ReLU(),
        )

        # Scalar encoder (~100) → 128
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Combined dimension: 256 + 128 + 128 = 512
        self._features_dim = features_dim

        # Final combination layer
        self.combiner = nn.Sequential(
            nn.Linear(256 + 128 + 128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process observations through encoders.

        Args:
            observations: Dict with keys 'screen', 'minimap', 'scalars'

        Returns:
            Combined feature vector of shape (batch_size, features_dim)
        """
        screen = observations['screen']  # (batch, 20, 64, 64)
        minimap = observations['minimap']  # (batch, 11, 64, 64)
        scalars = observations['scalars']  # (batch, ~100)

        # Encode each modality
        screen_features = self.screen_encoder(screen)  # (batch, 256)
        minimap_features = self.minimap_encoder(minimap)  # (batch, 128)
        scalar_features = self.scalar_encoder(scalars)  # (batch, 128)

        # Concatenate and combine
        combined = torch.cat([screen_features, minimap_features, scalar_features], dim=1)
        features = self.combiner(combined)  # (batch, 512)

        return features


class SpatialActorCriticPolicy(nn.Module):
    """
    Custom Actor-Critic policy for spatial actions.

    Outputs:
      - Action type logits (discrete)
      - Screen location heatmap (spatial)
      - Minimap location heatmap (spatial)
      - Value estimate
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        num_action_types: int = 50,
        features_dim: int = 512,
        use_lstm: bool = True,
    ):
        super().__init__()

        self.observation_space = observation_space
        self.num_action_types = num_action_types
        self.use_lstm = use_lstm

        # Feature extractor
        self.features_extractor = SpatialFeaturesExtractor(
            observation_space, features_dim
        )

        # LSTM for temporal reasoning (optional but recommended)
        if use_lstm:
            self.lstm = nn.LSTM(features_dim, 256, num_layers=1, batch_first=True)
            policy_input_dim = 256
        else:
            policy_input_dim = features_dim

        # Action type head (discrete)
        self.action_type_head = nn.Sequential(
            nn.Linear(policy_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_action_types),
        )

        # Spatial heads (output 64x64 heatmaps)
        # Screen location head
        self.screen_location_head = nn.Sequential(
            nn.Linear(policy_input_dim, 1024),
            nn.ReLU(),
            nn.Unflatten(1, (16, 8, 8)),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # → 8x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),  # → 4x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),  # → 1x64x64
            nn.Flatten(start_dim=1),  # → 4096
        )

        # Minimap location head (same structure)
        self.minimap_location_head = nn.Sequential(
            nn.Linear(policy_input_dim, 512),
            nn.ReLU(),
            nn.Unflatten(1, (8, 8, 8)),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),  # → 4x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),  # → 1x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # → 1x64x64
            nn.Flatten(start_dim=1),  # → 4096
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(policy_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # LSTM hidden state (will be set during forward pass)
        self.lstm_hidden = None

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        lstm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the policy.

        Args:
            observations: Dict with 'screen', 'minimap', 'scalars'
            lstm_states: Optional LSTM hidden states (h, c)

        Returns:
            Dict containing:
              - action_type_logits: (batch, num_actions)
              - screen_logits: (batch, 64*64)
              - minimap_logits: (batch, 64*64)
              - value: (batch, 1)
              - lstm_states: Updated LSTM states (if using LSTM)
        """
        # Extract features
        features = self.features_extractor(observations)  # (batch, features_dim)

        # Apply LSTM if enabled
        if self.use_lstm:
            # Add sequence dimension for LSTM
            features_seq = features.unsqueeze(1)  # (batch, 1, features_dim)

            if lstm_states is not None:
                lstm_out, new_lstm_states = self.lstm(features_seq, lstm_states)
            else:
                lstm_out, new_lstm_states = self.lstm(features_seq)

            policy_features = lstm_out.squeeze(1)  # (batch, 256)
        else:
            policy_features = features
            new_lstm_states = None

        # Action heads
        action_type_logits = self.action_type_head(policy_features)  # (batch, num_actions)
        screen_logits = self.screen_location_head(policy_features)  # (batch, 4096)
        minimap_logits = self.minimap_location_head(policy_features)  # (batch, 4096)
        value = self.value_head(policy_features)  # (batch, 1)

        return {
            'action_type_logits': action_type_logits,
            'screen_logits': screen_logits,
            'minimap_logits': minimap_logits,
            'value': value,
            'lstm_states': new_lstm_states,
        }

    def get_action_and_value(
        self,
        observations: Dict[str, torch.Tensor],
        lstm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, value, log prob, and entropy.

        Used during training and inference.

        Args:
            observations: Observation dict
            lstm_states: Optional LSTM states
            action: Optional action dict (for log prob calculation)
            deterministic: If True, take argmax instead of sampling

        Returns:
            (action_dict, value, log_prob, entropy)
        """
        outputs = self.forward(observations, lstm_states)

        action_type_logits = outputs['action_type_logits']
        screen_logits = outputs['screen_logits']
        minimap_logits = outputs['minimap_logits']
        value = outputs['value']

        # Sample or take argmax
        action_type_dist = torch.distributions.Categorical(logits=action_type_logits)
        screen_dist = torch.distributions.Categorical(logits=screen_logits)
        minimap_dist = torch.distributions.Categorical(logits=minimap_logits)

        if action is None:
            # Sample new action
            if deterministic:
                action_type = action_type_logits.argmax(dim=1)
                screen_idx = screen_logits.argmax(dim=1)
                minimap_idx = minimap_logits.argmax(dim=1)
            else:
                action_type = action_type_dist.sample()
                screen_idx = screen_dist.sample()
                minimap_idx = minimap_dist.sample()

            action = {
                'action_type': action_type,
                'screen_idx': screen_idx,
                'minimap_idx': minimap_idx,
            }
        else:
            action_type = action['action_type']
            screen_idx = action['screen_idx']
            minimap_idx = action['minimap_idx']

        # Calculate log probabilities
        action_type_log_prob = action_type_dist.log_prob(action_type)
        screen_log_prob = screen_dist.log_prob(screen_idx)
        minimap_log_prob = minimap_dist.log_prob(minimap_idx)

        # Combined log prob (assume independence)
        log_prob = action_type_log_prob + screen_log_prob + minimap_log_prob

        # Entropy (for exploration bonus)
        entropy = action_type_dist.entropy() + screen_dist.entropy() + minimap_dist.entropy()

        return action, value, log_prob, entropy

    def get_value(
        self,
        observations: Dict[str, torch.Tensor],
        lstm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get value estimate only (for advantage calculation)."""
        outputs = self.forward(observations, lstm_states)
        return outputs['value']


def convert_spatial_idx_to_coords(idx: int, size: int = 64) -> Tuple[int, int]:
    """Convert flattened spatial index to (x, y) coordinates."""
    y = idx // size
    x = idx % size
    return x, y


def convert_coords_to_spatial_idx(x: int, y: int, size: int = 64) -> int:
    """Convert (x, y) coordinates to flattened spatial index."""
    return y * size + x
