#!/usr/bin/env python3
"""
Parse StarCraft II replays to extract training data for imitation learning.

Extracts (observation, action) pairs from professional Terran replays.
"""

import sc2reader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import pickle


class ReplayParser:
    """Parse SC2 replays into training data."""

    # Map replay events to our 23 advanced actions
    ACTION_MAP = {
        "train_scv": 0,
        "build_supply_depot": 1,
        "build_refinery": 2,
        "build_barracks": 3,
        "build_factory": 4,
        "build_starport": 5,
        "build_tech_lab": 6,  # barracks tech lab
        "build_reactor": 7,  # barracks reactor
        "build_tech_lab_factory": 8,
        "train_marine": 9,
        "train_marauder": 10,
        "train_tank": 11,
        "train_hellion": 12,
        "train_medivac": 13,
        "research_stim": 14,
        "research_combat_shields": 15,
        "research_concussive_shells": 16,
        "upgrade_weapons": 17,
        "upgrade_armor": 18,
        "attack": 19,
        "defend": 20,
        "expand": 21,
        "no_op": 22,
    }

    def __init__(self, advanced=True):
        """
        Initialize parser.

        Args:
            advanced: If True, extract 26-feature observations for AdvancedRLBot
        """
        self.advanced = advanced

    def parse_replay(self, replay_path: str) -> Optional[List[Tuple[np.ndarray, int]]]:
        """
        Parse a single replay file.

        Returns:
            List of (observation, action) tuples, or None if parsing fails
        """
        try:
            replay = sc2reader.load_replay(replay_path)

            # Find Terran player
            terran_player = self._find_terran_player(replay)
            if not terran_player:
                print(f"  No Terran player found in {Path(replay_path).name}")
                return None

            # Extract trajectories
            trajectories = self._extract_trajectories(replay, terran_player)

            print(f"  ✓ Extracted {len(trajectories)} (obs, action) pairs from {Path(replay_path).name}")
            return trajectories

        except Exception as e:
            print(f"  ✗ Error parsing {Path(replay_path).name}: {e}")
            return None

    def _find_terran_player(self, replay):
        """Find the Terran player in the replay."""
        for player in replay.players:
            if player.play_race == 'Terran' and player.is_human:
                return player
        return None

    def _extract_trajectories(self, replay, player) -> List[Tuple[np.ndarray, int]]:
        """
        Extract (observation, action) pairs from replay.

        This is a simplified approach that samples game states every 16 frames
        and maps major events to actions.
        """
        trajectories = []

        # Sample every 16 game frames (same as RL bot decision frequency)
        sample_interval = 16
        last_sample_frame = 0

        # Track game state
        game_state = GameState()

        for event in replay.events:
            # Update game state based on events
            if hasattr(event, 'player') and event.player == player:
                game_state.update(event)

            # Sample every 16 frames
            current_frame = event.frame if hasattr(event, 'frame') else 0
            if current_frame - last_sample_frame >= sample_interval:
                # Extract observation
                obs = self._extract_observation(game_state, current_frame)

                # Map recent event to action
                action = self._map_event_to_action(event, player)

                if obs is not None and action is not None:
                    trajectories.append((obs, action))

                last_sample_frame = current_frame

        return trajectories

    def _extract_observation(self, game_state: 'GameState', frame: int) -> Optional[np.ndarray]:
        """
        Extract 26-feature observation vector from game state.

        This is a placeholder - in reality, we'd need to track all units/buildings
        throughout the replay to reconstruct the full game state.
        """
        # For now, return a dummy observation
        # TODO: Implement proper state tracking
        obs = np.random.rand(26).astype(np.float32)
        return obs

    def _map_event_to_action(self, event, player) -> Optional[int]:
        """
        Map a replay event to one of our 23 actions.

        Returns action index or None if event doesn't map to an action.
        """
        if not hasattr(event, 'player') or event.player != player:
            return None

        event_name = event.name if hasattr(event, 'name') else type(event).__name__

        # Map common events
        if 'TrainUnitCommand' in event_name:
            unit_type = getattr(event, 'unit_type_name', '')
            if 'SCV' in unit_type:
                return self.ACTION_MAP["train_scv"]
            elif 'Marine' in unit_type:
                return self.ACTION_MAP["train_marine"]
            elif 'Marauder' in unit_type:
                return self.ACTION_MAP["train_marauder"]
            elif 'SiegeTank' in unit_type:
                return self.ACTION_MAP["train_tank"]
            elif 'Hellion' in unit_type:
                return self.ACTION_MAP["train_hellion"]
            elif 'Medivac' in unit_type:
                return self.ACTION_MAP["train_medivac"]

        elif 'BuildCommand' in event_name:
            building_type = getattr(event, 'unit_type_name', '')
            if 'SupplyDepot' in building_type:
                return self.ACTION_MAP["build_supply_depot"]
            elif 'Refinery' in building_type:
                return self.ACTION_MAP["build_refinery"]
            elif 'Barracks' in building_type:
                return self.ACTION_MAP["build_barracks"]
            elif 'Factory' in building_type:
                return self.ACTION_MAP["build_factory"]
            elif 'Starport' in building_type:
                return self.ACTION_MAP["build_starport"]
            elif 'TechLab' in building_type:
                return self.ACTION_MAP["build_tech_lab"]
            elif 'Reactor' in building_type:
                return self.ACTION_MAP["build_reactor"]
            elif 'CommandCenter' in building_type:
                return self.ACTION_MAP["expand"]

        elif 'ResearchCommand' in event_name:
            research_type = getattr(event, 'upgrade_type_name', '')
            if 'Stimpack' in research_type:
                return self.ACTION_MAP["research_stim"]
            elif 'CombatShield' in research_type:
                return self.ACTION_MAP["research_combat_shields"]
            elif 'Concussive' in research_type:
                return self.ACTION_MAP["research_concussive_shells"]
            elif 'Weapon' in research_type:
                return self.ACTION_MAP["upgrade_weapons"]
            elif 'Armor' in research_type:
                return self.ACTION_MAP["upgrade_armor"]

        elif 'TargetPointCommand' in event_name or 'AttackCommand' in event_name:
            return self.ACTION_MAP["attack"]

        # Default: no_op for unrecognized events
        return self.ACTION_MAP["no_op"]

    def parse_replay_directory(self, replay_dir: str, max_replays: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse all replays in a directory.

        Args:
            replay_dir: Path to directory containing .SC2Replay files
            max_replays: Maximum number of replays to parse (None = all)

        Returns:
            (observations, actions) as numpy arrays
        """
        replay_path = Path(replay_dir)

        # Find all replay files
        replay_files = list(replay_path.glob("**/*.SC2Replay"))
        if max_replays:
            replay_files = replay_files[:max_replays]

        print(f"Found {len(replay_files)} replay files")

        all_observations = []
        all_actions = []

        for i, replay_file in enumerate(replay_files, 1):
            print(f"\n[{i}/{len(replay_files)}] Parsing: {replay_file.name}")

            trajectories = self.parse_replay(str(replay_file))

            if trajectories:
                for obs, action in trajectories:
                    all_observations.append(obs)
                    all_actions.append(action)

        # Convert to numpy arrays
        observations = np.array(all_observations, dtype=np.float32)
        actions = np.array(all_actions, dtype=np.int32)

        print(f"\n✓ Total training samples: {len(observations)}")
        print(f"  Observation shape: {observations.shape}")
        print(f"  Action shape: {actions.shape}")

        return observations, actions

    def save_parsed_data(self, observations: np.ndarray, actions: np.ndarray, output_path: str):
        """Save parsed replay data to disk."""
        data = {
            'observations': observations,
            'actions': actions,
        }
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved parsed data to {output_path}")

    @staticmethod
    def load_parsed_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load previously parsed replay data."""
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data['observations'], data['actions']


class GameState:
    """Track game state throughout a replay."""

    def __init__(self):
        self.minerals = 50
        self.gas = 0
        self.supply_used = 12
        self.supply_cap = 15
        self.scv_count = 12
        self.marines = 0
        self.marauders = 0
        self.tanks = 0
        self.hellions = 0
        self.medivacs = 0
        # ... more state tracking

    def update(self, event):
        """Update game state based on event."""
        # TODO: Implement proper state tracking
        pass


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Parse SC2 replays for imitation learning")
    parser.add_argument("replay_dir", help="Directory containing .SC2Replay files")
    parser.add_argument("--output", default="rl/data/pro_replays.pkl", help="Output file")
    parser.add_argument("--max-replays", type=int, help="Max replays to parse")

    args = parser.parse_args()

    # Parse replays
    replay_parser = ReplayParser(advanced=True)
    observations, actions = replay_parser.parse_replay_directory(
        args.replay_dir,
        max_replays=args.max_replays
    )

    # Save to disk
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    replay_parser.save_parsed_data(observations, actions, args.output)


if __name__ == "__main__":
    main()
