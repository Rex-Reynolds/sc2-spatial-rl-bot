"""
Spatial Feature Extraction for SC2

Converts game state into feature maps (screen + minimap) for CNN processing.
Based on AlphaStar and DeepMind's SC2LE architecture.
"""

import numpy as np
from typing import Tuple, List
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2


class SpatialFeatureExtractor:
    """
    Extract spatial features from SC2 game state.

    Outputs:
      - Screen: 64x64 x 20 channels
      - Minimap: 64x64 x 11 channels
      - Scalars: 100+ features
    """

    SCREEN_SIZE = 64
    MINIMAP_SIZE = 64

    # Feature channels
    NUM_SCREEN_CHANNELS = 20
    NUM_MINIMAP_CHANNELS = 11

    def __init__(self):
        pass

    def extract_features(self, bot: BotAI) -> dict:
        """
        Extract all features from bot state.

        Returns:
            dict with keys: 'screen', 'minimap', 'scalars'
        """
        try:
            screen = self._extract_screen_features(bot)
            minimap = self._extract_minimap_features(bot)
            scalars = self._extract_scalar_features(bot)

            return {
                'screen': screen,      # (20, 64, 64)
                'minimap': minimap,    # (11, 64, 64)
                'scalars': scalars     # (N,)
            }
        except Exception as e:
            # Return zeros if extraction fails
            print(f"Feature extraction error: {e}")
            return {
                'screen': np.zeros((20, 64, 64), dtype=np.float32),
                'minimap': np.zeros((11, 64, 64), dtype=np.float32),
                'scalars': np.zeros((90,), dtype=np.float32),
            }

    def _extract_screen_features(self, bot: BotAI) -> np.ndarray:
        """
        Extract screen feature maps (20 channels).

        Screen is centered on player's base (can be adjusted).
        """
        screen = np.zeros((self.NUM_SCREEN_CHANNELS, self.SCREEN_SIZE, self.SCREEN_SIZE), dtype=np.float32)

        # Define screen bounds (centered on main base)
        camera_center = bot.start_location
        screen_radius = 40  # Game units

        # Channel 0: Player relative (self=1, enemy=2, neutral=3)
        # Channel 1: Unit type (encoded)
        # Channel 2: Selected units
        # Channel 3: Unit hit points (ratio)
        # Channel 4: Unit shields (ratio)
        # Channel 5: Unit energy (ratio)
        # Channel 6: Unit density
        # Channel 7: Friendly unit density
        # Channel 8: Enemy unit density
        # Channel 9: Height map
        # Channel 10: Visibility
        # Channel 11: Creep
        # Channel 12: Buildable
        # Channel 13: Pathable
        # Channel 14: Unit hit points (absolute, normalized)
        # Channel 15: Unit shields (absolute, normalized)
        # Channel 16: Unit energy (absolute, normalized)
        # Channel 17: Selected unit density
        # Channel 18: Cargo (units in transports)
        # Channel 19: Cargo size

        # Helper: Convert game position to screen coordinates
        def pos_to_screen(pos: Point2) -> Tuple[int, int]:
            """Convert game position to screen pixel."""
            rel_x = (pos.x - camera_center.x + screen_radius) / (2 * screen_radius)
            rel_y = (pos.y - camera_center.y + screen_radius) / (2 * screen_radius)

            pixel_x = int(rel_x * self.SCREEN_SIZE)
            pixel_y = int(rel_y * self.SCREEN_SIZE)

            # Clamp to valid range
            pixel_x = max(0, min(self.SCREEN_SIZE - 1, pixel_x))
            pixel_y = max(0, min(self.SCREEN_SIZE - 1, pixel_y))

            return pixel_x, pixel_y

        # Extract friendly units
        for unit in bot.units:
            x, y = pos_to_screen(unit.position)

            # Channel 0: Player relative
            screen[0, y, x] = 1.0  # Self

            # Channel 1: Unit type (encode as normalized ID)
            screen[1, y, x] = unit.type_id.value / 2000.0

            # Channel 3: HP ratio
            screen[3, y, x] = unit.health_percentage

            # Channel 4: Shield ratio
            if unit.shield_max > 0:
                screen[4, y, x] = unit.shield / unit.shield_max

            # Channel 5: Energy ratio
            if unit.energy_max > 0:
                screen[5, y, x] = unit.energy / unit.energy_max

            # Channel 7: Friendly unit density
            screen[7, y, x] += 0.1  # Accumulate density

        # Extract friendly structures
        for structure in bot.structures:
            x, y = pos_to_screen(structure.position)

            screen[0, y, x] = 1.0  # Self
            screen[1, y, x] = structure.type_id.value / 2000.0
            screen[3, y, x] = structure.health_percentage

            if structure.shield_max > 0:
                screen[4, y, x] = structure.shield / structure.shield_max

        # Extract enemy units
        for unit in bot.enemy_units:
            x, y = pos_to_screen(unit.position)

            # Channel 0: Player relative
            screen[0, y, x] = 2.0  # Enemy

            # Channel 1: Unit type
            screen[1, y, x] = unit.type_id.value / 2000.0

            # Channel 3: HP ratio (if visible)
            screen[3, y, x] = unit.health_percentage

            # Channel 8: Enemy unit density
            screen[8, y, x] += 0.1

        # Extract enemy structures
        for structure in bot.enemy_structures:
            x, y = pos_to_screen(structure.position)

            screen[0, y, x] = 2.0  # Enemy
            screen[1, y, x] = structure.type_id.value / 2000.0
            screen[3, y, x] = structure.health_percentage

        # Channel 6: Overall unit density
        screen[6] = screen[7] + screen[8]

        # Channel 9: Height map (simplified - use game terrain)
        map_width = bot.game_info.map_size[0]
        map_height = bot.game_info.map_size[1]

        for y_idx in range(self.SCREEN_SIZE):
            for x_idx in range(self.SCREEN_SIZE):
                # Convert screen coords back to game position
                game_x = camera_center.x - screen_radius + (x_idx / self.SCREEN_SIZE) * 2 * screen_radius
                game_y = camera_center.y - screen_radius + (y_idx / self.SCREEN_SIZE) * 2 * screen_radius

                game_pos = Point2((game_x, game_y))
                rounded_pos = game_pos.rounded

                # Check if position is within map bounds
                in_bounds = (0 <= rounded_pos[0] < map_width and
                           0 <= rounded_pos[1] < map_height)

                if in_bounds:
                    # Get height (0-255, normalize to 0-1)
                    height = bot.get_terrain_height(game_pos) / 255.0
                    screen[9, y_idx, x_idx] = height

                    # Channel 10: Visibility (1 if visible, 0 if fog)
                    is_visible = bot.state.visibility[rounded_pos] == 2
                    screen[10, y_idx, x_idx] = 1.0 if is_visible else 0.0

                    # Channel 12: Buildable (use placement grid, not async check)
                    buildable = bot.in_placement_grid(game_pos)
                    screen[12, y_idx, x_idx] = 1.0 if buildable else 0.0

                    # Channel 13: Pathable
                    pathable = bot.in_pathing_grid(game_pos)
                    screen[13, y_idx, x_idx] = 1.0 if pathable else 0.0
                else:
                    # Out of bounds - set to default values
                    screen[9, y_idx, x_idx] = 0.0   # Unknown height
                    screen[10, y_idx, x_idx] = 0.0  # Not visible
                    screen[12, y_idx, x_idx] = 0.0  # Not buildable
                    screen[13, y_idx, x_idx] = 0.0  # Not pathable

        # Normalize density channels
        screen[6] = np.clip(screen[6], 0, 1)
        screen[7] = np.clip(screen[7], 0, 1)
        screen[8] = np.clip(screen[8], 0, 1)

        return screen

    def _extract_minimap_features(self, bot: BotAI) -> np.ndarray:
        """
        Extract minimap feature maps (11 channels).

        Minimap covers entire game map.
        """
        minimap = np.zeros((self.NUM_MINIMAP_CHANNELS, self.MINIMAP_SIZE, self.MINIMAP_SIZE), dtype=np.float32)

        # Map dimensions
        map_width = bot.game_info.map_size[0]
        map_height = bot.game_info.map_size[1]

        def pos_to_minimap(pos: Point2) -> Tuple[int, int]:
            """Convert game position to minimap pixel."""
            pixel_x = int((pos.x / map_width) * self.MINIMAP_SIZE)
            pixel_y = int((pos.y / map_height) * self.MINIMAP_SIZE)

            pixel_x = max(0, min(self.MINIMAP_SIZE - 1, pixel_x))
            pixel_y = max(0, min(self.MINIMAP_SIZE - 1, pixel_y))

            return pixel_x, pixel_y

        # Channel 0: Height map
        for y in range(self.MINIMAP_SIZE):
            for x in range(self.MINIMAP_SIZE):
                game_x = (x / self.MINIMAP_SIZE) * map_width
                game_y = (y / self.MINIMAP_SIZE) * map_height
                game_pos = Point2((game_x, game_y))
                rounded_pos = game_pos.rounded

                # Check bounds
                in_bounds = (0 <= rounded_pos[0] < map_width and
                           0 <= rounded_pos[1] < map_height)

                if in_bounds:
                    height = bot.get_terrain_height(game_pos) / 255.0
                    minimap[0, y, x] = height

                    # Channel 1: Visibility
                    is_visible = bot.state.visibility[rounded_pos] == 2
                    minimap[1, y, x] = 1.0 if is_visible else 0.0

                    # Channel 9: Buildable
                    minimap[9, y, x] = 1.0 if bot.in_placement_grid(game_pos) else 0.0

                    # Channel 10: Pathable
                    minimap[10, y, x] = 1.0 if bot.in_pathing_grid(game_pos) else 0.0

        # Channel 3: Camera position (where screen is looking)
        cam_x, cam_y = pos_to_minimap(bot.start_location)
        minimap[3, cam_y, cam_x] = 1.0

        # Channel 4: Player relative (friendly units/structures)
        for unit in bot.units:
            x, y = pos_to_minimap(unit.position)
            minimap[4, y, x] = 1.0  # Self

        for structure in bot.structures:
            x, y = pos_to_minimap(structure.position)
            minimap[4, y, x] = 1.0

        # Enemy units/structures
        for unit in bot.enemy_units:
            x, y = pos_to_minimap(unit.position)
            minimap[4, y, x] = 2.0  # Enemy

        for structure in bot.enemy_structures:
            x, y = pos_to_minimap(structure.position)
            minimap[4, y, x] = 2.0

        return minimap

    def _extract_scalar_features(self, bot: BotAI) -> np.ndarray:
        """
        Extract scalar features (economy, units, etc).

        Returns ~100 scalar values.
        """
        features = []

        # === Economy (10 features) ===
        features.extend([
            bot.minerals / 5000.0,
            bot.vespene / 5000.0,
            bot.supply_used / 200.0,
            bot.supply_cap / 200.0,
            bot.supply_left / 200.0,
            bot.supply_workers / 100.0,
            bot.supply_army / 200.0,
            (bot.minerals + bot.vespene) / 10000.0,  # Total resources
            bot.state.score.collection_rate_minerals / 2000.0,
            bot.state.score.collection_rate_vespene / 2000.0,
        ])

        # === Unit counts (30 features) ===
        unit_types = [
            UnitTypeId.SCV, UnitTypeId.MARINE, UnitTypeId.MARAUDER,
            UnitTypeId.REAPER, UnitTypeId.GHOST, UnitTypeId.HELLION,
            UnitTypeId.SIEGETANK, UnitTypeId.THOR, UnitTypeId.VIKING,
            UnitTypeId.MEDIVAC, UnitTypeId.LIBERATOR, UnitTypeId.BANSHEE,
            UnitTypeId.RAVEN, UnitTypeId.BATTLECRUISER,
        ]

        for unit_type in unit_types:
            count = bot.units(unit_type).amount
            features.append(min(count / 50.0, 1.0))

        # Pad to 30
        while len(features) < 40:
            features.append(0.0)

        # === Building counts (20 features) ===
        building_types = [
            UnitTypeId.COMMANDCENTER, UnitTypeId.SUPPLYDEPOT,
            UnitTypeId.REFINERY, UnitTypeId.BARRACKS, UnitTypeId.FACTORY,
            UnitTypeId.STARPORT, UnitTypeId.ENGINEERINGBAY,
            UnitTypeId.ARMORY, UnitTypeId.FUSIONCORE,
            UnitTypeId.BARRACKSTECHLAB, UnitTypeId.BARRACKSREACTOR,
            UnitTypeId.FACTORYTECHLAB, UnitTypeId.FACTORYREACTOR,
        ]

        for building_type in building_types:
            count = bot.structures(building_type).amount
            features.append(min(count / 10.0, 1.0))

        # Pad to 60
        while len(features) < 60:
            features.append(0.0)

        # === Upgrades (20 features) ===
        from sc2.ids.upgrade_id import UpgradeId

        upgrade_checks = [
            UpgradeId.STIMPACK,
            UpgradeId.SHIELDWALL,
            UpgradeId.PUNISHERGRENADES,
            UpgradeId.TERRANINFANTRYWEAPONSLEVEL1,
            UpgradeId.TERRANINFANTRYWEAPONSLEVEL2,
            UpgradeId.TERRANINFANTRYWEAPONSLEVEL3,
            UpgradeId.TERRANINFANTRYARMORSLEVEL1,
            UpgradeId.TERRANINFANTRYARMORSLEVEL2,
            UpgradeId.TERRANINFANTRYARMORSLEVEL3,
        ]

        for upgrade in upgrade_checks:
            has_upgrade = 1.0 if upgrade in bot.state.upgrades else 0.0
            features.append(has_upgrade)

        # Pad to 80
        while len(features) < 80:
            features.append(0.0)

        # === Game state (10 features) ===
        features.extend([
            bot.time / 1800.0,  # Game time (normalized to 30 min)
            bot.state.game_loop / 30000.0,  # Game loop
            len(bot.enemy_units) / 100.0,
            len(bot.enemy_structures) / 50.0,
            1.0 if bot.townhalls else 0.0,  # Has base
            bot.townhalls.amount / 5.0 if bot.townhalls else 0.0,
            len(bot.units) / 200.0,
            len(bot.structures) / 100.0,
            bot.state.score.total_damage_dealt_life / 100000.0,
            bot.state.score.total_damage_taken_life / 100000.0,
        ])

        return np.array(features, dtype=np.float32)


# Async version for use in bot
class AsyncSpatialFeatureExtractor(SpatialFeatureExtractor):
    """Async wrapper for feature extraction (needed for buildable checks)."""

    async def extract_features_async(self, bot: BotAI) -> dict:
        """Async version that can await bot methods."""
        # For now, use synchronous version
        # TODO: Make buildable checks async if needed
        return self.extract_features(bot)
