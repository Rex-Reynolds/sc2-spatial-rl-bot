"""
Spatial RL Bot for SC2

Uses spatial observations and actions with CNN policy.
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.ability_id import AbilityId
from sc2.data import Result
from sc2.position import Point2
import numpy as np
from typing import Dict, Optional

from rl.spatial_features import SpatialFeatureExtractor


class SpatialRLBot(BotAI):
    """
    RL bot with spatial actions.

    Action Space (50 simplified actions):
      0: No-op
      1: Select idle workers
      2: Select all army
      3-4: Select specific unit types (marines, tanks, etc.)
      5: Train SCV
      6-10: Build structures (depot, barracks, factory, etc.)
      11-20: Train units (marine, marauder, tank, medivac, etc.)
      21-25: Research upgrades (stim, shields, weapons, etc.)
      26-35: Attack/Move commands (use screen coords)
      36-40: Special abilities (stim, siege, medivac heal)
      41-49: Strategic actions (expand, defend, scout)
    """

    # Action definitions
    ACTION_NAMES = [
        "no_op",                    # 0
        "select_idle_workers",      # 1
        "select_army",              # 2
        "select_marines",           # 3
        "select_tanks",             # 4
        "train_scv",                # 5
        "build_supply_depot",       # 6
        "build_barracks",           # 7
        "build_refinery",           # 8
        "build_factory",            # 9
        "build_starport",           # 10
        "train_marine",             # 11
        "train_marauder",           # 12
        "train_tank",               # 13
        "train_hellion",            # 14
        "train_medivac",            # 15
        "build_tech_lab_barracks",  # 16
        "build_reactor_barracks",   # 17
        "research_stim",            # 18
        "research_combat_shields",  # 19
        "research_concussive",      # 20
        "upgrade_weapons",          # 21
        "upgrade_armor",            # 22
        "move_screen",              # 23 (uses screen coords)
        "attack_screen",            # 24 (uses screen coords)
        "move_minimap",             # 25 (uses minimap coords)
        "attack_minimap",           # 26 (uses minimap coords)
        "patrol_screen",            # 27
        "hold_position",            # 28
        "stop",                     # 29
        "use_stim",                 # 30
        "siege_tanks",              # 31
        "unsiege_tanks",            # 32
        "load_medivac",             # 33
        "unload_medivac",           # 34
        "build_at_location",        # 35 (uses screen coords)
        "expand",                   # 36
        "defend_base",              # 37
        "defend_natural",           # 38
        "attack_enemy_base",        # 39
        "retreat",                  # 40
        "scout",                    # 41
        "focus_fire",               # 42 (target from screen)
        "split_army",               # 43
        "rally_point",              # 44 (uses screen)
        "cancel_building",          # 45
        "salvage_bunker",           # 46
        "lift_building",            # 47
        "land_building",            # 48 (uses screen)
        "scan",                     # 49 (uses screen, if orbital)
    ]

    def __init__(self, env, player_id=1, policy=None):
        super().__init__()
        self.env = env
        self.player_id = player_id
        self.custom_policy = policy
        self.feature_extractor = SpatialFeatureExtractor()

        # Reward tracking
        self.prev_enemy_units = 0
        self.prev_own_units = 0
        self.prev_minerals = 0
        self.step_count = 0

        # Selected units (for sequential actions)
        self.selected_units = []

    async def on_start(self):
        """Initialize bot."""
        self.client.game_step = 4  # Fast for micro (4 frames = ~0.25 sec)

    async def on_step(self, iteration: int):
        """Main game loop - make decisions every N frames."""
        if iteration % self.env.step_interval != 0:
            await self.distribute_workers()
            return

        self.step_count += 1

        # Get spatial observation
        obs = self._get_spatial_observation()

        # Get action from policy
        if self.player_id == 1:
            if self.env.policy is not None:
                action_dict = self.env.policy(obs)
            else:
                # Random action for testing
                action_dict = self._random_action()

            # Execute action
            await self._execute_spatial_action(action_dict)

            # Calculate reward
            reward = self._calculate_reward()
            done = self._check_if_done()

            # Add to trajectory
            info = {
                "step": self.step_count,
                "result": self.env.game_result if done else None,
            }
            if done and self.env.game_result:
                info["episode"] = {
                    "r": self.env.episode_reward,
                    "l": self.step_count,
                }
            self.env.add_step_to_trajectory(obs, action_dict, reward, done, info)

        else:
            # Player 2 (opponent)
            if self.custom_policy is not None:
                action_dict = self.custom_policy(obs)
            else:
                action_dict = self._random_action()
            await self._execute_spatial_action(action_dict)

        # Basic worker management
        await self.distribute_workers()

    def _get_spatial_observation(self) -> Dict[str, np.ndarray]:
        """Extract spatial features from game state."""
        features = self.feature_extractor.extract_features(self)
        return features

    def _random_action(self) -> Dict[str, int]:
        """Generate random action for testing."""
        return {
            'action_type': np.random.randint(0, len(self.ACTION_NAMES)),
            'screen_idx': np.random.randint(0, 64 * 64),
            'minimap_idx': np.random.randint(0, 64 * 64),
        }

    async def _execute_spatial_action(self, action_dict: Dict[str, int]):
        """Execute spatial action."""
        action_type = action_dict['action_type']
        screen_idx = action_dict['screen_idx']
        minimap_idx = action_dict['minimap_idx']

        # Convert indices to coordinates
        screen_x = screen_idx % 64
        screen_y = screen_idx // 64
        minimap_x = minimap_idx % 64
        minimap_y = minimap_idx // 64

        # Convert screen/minimap coords to game position
        screen_pos = self._screen_to_game_pos(screen_x, screen_y)
        minimap_pos = self._minimap_to_game_pos(minimap_x, minimap_y)

        action_name = self.ACTION_NAMES[action_type] if action_type < len(self.ACTION_NAMES) else "invalid"

        # Execute action based on type
        try:
            if action_type == 0:  # no_op
                pass
            elif action_type == 1:  # select_idle_workers
                self.selected_units = self.workers.idle
            elif action_type == 2:  # select_army
                self.selected_units = self.units.exclude_type([UnitTypeId.SCV])
            elif action_type == 3:  # select_marines
                self.selected_units = self.units(UnitTypeId.MARINE)
            elif action_type == 4:  # select_tanks
                self.selected_units = self.units(UnitTypeId.SIEGETANK)
            elif action_type == 5:  # train_scv
                await self._train_scv()
            elif action_type == 6:  # build_supply_depot
                await self._build_supply_depot()
            elif action_type == 7:  # build_barracks
                await self._build_barracks()
            elif action_type == 8:  # build_refinery
                await self._build_refinery()
            elif action_type == 9:  # build_factory
                await self._build_factory()
            elif action_type == 10:  # build_starport
                await self._build_starport()
            elif action_type == 11:  # train_marine
                await self._train_unit(UnitTypeId.MARINE, UnitTypeId.BARRACKS)
            elif action_type == 12:  # train_marauder
                await self._train_unit(UnitTypeId.MARAUDER, UnitTypeId.BARRACKS)
            elif action_type == 13:  # train_tank
                await self._train_unit(UnitTypeId.SIEGETANK, UnitTypeId.FACTORY)
            elif action_type == 14:  # train_hellion
                await self._train_unit(UnitTypeId.HELLION, UnitTypeId.FACTORY)
            elif action_type == 15:  # train_medivac
                await self._train_unit(UnitTypeId.MEDIVAC, UnitTypeId.STARPORT)
            elif action_type == 18:  # research_stim
                await self._research_upgrade(UpgradeId.STIMPACK, UnitTypeId.BARRACKSTECHLAB)
            elif action_type == 23:  # move_screen
                if self.selected_units:
                    for unit in self.selected_units:
                        unit.move(screen_pos)
            elif action_type == 24:  # attack_screen
                if self.selected_units:
                    for unit in self.selected_units:
                        unit.attack(screen_pos)
            elif action_type == 25:  # move_minimap
                if self.selected_units:
                    for unit in self.selected_units:
                        unit.move(minimap_pos)
            elif action_type == 26:  # attack_minimap
                if self.selected_units:
                    for unit in self.selected_units:
                        unit.attack(minimap_pos)
            elif action_type == 30:  # use_stim
                for unit in self.units({UnitTypeId.MARINE, UnitTypeId.MARAUDER}):
                    if unit.health_percentage > 0.3:
                        unit(AbilityId.EFFECT_STIM)
            elif action_type == 36:  # expand
                await self._expand()
            elif action_type == 39:  # attack_enemy_base
                await self._attack_enemy_base()
            elif action_type == 40:  # retreat
                await self._retreat()

        except Exception as e:
            # Silently handle invalid actions
            pass

    def _screen_to_game_pos(self, screen_x: int, screen_y: int) -> Point2:
        """Convert screen coordinates to game position."""
        screen_radius = 40
        camera_center = self.start_location

        # Convert pixel to relative position
        rel_x = (screen_x / 64.0) * 2 - 1  # -1 to 1
        rel_y = (screen_y / 64.0) * 2 - 1

        game_x = camera_center.x + rel_x * screen_radius
        game_y = camera_center.y + rel_y * screen_radius

        return Point2((game_x, game_y))

    def _minimap_to_game_pos(self, minimap_x: int, minimap_y: int) -> Point2:
        """Convert minimap coordinates to game position."""
        map_width = self.game_info.map_size[0]
        map_height = self.game_info.map_size[1]

        game_x = (minimap_x / 64.0) * map_width
        game_y = (minimap_y / 64.0) * map_height

        return Point2((game_x, game_y))

    # === Basic Actions ===

    async def _train_scv(self):
        for cc in self.townhalls.idle:
            if self.can_afford(UnitTypeId.SCV) and self.supply_left > 0:
                cc.train(UnitTypeId.SCV)
                return

    async def _build_supply_depot(self):
        if self.can_afford(UnitTypeId.SUPPLYDEPOT) and self.supply_cap < 200:
            workers = self.workers.gathering
            if workers:
                worker = workers.closest_to(self.start_location)
                location = await self.find_placement(
                    UnitTypeId.SUPPLYDEPOT,
                    near=self.start_location.towards(self.game_info.map_center, 8)
                )
                if location:
                    worker.build(UnitTypeId.SUPPLYDEPOT, location)

    async def _build_barracks(self):
        if not self.structures(UnitTypeId.SUPPLYDEPOT).ready:
            return
        if self.can_afford(UnitTypeId.BARRACKS):
            workers = self.workers.gathering
            if workers:
                worker = workers.closest_to(self.start_location)
                location = await self.find_placement(
                    UnitTypeId.BARRACKS,
                    near=self.start_location.towards(self.game_info.map_center, 12)
                )
                if location:
                    worker.build(UnitTypeId.BARRACKS, location)

    async def _build_refinery(self):
        for townhall in self.townhalls.ready:
            vespenes = self.vespene_geyser.closer_than(10, townhall)
            for vespene in vespenes:
                if self.structures(UnitTypeId.REFINERY).closer_than(1, vespene):
                    continue
                if self.can_afford(UnitTypeId.REFINERY):
                    workers = self.workers.gathering
                    if workers:
                        worker = workers.closest_to(vespene)
                        worker.build(UnitTypeId.REFINERY, vespene)
                        return

    async def _build_factory(self):
        if not self.structures(UnitTypeId.BARRACKS).ready:
            return
        if self.can_afford(UnitTypeId.FACTORY):
            workers = self.workers.gathering
            if workers:
                worker = workers.closest_to(self.start_location)
                location = await self.find_placement(
                    UnitTypeId.FACTORY,
                    near=self.start_location.towards(self.game_info.map_center, 16)
                )
                if location:
                    worker.build(UnitTypeId.FACTORY, location)

    async def _build_starport(self):
        if not self.structures(UnitTypeId.FACTORY).ready:
            return
        if self.can_afford(UnitTypeId.STARPORT):
            workers = self.workers.gathering
            if workers:
                worker = workers.closest_to(self.start_location)
                location = await self.find_placement(
                    UnitTypeId.STARPORT,
                    near=self.start_location.towards(self.game_info.map_center, 20)
                )
                if location:
                    worker.build(UnitTypeId.STARPORT, location)

    async def _train_unit(self, unit_type: UnitTypeId, building_type: UnitTypeId):
        for building in self.structures(building_type).ready.idle:
            if self.can_afford(unit_type) and self.supply_left > 0:
                building.train(unit_type)
                return

    async def _research_upgrade(self, upgrade: UpgradeId, building_type: UnitTypeId):
        if upgrade in self.state.upgrades:
            return
        for building in self.structures(building_type).ready.idle:
            if self.can_afford(upgrade):
                building.research(upgrade)
                return

    async def _expand(self):
        if self.can_afford(UnitTypeId.COMMANDCENTER):
            location = await self.get_next_expansion()
            if location:
                workers = self.workers.gathering
                if workers:
                    worker = workers.closest_to(location)
                    worker.build(UnitTypeId.COMMANDCENTER, location)

    async def _attack_enemy_base(self):
        if self.enemy_start_locations:
            target = self.enemy_start_locations[0]
            for unit in self.units.exclude_type([UnitTypeId.SCV]):
                unit.attack(target)

    async def _retreat(self):
        for unit in self.units.exclude_type([UnitTypeId.SCV]):
            if unit.health_percentage < 0.5:
                unit.move(self.start_location)

    def _calculate_reward(self) -> float:
        """Calculate step reward."""
        reward = 0.0

        # Kill rewards
        enemy_units_now = len(self.enemy_units) + len(self.enemy_structures)
        own_units_now = len(self.units) + len(self.structures)

        units_killed = self.prev_enemy_units - enemy_units_now
        units_lost = self.prev_own_units - own_units_now

        reward += units_killed * 0.2
        reward -= units_lost * 0.1

        self.prev_enemy_units = enemy_units_now
        self.prev_own_units = own_units_now

        return reward

    def _check_if_done(self) -> bool:
        """Check if game is over."""
        if self.env.game_result is not None:
            # Add final reward
            if self.env.game_result == Result.Victory:
                return True
            elif self.env.game_result == Result.Defeat:
                return True
        return False
