"""
Advanced RL bot with expanded action and observation space.

Supports full tech tree: gas, factories, starports, upgrades, multiple unit types.
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.data import Result
import numpy as np


class AdvancedRLBot(BotAI):
    """
    Advanced RL bot with expanded capabilities.

    Action Space (23 actions):
    0: train_scv
    1: build_supply_depot
    2: build_refinery
    3: build_barracks
    4: build_factory
    5: build_starport
    6: build_tech_lab_barracks
    7: build_reactor_barracks
    8: build_tech_lab_factory
    9: train_marine
    10: train_marauder
    11: train_tank
    12: train_hellion
    13: train_medivac
    14: research_stim
    15: research_combat_shields
    16: research_concussive_shells
    17: upgrade_infantry_weapons
    18: upgrade_infantry_armor
    19: attack
    20: defend
    21: expand
    22: no_op

    Observation Space (26 features):
    - minerals, gas, supply_used, supply_cap
    - worker_count, marine_count, marauder_count, tank_count, hellion_count, medivac_count
    - cc_count, barracks_count, factory_count, starport_count
    - refinery_count, tech_lab_count, reactor_count
    - has_stim, has_combat_shields, has_concussive_shells
    - infantry_weapons_level, infantry_armor_level
    - enemy_unit_count, enemy_structure_count
    - game_time, army_supply
    """

    ACTION_NAMES = [
        "train_scv", "build_supply_depot", "build_refinery",
        "build_barracks", "build_factory", "build_starport",
        "build_tech_lab_barracks", "build_reactor_barracks", "build_tech_lab_factory",
        "train_marine", "train_marauder", "train_tank", "train_hellion", "train_medivac",
        "research_stim", "research_combat_shields", "research_concussive_shells",
        "upgrade_infantry_weapons", "upgrade_infantry_armor",
        "attack", "defend", "expand", "no_op"
    ]

    def __init__(self, env, player_id=1, policy=None):
        super().__init__()
        self.env = env
        self.player_id = player_id
        self.custom_policy = policy
        self.attack_started = False

        # Reward tracking (only for player 1)
        self.prev_enemy_units = 0
        self.prev_own_units = 0
        self.prev_minerals = 0
        self.prev_gas = 0
        self.step_count = 0

    async def on_start(self):
        """Initialize bot."""
        self.client.game_step = 8

    async def on_step(self, iteration: int):
        """Main game loop - make decisions every 16 frames."""
        if iteration % 16 != 0:
            await self.distribute_workers()
            return

        self.step_count += 1

        # Get observation (25 features)
        obs = self._get_observation()

        # Get action from policy
        if self.player_id == 1:
            if self.env.policy is not None:
                action, _ = self.env.policy(obs)
                action = int(action)
            else:
                action = np.random.randint(0, 23)  # 23 actions

            # Execute action
            await self._execute_action(action)

            # Calculate reward
            reward = self._calculate_step_reward()
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
            self.env.add_step_to_trajectory(obs, action, reward, done, info)

        else:
            # Player 2 (opponent)
            if self.custom_policy is not None:
                action, _ = self.custom_policy(obs)
                action = int(action)
            else:
                action = np.random.randint(0, 23)
            await self._execute_action(action)

        # Basic worker management
        await self.distribute_workers()

    def _get_observation(self) -> np.ndarray:
        """Get 26-feature observation vector."""
        # Economy (4)
        minerals_norm = min(self.minerals / 2000.0, 1.0)
        gas_norm = min(self.vespene / 2000.0, 1.0)
        supply_used_norm = self.supply_used / 200.0
        supply_cap_norm = self.supply_cap / 200.0

        # Units (6)
        worker_count_norm = min(self.supply_workers / 80.0, 1.0)
        marine_count_norm = min(self.units(UnitTypeId.MARINE).amount / 50.0, 1.0)
        marauder_count_norm = min(self.units(UnitTypeId.MARAUDER).amount / 30.0, 1.0)
        tank_count_norm = min(self.units(UnitTypeId.SIEGETANK).amount / 20.0, 1.0)
        hellion_count_norm = min(self.units(UnitTypeId.HELLION).amount / 20.0, 1.0)
        medivac_count_norm = min(self.units(UnitTypeId.MEDIVAC).amount / 10.0, 1.0)

        # Buildings (7)
        cc_count_norm = min(self.townhalls.amount / 3.0, 1.0)
        barracks_count_norm = min(self.structures(UnitTypeId.BARRACKS).amount / 10.0, 1.0)
        factory_count_norm = min(self.structures(UnitTypeId.FACTORY).amount / 5.0, 1.0)
        starport_count_norm = min(self.structures(UnitTypeId.STARPORT).amount / 3.0, 1.0)
        refinery_count_norm = min(self.structures(UnitTypeId.REFINERY).amount / 6.0, 1.0)
        tech_lab_count_norm = min(
            (self.structures(UnitTypeId.BARRACKSTECHLAB).amount +
             self.structures(UnitTypeId.FACTORYTECHLAB).amount) / 5.0, 1.0
        )
        reactor_count_norm = min(
            (self.structures(UnitTypeId.BARRACKSREACTOR).amount +
             self.structures(UnitTypeId.FACTORYREACTOR).amount) / 5.0, 1.0
        )

        # Upgrades (5)
        has_stim = 1.0 if UpgradeId.STIMPACK in self.state.upgrades else 0.0
        has_combat_shields = 1.0 if UpgradeId.SHIELDWALL in self.state.upgrades else 0.0
        has_concussive = 1.0 if UpgradeId.PUNISHERGRENADES in self.state.upgrades else 0.0

        weapon_level = 0.0
        for upgrade in self.state.upgrades:
            if upgrade in [UpgradeId.TERRANINFANTRYWEAPONSLEVEL1,
                          UpgradeId.TERRANINFANTRYWEAPONSLEVEL2,
                          UpgradeId.TERRANINFANTRYWEAPONSLEVEL3]:
                weapon_level = min(len([u for u in self.state.upgrades
                                       if "INFANTRYWEAPONS" in str(u)]) / 3.0, 1.0)

        armor_level = 0.0
        for upgrade in self.state.upgrades:
            if upgrade in [UpgradeId.TERRANINFANTRYARMORSLEVEL1,
                          UpgradeId.TERRANINFANTRYARMORSLEVEL2,
                          UpgradeId.TERRANINFANTRYARMORSLEVEL3]:
                armor_level = min(len([u for u in self.state.upgrades
                                      if "INFANTRYARMOR" in str(u)]) / 3.0, 1.0)

        # Enemy (2)
        enemy_units_norm = min(len(self.enemy_units) / 50.0, 1.0)
        enemy_structures_norm = min(len(self.enemy_structures) / 20.0, 1.0)

        # Game state (2)
        game_time_norm = min(self.time / 600.0, 1.0)
        army_supply_norm = min(self.supply_army / 100.0, 1.0)

        obs = np.array([
            minerals_norm, gas_norm, supply_used_norm, supply_cap_norm,
            worker_count_norm, marine_count_norm, marauder_count_norm,
            tank_count_norm, hellion_count_norm, medivac_count_norm,
            cc_count_norm, barracks_count_norm, factory_count_norm,
            starport_count_norm, refinery_count_norm, tech_lab_count_norm,
            reactor_count_norm, has_stim, has_combat_shields, has_concussive,
            weapon_level, armor_level, enemy_units_norm, enemy_structures_norm,
            game_time_norm, army_supply_norm
        ], dtype=np.float32)

        return obs

    async def _execute_action(self, action: int):
        """Execute action from expanded action space."""
        if action == 0:
            await self._train_scv()
        elif action == 1:
            await self._build_supply_depot()
        elif action == 2:
            await self._build_refinery()
        elif action == 3:
            await self._build_barracks()
        elif action == 4:
            await self._build_factory()
        elif action == 5:
            await self._build_starport()
        elif action == 6:
            await self._build_tech_lab(UnitTypeId.BARRACKS)
        elif action == 7:
            await self._build_reactor(UnitTypeId.BARRACKS)
        elif action == 8:
            await self._build_tech_lab(UnitTypeId.FACTORY)
        elif action == 9:
            await self._train_marine()
        elif action == 10:
            await self._train_marauder()
        elif action == 11:
            await self._train_tank()
        elif action == 12:
            await self._train_hellion()
        elif action == 13:
            await self._train_medivac()
        elif action == 14:
            await self._research_stim()
        elif action == 15:
            await self._research_combat_shields()
        elif action == 16:
            await self._research_concussive_shells()
        elif action == 17:
            await self._upgrade_infantry_weapons()
        elif action == 18:
            await self._upgrade_infantry_armor()
        elif action == 19:
            await self._attack()
        elif action == 20:
            await self._defend()
        elif action == 21:
            await self._expand()
        elif action == 22:
            pass  # no_op

    # Basic actions
    async def _train_scv(self):
        for cc in self.townhalls.idle:
            if self.can_afford(UnitTypeId.SCV) and self.supply_left > 0:
                cc.train(UnitTypeId.SCV)
                return

    async def _build_supply_depot(self):
        if (self.can_afford(UnitTypeId.SUPPLYDEPOT) and self.supply_cap < 200
                and not self.already_pending(UnitTypeId.SUPPLYDEPOT)):
            workers = self.workers.gathering
            if workers:
                worker = workers.closest_to(self.start_location)
                location = await self.find_placement(
                    UnitTypeId.SUPPLYDEPOT,
                    near=self.start_location.towards(self.game_info.map_center, 8)
                )
                if location:
                    worker.build(UnitTypeId.SUPPLYDEPOT, location)

    async def _build_refinery(self):
        """Build refinery on vespene geyser."""
        for townhall in self.townhalls.ready:
            vespenes = self.vespene_geyser.closer_than(10, townhall)
            for vespene in vespenes:
                if self.structures(UnitTypeId.REFINERY).closer_than(1, vespene):
                    continue
                if self.already_pending(UnitTypeId.REFINERY):
                    continue
                if self.can_afford(UnitTypeId.REFINERY):
                    workers = self.workers.gathering
                    if workers:
                        worker = workers.closest_to(vespene)
                        worker.build(UnitTypeId.REFINERY, vespene)
                        return

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

    async def _build_tech_lab(self, building_type):
        """Build tech lab on barracks or factory."""
        buildings = self.structures(building_type).ready.idle
        for building in buildings:
            if building.has_add_on:
                continue
            if self.can_afford(UnitTypeId.BARRACKSTECHLAB):
                building.build(UnitTypeId.BARRACKSTECHLAB if building_type == UnitTypeId.BARRACKS
                              else UnitTypeId.FACTORYTECHLAB)
                return

    async def _build_reactor(self, building_type):
        """Build reactor on barracks or factory."""
        buildings = self.structures(building_type).ready.idle
        for building in buildings:
            if building.has_add_on:
                continue
            if self.can_afford(UnitTypeId.BARRACKSREACTOR):
                building.build(UnitTypeId.BARRACKSREACTOR if building_type == UnitTypeId.BARRACKS
                              else UnitTypeId.FACTORYREACTOR)
                return

    # Unit training
    async def _train_marine(self):
        for barracks in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.can_afford(UnitTypeId.MARINE) and self.supply_left > 0:
                barracks.train(UnitTypeId.MARINE)
                return

    async def _train_marauder(self):
        for barracks in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if (barracks.has_add_on and self.can_afford(UnitTypeId.MARAUDER)
                    and self.supply_left > 0):
                barracks.train(UnitTypeId.MARAUDER)
                return

    async def _train_tank(self):
        for factory in self.structures(UnitTypeId.FACTORY).ready.idle:
            if (factory.has_add_on and self.can_afford(UnitTypeId.SIEGETANK)
                    and self.supply_left > 0):
                factory.train(UnitTypeId.SIEGETANK)
                return

    async def _train_hellion(self):
        for factory in self.structures(UnitTypeId.FACTORY).ready.idle:
            if self.can_afford(UnitTypeId.HELLION) and self.supply_left > 0:
                factory.train(UnitTypeId.HELLION)
                return

    async def _train_medivac(self):
        for starport in self.structures(UnitTypeId.STARPORT).ready.idle:
            if self.can_afford(UnitTypeId.MEDIVAC) and self.supply_left > 0:
                starport.train(UnitTypeId.MEDIVAC)
                return

    # Upgrades
    async def _research_stim(self):
        if UpgradeId.STIMPACK in self.state.upgrades:
            return
        tech_labs = self.structures(UnitTypeId.BARRACKSTECHLAB).ready
        if tech_labs and self.can_afford(UpgradeId.STIMPACK):
            tech_labs.first.research(UpgradeId.STIMPACK)

    async def _research_combat_shields(self):
        if UpgradeId.SHIELDWALL in self.state.upgrades:
            return
        tech_labs = self.structures(UnitTypeId.BARRACKSTECHLAB).ready
        if tech_labs and self.can_afford(UpgradeId.SHIELDWALL):
            tech_labs.first.research(UpgradeId.SHIELDWALL)

    async def _research_concussive_shells(self):
        if UpgradeId.PUNISHERGRENADES in self.state.upgrades:
            return
        tech_labs = self.structures(UnitTypeId.BARRACKSTECHLAB).ready
        if tech_labs and self.can_afford(UpgradeId.PUNISHERGRENADES):
            tech_labs.first.research(UpgradeId.PUNISHERGRENADES)

    async def _upgrade_infantry_weapons(self):
        if not self.structures(UnitTypeId.ENGINEERINGBAY).ready:
            return
        eng_bay = self.structures(UnitTypeId.ENGINEERINGBAY).ready.first
        if self.can_afford(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1):
            eng_bay.research(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1)

    async def _upgrade_infantry_armor(self):
        if not self.structures(UnitTypeId.ENGINEERINGBAY).ready:
            return
        eng_bay = self.structures(UnitTypeId.ENGINEERINGBAY).ready.first
        if self.can_afford(UpgradeId.TERRANINFANTRYARMORSLEVEL1):
            eng_bay.research(UpgradeId.TERRANINFANTRYARMORSLEVEL1)

    # Tactics
    async def _attack(self):
        self.attack_started = True
        army = self.units.of_type([
            UnitTypeId.MARINE, UnitTypeId.MARAUDER,
            UnitTypeId.SIEGETANK, UnitTypeId.HELLION, UnitTypeId.MEDIVAC
        ])

        if self.enemy_units or self.enemy_structures:
            all_enemies = self.enemy_units | self.enemy_structures
            for unit in army:
                if unit.is_idle:
                    closest_enemy = all_enemies.closest_to(unit)
                    unit.attack(closest_enemy)
        else:
            enemy_start = self.enemy_start_locations[0]
            for unit in army:
                if unit.is_idle:
                    unit.attack(enemy_start)

    async def _defend(self):
        self.attack_started = False
        army = self.units.of_type([
            UnitTypeId.MARINE, UnitTypeId.MARAUDER,
            UnitTypeId.SIEGETANK, UnitTypeId.HELLION, UnitTypeId.MEDIVAC
        ])
        defend_pos = self.start_location.towards(self.game_info.map_center, 15)
        for unit in army:
            if unit.distance_to(defend_pos) > 10:
                unit.move(defend_pos)

    async def _expand(self):
        """Build expansion command center."""
        if self.townhalls.amount >= 3 or self.already_pending(UnitTypeId.COMMANDCENTER):
            return
        if self.can_afford(UnitTypeId.COMMANDCENTER):
            location = await self.get_next_expansion()
            if location:
                workers = self.workers.gathering
                if workers:
                    worker = workers.closest_to(location)
                    worker.build(UnitTypeId.COMMANDCENTER, location)

    def _calculate_step_reward(self) -> float:
        """Calculate reward for this step."""
        reward = 0.0

        # Enemy units killed
        current_enemy_units = len(self.enemy_units)
        if current_enemy_units < self.prev_enemy_units:
            reward += (self.prev_enemy_units - current_enemy_units) * 0.1
        self.prev_enemy_units = current_enemy_units

        # Own units lost
        current_own_units = self.supply_army + self.supply_workers
        if current_own_units < self.prev_own_units:
            reward -= (self.prev_own_units - current_own_units) * 0.05
        self.prev_own_units = current_own_units

        # Economy
        if self.minerals > self.prev_minerals:
            reward += (self.minerals - self.prev_minerals) * 0.0001
        self.prev_minerals = self.minerals

        if self.vespene > self.prev_gas:
            reward += (self.vespene - self.prev_gas) * 0.0002
        self.prev_gas = self.vespene

        return reward

    def _check_if_done(self) -> bool:
        """Check if game ended."""
        if not self.townhalls:
            self.env.game_result = Result.Defeat
            return True
        elif not self.enemy_structures:
            self.env.game_result = Result.Victory
            return True
        return False
