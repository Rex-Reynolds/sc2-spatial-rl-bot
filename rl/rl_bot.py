"""
RL-controlled bot that executes actions from the RL agent's policy.

Simplified approach: Bot calls env.policy() to get actions and records
trajectory for training.
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.data import Result
import numpy as np


class RLBot(BotAI):
    """
    Bot controlled by RL agent's policy.

    Collects trajectory during game for training.
    """

    ACTION_NAMES = [
        "train_scv",
        "build_supply_depot",
        "build_barracks",
        "train_marine",
        "attack",
        "defend",
        "no_op",
    ]

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.attack_started = False

        # Reward tracking
        self.prev_enemy_units = 0
        self.prev_own_units = 0
        self.prev_minerals = 0
        self.prev_gas = 0

        # Step tracking
        self.step_count = 0

    async def on_start(self):
        """Initialize bot on game start."""
        self.client.game_step = 8

    async def on_step(self, iteration: int):
        """
        Main game loop step.

        Every N iterations:
        1. Get observation
        2. Ask policy for action
        3. Execute action
        4. Record to trajectory
        """
        # Only make RL decisions every 16 steps (~1 second)
        if iteration % 16 != 0:
            await self.distribute_workers()
            return

        self.step_count += 1

        # Get observation
        obs = self._get_observation()

        # Get action from policy (or random if no policy set)
        if self.env.policy is not None:
            action, _ = self.env.policy(obs)
            action = int(action)  # Convert from numpy
        else:
            # Random action if no policy
            action = np.random.randint(0, 7)

        # Execute action
        await self._execute_action(action)

        # Calculate reward for this step
        reward = self._calculate_step_reward()

        # Check if game is done
        done = self._check_if_done()

        # Add to trajectory
        info = {"step": self.step_count}
        self.env.add_step_to_trajectory(obs, action, reward, done, info)

        # Basic worker distribution
        await self.distribute_workers()

    def _get_observation(self) -> np.ndarray:
        """Get current observation (11 features, normalized to [0, 1])."""
        # Normalize values to [0, 1]
        minerals_norm = min(self.minerals / 2000.0, 1.0)
        gas_norm = min(self.vespene / 2000.0, 1.0)
        supply_used_norm = self.supply_used / 200.0
        supply_cap_norm = self.supply_cap / 200.0

        scv_count_norm = min(self.supply_workers / 80.0, 1.0)
        marine_count_norm = min(self.units(UnitTypeId.MARINE).amount / 50.0, 1.0)
        barracks_count_norm = min(
            self.structures(UnitTypeId.BARRACKS).amount / 10.0, 1.0
        )

        enemy_units_norm = min(len(self.enemy_units) / 50.0, 1.0)
        enemy_structures_norm = min(len(self.enemy_structures) / 20.0, 1.0)

        game_time_norm = min(self.time / 600.0, 1.0)  # Normalize to 10 minutes

        # Calculate army strength difference
        own_army = self.units(UnitTypeId.MARINE).amount
        enemy_army = len(self.enemy_units)
        if own_army + enemy_army > 0:
            army_diff = (own_army - enemy_army) / (own_army + enemy_army + 1)
            army_strength_norm = (army_diff + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        else:
            army_strength_norm = 0.5

        obs = np.array(
            [
                minerals_norm,
                gas_norm,
                supply_used_norm,
                supply_cap_norm,
                scv_count_norm,
                marine_count_norm,
                barracks_count_norm,
                enemy_units_norm,
                enemy_structures_norm,
                game_time_norm,
                army_strength_norm,
            ],
            dtype=np.float32,
        )

        return obs

    async def _execute_action(self, action: int):
        """Execute the action chosen by the RL agent."""
        if action == 0:  # train_scv
            await self._train_scv()
        elif action == 1:  # build_supply_depot
            await self._build_supply_depot()
        elif action == 2:  # build_barracks
            await self._build_barracks()
        elif action == 3:  # train_marine
            await self._train_marine()
        elif action == 4:  # attack
            await self._attack()
        elif action == 5:  # defend
            await self._defend()
        elif action == 6:  # no_op
            pass  # Do nothing

    async def _train_scv(self):
        """Train an SCV if possible."""
        for cc in self.townhalls.idle:
            if self.can_afford(UnitTypeId.SCV) and self.supply_left > 0:
                cc.train(UnitTypeId.SCV)
                return

    async def _build_supply_depot(self):
        """Build a supply depot if possible."""
        if (
            self.can_afford(UnitTypeId.SUPPLYDEPOT)
            and self.supply_cap < 200
            and not self.already_pending(UnitTypeId.SUPPLYDEPOT)
        ):
            workers = self.workers.gathering
            if workers:
                worker = workers.closest_to(self.start_location)
                location = await self.find_placement(
                    UnitTypeId.SUPPLYDEPOT,
                    near=self.start_location.towards(self.game_info.map_center, 8),
                )
                if location:
                    worker.build(UnitTypeId.SUPPLYDEPOT, location)

    async def _build_barracks(self):
        """Build a barracks if possible."""
        # Need supply depot first
        if not self.structures(UnitTypeId.SUPPLYDEPOT).ready:
            return

        if self.can_afford(UnitTypeId.BARRACKS):
            workers = self.workers.gathering
            if workers:
                worker = workers.closest_to(self.start_location)
                location = await self.find_placement(
                    UnitTypeId.BARRACKS,
                    near=self.start_location.towards(self.game_info.map_center, 12),
                )
                if location:
                    worker.build(UnitTypeId.BARRACKS, location)

    async def _train_marine(self):
        """Train a marine if possible."""
        for barracks in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.can_afford(UnitTypeId.MARINE) and self.supply_left > 0:
                barracks.train(UnitTypeId.MARINE)
                return

    async def _attack(self):
        """Send army to attack."""
        self.attack_started = True
        marines = self.units(UnitTypeId.MARINE)
        enemy_start = self.enemy_start_locations[0]

        if self.enemy_units or self.enemy_structures:
            all_enemies = self.enemy_units | self.enemy_structures
            for marine in marines:
                if marine.is_idle:
                    closest_enemy = all_enemies.closest_to(marine)
                    marine.attack(closest_enemy)
        else:
            for marine in marines:
                if marine.is_idle:
                    marine.attack(enemy_start)

    async def _defend(self):
        """Pull army back to defensive position."""
        self.attack_started = False
        marines = self.units(UnitTypeId.MARINE)
        defend_pos = self.start_location.towards(self.game_info.map_center, 15)

        for marine in marines:
            if marine.distance_to(defend_pos) > 10:
                marine.move(defend_pos)

    def _calculate_step_reward(self) -> float:
        """Calculate reward for this step."""
        reward = 0.0

        # Enemy units killed
        current_enemy_units = len(self.enemy_units)
        if current_enemy_units < self.prev_enemy_units:
            kills = self.prev_enemy_units - current_enemy_units
            reward += kills * 0.1
        self.prev_enemy_units = current_enemy_units

        # Own units lost
        current_own_units = (
            self.units(UnitTypeId.MARINE).amount + self.supply_workers
        )
        if current_own_units < self.prev_own_units:
            losses = self.prev_own_units - current_own_units
            reward -= losses * 0.05
        self.prev_own_units = current_own_units

        # Economy growth (small bonus)
        if self.minerals > self.prev_minerals:
            reward += (self.minerals - self.prev_minerals) * 0.0001
        self.prev_minerals = self.minerals

        if self.vespene > self.prev_gas:
            reward += (self.vespene - self.prev_gas) * 0.0002
        self.prev_gas = self.vespene

        return reward

    def _check_if_done(self) -> bool:
        """Check if game has ended."""
        if not self.townhalls:
            # We lost (no command centers)
            self.env.game_result = Result.Defeat
            return True
        elif not self.enemy_structures:
            # We won (enemy has no structures)
            self.env.game_result = Result.Victory
            return True
        return False
