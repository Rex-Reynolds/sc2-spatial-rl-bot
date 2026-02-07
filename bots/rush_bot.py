"""Marine rush bot with tunable parameters."""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2


# Tunable constants for easy iteration
MAX_WORKERS = 16  # Stop SCV production at this count (start with 12)
BARRACKS_COUNT = 2  # Number of barracks to build
ATTACK_MARINE_THRESHOLD = 8  # Marines needed before attacking


class RushBot(BotAI):
    """
    Aggressive marine rush bot.

    Strategy:
    1. Train SCVs up to MAX_WORKERS
    2. Build supply depots when supply_left < 3
    3. Build barracks towards map center for shorter travel time
    4. Train marines from all idle barracks
    5. Attack with all marines + workers once threshold is reached (all-in)
    6. Keep sending newly trained marines as reinforcements
    """

    def __init__(self):
        super().__init__()
        self.attack_started = False

    async def on_start(self):
        """Initialize bot settings."""
        # Set game step to 2 for faster reactions (8 steps per second)
        self.client.game_step = 2

    async def on_step(self, iteration: int):
        """Execute bot logic each game step."""
        await self.distribute_workers()  # Optimize mineral/gas distribution
        await self.train_scvs()
        await self.build_supply_depots()
        await self.build_barracks()
        await self.train_marines()
        await self.attack()

    async def train_scvs(self):
        """Train SCVs up to MAX_WORKERS."""
        if self.supply_workers >= MAX_WORKERS:
            return

        for cc in self.townhalls.idle:
            if self.can_afford(UnitTypeId.SCV):
                cc.train(UnitTypeId.SCV)

    async def build_supply_depots(self):
        """Build supply depots when supply is low."""
        # Calculate how much supply we'll need soon
        # Account for supply currently being built
        supply_pending = self.already_pending(UnitTypeId.SUPPLYDEPOT) * 8
        effective_supply_left = self.supply_left + supply_pending

        # Build if supply will be tight soon (more proactive)
        if (
            effective_supply_left < 6
            and self.supply_cap < 200
            and self.can_afford(UnitTypeId.SUPPLYDEPOT)
            # Allow multiple depots to be built at once if needed
            and self.already_pending(UnitTypeId.SUPPLYDEPOT) < 2
        ):
            workers = self.workers.gathering
            if workers:
                worker = workers.closest_to(self.start_location)
                # Build near the command center
                location = await self.find_placement(
                    UnitTypeId.SUPPLYDEPOT,
                    near=self.start_location.towards(self.game_info.map_center, 8),
                )
                if location:
                    worker.build(UnitTypeId.SUPPLYDEPOT, location)

    async def build_barracks(self):
        """Build barracks towards map center."""
        if self.structures(UnitTypeId.BARRACKS).amount >= BARRACKS_COUNT:
            return

        # Need at least one supply depot (built or building) before starting barracks
        if not self.structures(UnitTypeId.SUPPLYDEPOT) and not self.already_pending(UnitTypeId.SUPPLYDEPOT):
            return

        if (
            self.can_afford(UnitTypeId.BARRACKS)
            and self.already_pending(UnitTypeId.BARRACKS) < BARRACKS_COUNT
        ):
            workers = self.workers.gathering
            if workers:
                worker = workers.closest_to(self.start_location)
                # Build barracks closer to map center for shorter marine travel time
                location = await self.find_placement(
                    UnitTypeId.BARRACKS,
                    near=self.start_location.towards(self.game_info.map_center, 15),
                )
                if location:
                    worker.build(UnitTypeId.BARRACKS, location)

    async def train_marines(self):
        """Train marines from all idle barracks."""
        for barracks in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.can_afford(UnitTypeId.MARINE):
                barracks.train(UnitTypeId.MARINE)

    async def attack(self):
        """
        Attack logic: All-in once marine threshold is reached.

        - Send all marines + all workers to enemy base
        - Keep sending reinforcements as they train
        - If command center is destroyed, send everything
        """
        marines = self.units(UnitTypeId.MARINE)
        enemy_start = self.enemy_start_locations[0]

        # Start attack if we have enough marines or if our base is destroyed
        if not self.attack_started:
            if len(marines) >= ATTACK_MARINE_THRESHOLD or not self.townhalls:
                self.attack_started = True
                print(f"[RushBot] Attack started with {len(marines)} marines!")

        # Once attack is started, send everything
        if self.attack_started:
            # Target enemy units if visible, otherwise attack-move to enemy base
            if self.enemy_units or self.enemy_structures:
                # Attack closest enemy unit/structure
                all_enemies = self.enemy_units | self.enemy_structures
                for marine in marines:
                    if marine.is_idle or marine.is_gathering:
                        closest_enemy = all_enemies.closest_to(marine)
                        marine.attack(closest_enemy)

                # All-in: Send workers too
                for worker in self.workers:
                    if worker.is_idle or worker.is_gathering:
                        closest_enemy = all_enemies.closest_to(worker)
                        worker.attack(closest_enemy)
            else:
                # No visible enemies, attack-move to enemy base to search
                for marine in marines:
                    if marine.is_idle or marine.is_gathering:
                        marine.attack(enemy_start)

                for worker in self.workers:
                    if worker.is_idle or worker.is_gathering:
                        worker.attack(enemy_start)

        # Even before attack starts, if marines are idle, send them forward
        elif marines:
            for marine in marines.idle:
                marine.attack(enemy_start)
