"""Siege tank focused bot with defensive positioning."""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId


# Tunable constants
MAX_WORKERS = 20
FACTORY_COUNT = 2
TANK_COUNT_ATTACK = 4  # Attack with this many tanks
MARINE_COUNT = 10  # Supporting marines


class TankBot(BotAI):
    """
    Siege tank focused strategy.

    Strategy:
    1. Build economy (20 workers)
    2. Build factory for siege tanks
    3. Build barracks for marines (support)
    4. Mass tanks with marine support
    5. Siege near enemy and push
    """

    def __init__(self):
        super().__init__()
        self.attack_started = False
        self.siege_position = None

    async def on_start(self):
        """Initialize bot settings."""
        self.client.game_step = 2
        # Set siege position halfway to enemy
        enemy_start = self.enemy_start_locations[0]
        self.siege_position = self.start_location.towards(enemy_start, 40)

    async def on_step(self, iteration: int):
        """Execute bot logic each game step."""
        await self.distribute_workers()
        await self.train_scvs()
        await self.build_supply_depots()
        await self.build_barracks()
        await self.build_factory()
        await self.train_units()
        await self.manage_tanks()
        await self.attack()

    async def train_scvs(self):
        """Train SCVs up to MAX_WORKERS."""
        if self.supply_workers >= MAX_WORKERS:
            return

        for cc in self.townhalls.idle:
            if self.can_afford(UnitTypeId.SCV):
                cc.train(UnitTypeId.SCV)

    async def build_supply_depots(self):
        """Build supply depots proactively."""
        supply_pending = self.already_pending(UnitTypeId.SUPPLYDEPOT) * 8
        effective_supply_left = self.supply_left + supply_pending

        if (
            effective_supply_left < 8  # More buffer for tank production
            and self.supply_cap < 200
            and self.can_afford(UnitTypeId.SUPPLYDEPOT)
            and self.already_pending(UnitTypeId.SUPPLYDEPOT) < 3
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

    async def build_barracks(self):
        """Build one barracks for marine support."""
        if self.structures(UnitTypeId.BARRACKS).amount >= 1:
            return

        if not self.structures(UnitTypeId.SUPPLYDEPOT):
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

    async def build_factory(self):
        """Build factories for tank production."""
        if self.structures(UnitTypeId.FACTORY).amount >= FACTORY_COUNT:
            return

        # Need barracks before factory
        if not self.structures(UnitTypeId.BARRACKS).ready:
            return

        if (
            self.can_afford(UnitTypeId.FACTORY)
            and self.already_pending(UnitTypeId.FACTORY) < FACTORY_COUNT
        ):
            workers = self.workers.gathering
            if workers:
                worker = workers.closest_to(self.start_location)
                location = await self.find_placement(
                    UnitTypeId.FACTORY,
                    near=self.start_location.towards(self.game_info.map_center, 10),
                )
                if location:
                    worker.build(UnitTypeId.FACTORY, location)
                    print("[TankBot] Building factory...")

    async def train_units(self):
        """Train tanks from factories and marines from barracks."""
        # Prioritize tanks
        for factory in self.structures(UnitTypeId.FACTORY).ready.idle:
            if self.can_afford(UnitTypeId.SIEGETANK):
                factory.train(UnitTypeId.SIEGETANK)

        # Train marines for support (up to MARINE_COUNT)
        if self.units(UnitTypeId.MARINE).amount < MARINE_COUNT:
            for barracks in self.structures(UnitTypeId.BARRACKS).ready.idle:
                if self.can_afford(UnitTypeId.MARINE):
                    barracks.train(UnitTypeId.MARINE)

    async def manage_tanks(self):
        """Handle tank siege/unsiege logic."""
        tanks = self.units(UnitTypeId.SIEGETANK)
        sieged_tanks = self.units(UnitTypeId.SIEGETANKSIEGED)
        enemy_start = self.enemy_start_locations[0]

        # If we're attacking, siege tanks near enemies
        if self.attack_started:
            # Siege tanks near enemy units
            if self.enemy_units or self.enemy_structures:
                all_enemies = self.enemy_units | self.enemy_structures

                for tank in tanks:
                    # Siege if near enemies
                    if all_enemies.closer_than(13, tank):  # Tank range is 13
                        tank(AbilityId.SIEGEMODE_SIEGEMODE)

                # Unsiege tanks that are too far from action
                for tank in sieged_tanks:
                    if not all_enemies.closer_than(15, tank):
                        tank(AbilityId.UNSIEGE_UNSIEGE)

        # Defensive: keep tanks near siege position
        else:
            for tank in tanks:
                if tank.distance_to(self.siege_position) < 5:
                    # Siege tanks at defensive position
                    tank(AbilityId.SIEGEMODE_SIEGEMODE)
                elif tank.distance_to(self.siege_position) > 10:
                    # Move tanks to defensive position
                    tank.move(self.siege_position)

    async def attack(self):
        """
        Attack with tank/marine army.

        - Wait for 4+ tanks
        - Siege near enemy
        - Marines protect tanks
        """
        tanks = self.units(UnitTypeId.SIEGETANK)
        sieged_tanks = self.units(UnitTypeId.SIEGETANKSIEGED)
        all_tanks = tanks | sieged_tanks
        marines = self.units(UnitTypeId.MARINE)
        enemy_start = self.enemy_start_locations[0]

        # Start attack when we have enough tanks
        if not self.attack_started:
            if len(all_tanks) >= TANK_COUNT_ATTACK or not self.townhalls:
                self.attack_started = True
                print(f"[TankBot] Attack started with {len(all_tanks)} tanks!")

        # Attack mode
        if self.attack_started:
            if self.enemy_units or self.enemy_structures:
                all_enemies = self.enemy_units | self.enemy_structures

                # Move unsieged tanks toward enemy
                for tank in tanks:
                    if tank.is_idle:
                        # Move to siege range
                        closest_enemy = all_enemies.closest_to(tank)
                        if tank.distance_to(closest_enemy) > 12:
                            tank.move(closest_enemy.position)
                        else:
                            tank(AbilityId.SIEGEMODE_SIEGEMODE)

                # Marines: protect tanks and attack
                for marine in marines:
                    if marine.is_idle:
                        closest_enemy = all_enemies.closest_to(marine)
                        marine.attack(closest_enemy)
            else:
                # No visible enemies, push forward
                for tank in tanks:
                    if tank.is_idle:
                        tank.attack(enemy_start)

                for marine in marines:
                    if marine.is_idle:
                        marine.attack(enemy_start)

        # Pre-attack: defensive positioning
        else:
            # Keep tanks at siege position
            for tank in tanks:
                if tank.distance_to(self.siege_position) > 10 and tank.is_idle:
                    tank.move(self.siege_position)

            # Marines defend siege position
            for marine in marines:
                if marine.distance_to(self.siege_position) > 15 and marine.is_idle:
                    marine.move(self.siege_position)
