"""Defensive turtle bot that masses units before attacking."""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2


# Tunable constants
MAX_WORKERS = 20  # Economic cap for defense-focused play
BUNKER_COUNT = 2  # Number of bunkers to build
BARRACKS_COUNT = 3  # Number of barracks
ATTACK_MARINE_THRESHOLD = 20  # Wait for larger army before attacking
DEFENSE_RADIUS = 25  # How far from base to consider "defending"


class DefenseBot(BotAI):
    """
    Defensive turtle bot.

    Strategy:
    1. Build economy up to 20 workers
    2. Build bunkers near base entrance
    3. Mass marines and keep them near base
    4. Only attack when army is large (20+ marines)
    5. Defend aggressively if enemy enters base
    """

    def __init__(self):
        super().__init__()
        self.attack_started = False
        self.defense_position = None

    async def on_start(self):
        """Initialize bot settings."""
        self.client.game_step = 2
        # Set defense position towards map center from our base
        self.defense_position = self.start_location.towards(
            self.game_info.map_center, 10
        )

    async def on_step(self, iteration: int):
        """Execute bot logic each game step."""
        await self.distribute_workers()
        await self.train_scvs()
        await self.build_supply_depots()
        await self.build_bunkers()
        await self.build_barracks()
        await self.train_marines()
        await self.defend_and_attack()

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
            effective_supply_left < 6
            and self.supply_cap < 200
            and self.can_afford(UnitTypeId.SUPPLYDEPOT)
            and self.already_pending(UnitTypeId.SUPPLYDEPOT) < 2
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

    async def build_bunkers(self):
        """Build defensive bunkers near base entrance."""
        if self.structures(UnitTypeId.BUNKER).amount >= BUNKER_COUNT:
            return

        # Need barracks before bunker
        if not self.structures(UnitTypeId.BARRACKS):
            return

        if (
            self.can_afford(UnitTypeId.BUNKER)
            and self.already_pending(UnitTypeId.BUNKER) < 1
        ):
            workers = self.workers.gathering
            if workers:
                worker = workers.closest_to(self.start_location)
                # Build bunker towards map center (defensive position)
                location = await self.find_placement(
                    UnitTypeId.BUNKER,
                    near=self.defense_position,
                )
                if location:
                    worker.build(UnitTypeId.BUNKER, location)

    async def build_barracks(self):
        """Build barracks for marine production."""
        if self.structures(UnitTypeId.BARRACKS).amount >= BARRACKS_COUNT:
            return

        if not self.structures(UnitTypeId.SUPPLYDEPOT) and not self.already_pending(
            UnitTypeId.SUPPLYDEPOT
        ):
            return

        if (
            self.can_afford(UnitTypeId.BARRACKS)
            and self.already_pending(UnitTypeId.BARRACKS) < BARRACKS_COUNT
        ):
            workers = self.workers.gathering
            if workers:
                worker = workers.closest_to(self.start_location)
                location = await self.find_placement(
                    UnitTypeId.BARRACKS,
                    near=self.start_location.towards(self.game_info.map_center, 12),
                )
                if location:
                    worker.build(UnitTypeId.BARRACKS, location)

    async def train_marines(self):
        """Train marines from all idle barracks."""
        for barracks in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.can_afford(UnitTypeId.MARINE):
                barracks.train(UnitTypeId.MARINE)

    async def defend_and_attack(self):
        """
        Defensive strategy with counter-attack.

        - Keep marines near base for defense
        - Load marines into bunkers if available
        - Attack aggressively if enemy enters base
        - Only push out once army is large enough
        """
        marines = self.units(UnitTypeId.MARINE)
        bunkers = self.structures(UnitTypeId.BUNKER).ready
        enemy_start = self.enemy_start_locations[0]

        # Load bunkers with marines if under attack
        if bunkers:
            for bunker in bunkers:
                if bunker.cargo_used < bunker.cargo_max:
                    # Find nearby marines to load
                    nearby_marines = marines.closer_than(5, bunker.position)
                    for marine in nearby_marines:
                        if len(bunker.passengers) < bunker.cargo_max:
                            marine.move(bunker)
                            break

        # Check for enemies near our base
        enemy_near_base = False
        if self.enemy_units:
            for enemy in self.enemy_units:
                if enemy.distance_to(self.start_location) < DEFENSE_RADIUS:
                    enemy_near_base = True
                    break

        # PRIORITY: Defend if enemies are near
        if enemy_near_base:
            # Unload bunkers to defend
            for bunker in bunkers:
                if bunker.cargo_used > 0:
                    bunker(AbilityId.UNLOADALL_BUNKER)

            # Send all marines to defend
            for marine in marines:
                # Find closest enemy
                if self.enemy_units:
                    closest_enemy = self.enemy_units.closest_to(marine)
                    marine.attack(closest_enemy.position)
            return

        # Start attack if we have enough marines
        if not self.attack_started:
            if len(marines) >= ATTACK_MARINE_THRESHOLD:
                self.attack_started = True
                print(f"[DefenseBot] Attack started with {len(marines)} marines!")

                # Unload bunkers for attack
                for bunker in bunkers:
                    if bunker.cargo_used > 0:
                        bunker(AbilityId.UNLOADALL_BUNKER)

        # Attack mode: push to enemy base
        if self.attack_started:
            for marine in marines:
                if marine.is_idle:
                    marine.attack(enemy_start)
        else:
            # Defensive positioning: keep marines near defense position
            for marine in marines:
                if marine.distance_to(self.defense_position) > 15:
                    marine.move(self.defense_position)
