"""Macro-focused bot that prioritizes economy and expansion."""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2


# Tunable constants
MAX_WORKERS_PER_BASE = 16  # Optimal mineral saturation per base
EXPAND_AT_WORKERS = 14  # Expand when we have this many workers
BARRACKS_COUNT = 4  # More production once economy is strong
ATTACK_MARINE_THRESHOLD = 25  # Wait for even larger army
MIN_MARINES_BEFORE_EXPAND = 6  # Build some defense before expanding


class EconomyBot(BotAI):
    """
    Macro-focused economy bot.

    Strategy:
    1. Prioritize worker production (saturate bases)
    2. Fast expand to second base
    3. Build minimal defense while expanding
    4. Mass production once economy is strong
    5. Attack with overwhelming force
    """

    def __init__(self):
        super().__init__()
        self.attack_started = False

    async def on_start(self):
        """Initialize bot settings."""
        self.client.game_step = 2

    async def on_step(self, iteration: int):
        """Execute bot logic each game step."""
        await self.distribute_workers()
        await self.train_scvs()
        await self.build_supply_depots()
        await self.expand()
        await self.build_barracks()
        await self.train_marines()
        await self.attack()

    async def train_scvs(self):
        """Train SCVs to saturate all bases."""
        # Calculate target workers based on number of bases
        target_workers = len(self.townhalls) * MAX_WORKERS_PER_BASE

        if self.supply_workers >= target_workers:
            return

        # Train from all idle command centers
        for cc in self.townhalls.idle:
            if self.can_afford(UnitTypeId.SCV) and self.supply_workers < target_workers:
                cc.train(UnitTypeId.SCV)

    async def build_supply_depots(self):
        """Build supply depots proactively."""
        supply_pending = self.already_pending(UnitTypeId.SUPPLYDEPOT) * 8
        effective_supply_left = self.supply_left + supply_pending

        if (
            effective_supply_left < 8  # More buffer for macro-focused play
            and self.supply_cap < 200
            and self.can_afford(UnitTypeId.SUPPLYDEPOT)
            and self.already_pending(UnitTypeId.SUPPLYDEPOT) < 3  # Allow more pending
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

    async def expand(self):
        """Expand to new bases when economy allows."""
        # Don't expand if we already have 2+ bases or one pending
        if len(self.townhalls) + self.already_pending(UnitTypeId.COMMANDCENTER) >= 2:
            return

        # Expand conditions:
        # 1. Have enough workers at main base
        # 2. Have some marines for defense
        # 3. Can afford it
        marines = self.units(UnitTypeId.MARINE)
        if (
            self.supply_workers >= EXPAND_AT_WORKERS
            and len(marines) >= MIN_MARINES_BEFORE_EXPAND
            and self.can_afford(UnitTypeId.COMMANDCENTER)
        ):
            # Find expansion location
            location = await self.get_next_expansion()
            if location:
                workers = self.workers.gathering
                if workers:
                    worker = workers.closest_to(location)
                    worker.build(UnitTypeId.COMMANDCENTER, location)

    async def build_barracks(self):
        """Build multiple barracks once economy is established."""
        # Scale barracks with number of bases
        target_barracks = min(
            BARRACKS_COUNT, len(self.townhalls.ready) * 2
        )

        if self.structures(UnitTypeId.BARRACKS).amount >= target_barracks:
            return

        if not self.structures(UnitTypeId.SUPPLYDEPOT) and not self.already_pending(
            UnitTypeId.SUPPLYDEPOT
        ):
            return

        if (
            self.can_afford(UnitTypeId.BARRACKS)
            and self.already_pending(UnitTypeId.BARRACKS) < 2
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

    async def attack(self):
        """
        Attack with overwhelming force.

        - Build up large army before attacking
        - Keep some marines for base defense
        - Push with main force once threshold is reached
        """
        marines = self.units(UnitTypeId.MARINE)
        enemy_start = self.enemy_start_locations[0]

        # Start attack if we have overwhelming force
        if not self.attack_started:
            if len(marines) >= ATTACK_MARINE_THRESHOLD or not self.townhalls:
                self.attack_started = True
                print(f"[EconomyBot] Attack started with {len(marines)} marines!")

        # Attack mode
        if self.attack_started:
            # Send all but 5 marines to attack (keep some for defense)
            defense_reserve = 5
            attacking_marines = marines[:-defense_reserve] if len(marines) > defense_reserve else marines

            for marine in attacking_marines:
                if marine.is_idle or marine.distance_to(enemy_start) > 5:
                    marine.attack(enemy_start)

            # Keep reserve near main base
            if len(marines) > defense_reserve:
                for marine in marines[-defense_reserve:]:
                    if marine.distance_to(self.start_location) > 20:
                        marine.move(self.start_location)
        else:
            # Pre-attack: keep small defense force at each base
            for idx, townhall in enumerate(self.townhalls):
                # Assign 3 marines per base for defense
                defense_marines = marines[idx*3:(idx+1)*3]
                for marine in defense_marines:
                    if marine.distance_to(townhall.position) > 15:
                        marine.move(townhall.position)
