"""Marine rush bot with stim pack upgrade for enhanced combat."""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.ability_id import AbilityId


# Tunable constants
MAX_WORKERS = 18
BARRACKS_COUNT = 3
ATTACK_MARINE_THRESHOLD = 10  # Attack with stimmed marines


class StimBot(BotAI):
    """
    Enhanced marine rush with stim pack.

    Strategy:
    1. Build economy (18 workers)
    2. Build barracks with tech labs
    3. Research stim pack
    4. Mass marines with stim
    5. Attack with stimmed marines (2x attack speed!)
    """

    def __init__(self):
        super().__init__()
        self.attack_started = False
        self.stim_started = False

    async def on_start(self):
        """Initialize bot settings."""
        self.client.game_step = 2

    async def on_step(self, iteration: int):
        """Execute bot logic each game step."""
        await self.distribute_workers()
        await self.train_scvs()
        await self.build_supply_depots()
        await self.build_barracks()
        await self.upgrade_tech_lab()
        await self.research_stim()
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

    async def upgrade_tech_lab(self):
        """Add tech lab to first barracks for stim research."""
        # Only need one tech lab for stim research
        if self.structures(UnitTypeId.BARRACKSTECHLAB).ready:
            return

        if self.already_pending(UnitTypeId.BARRACKSTECHLAB):
            return

        # Find barracks without addon
        for rax in self.structures(UnitTypeId.BARRACKS).ready:
            if not rax.has_add_on and rax.is_idle:
                if self.can_afford(UnitTypeId.BARRACKSTECHLAB):
                    rax.build(UnitTypeId.BARRACKSTECHLAB)
                    return

    async def research_stim(self):
        """Research stim pack upgrade."""
        # Check if already researched or researching
        if (
            UpgradeId.STIMPACK in self.state.upgrades
            or self.already_pending_upgrade(UpgradeId.STIMPACK) > 0
        ):
            self.stim_started = True
            return

        # Need tech lab to research
        tech_labs = self.structures(UnitTypeId.BARRACKSTECHLAB).ready
        if not tech_labs:
            return

        # Research stim
        if self.can_afford(UpgradeId.STIMPACK):
            tech_labs.first.research(UpgradeId.STIMPACK)
            print("[StimBot] Researching stim pack!")

    async def train_marines(self):
        """Train marines from all idle barracks."""
        for barracks in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.can_afford(UnitTypeId.MARINE):
                barracks.train(UnitTypeId.MARINE)

    async def attack(self):
        """
        Attack with stimmed marines.

        - Wait for stim to research
        - Attack with 10+ marines
        - Use stim in combat!
        """
        marines = self.units(UnitTypeId.MARINE)
        enemy_start = self.enemy_start_locations[0]
        has_stim = UpgradeId.STIMPACK in self.state.upgrades

        # Start attack if we have enough marines and stim is done
        if not self.attack_started:
            if (len(marines) >= ATTACK_MARINE_THRESHOLD and has_stim) or not self.townhalls:
                self.attack_started = True
                print(f"[StimBot] Attack started with {len(marines)} stimmed marines!")

        # Attack mode
        if self.attack_started or (len(marines) >= ATTACK_MARINE_THRESHOLD * 1.5):
            # Target visible enemies first
            if self.enemy_units or self.enemy_structures:
                all_enemies = self.enemy_units | self.enemy_structures

                for marine in marines:
                    # Use stim if available and in combat
                    if (
                        has_stim
                        and marine.health_percentage > 0.5  # Don't stim if low HP
                        and not marine.has_buff(AbilityId.EFFECT_STIM)
                        and all_enemies.closer_than(10, marine)
                    ):
                        marine(AbilityId.EFFECT_STIM)

                    if marine.is_idle or marine.is_gathering:
                        closest_enemy = all_enemies.closest_to(marine)
                        marine.attack(closest_enemy)

                # Send workers too (all-in)
                for worker in self.workers:
                    if worker.is_idle or worker.is_gathering:
                        closest_enemy = all_enemies.closest_to(worker)
                        worker.attack(closest_enemy)
            else:
                # No visible enemies, attack-move to search
                for marine in marines:
                    if marine.is_idle or marine.is_gathering:
                        marine.attack(enemy_start)

                for worker in self.workers:
                    if worker.is_idle or worker.is_gathering:
                        worker.attack(enemy_start)

        # Pre-attack: keep marines near base
        elif marines:
            for marine in marines.idle:
                marine.move(self.start_location.towards(self.game_info.map_center, 10))
