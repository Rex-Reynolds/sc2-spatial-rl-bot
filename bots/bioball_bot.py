"""Classic Terran bio ball with Marines, Marauders, and Medivacs (MMM)."""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.ability_id import AbilityId


# Tunable constants
MAX_WORKERS = 24
BARRACKS_COUNT = 4
STARPORT_COUNT = 1
ATTACK_THRESHOLD = 20  # Total army supply before attacking
MARINE_RATIO = 0.6  # 60% marines
MARAUDER_RATIO = 0.3  # 30% marauders
MEDIVAC_RATIO = 0.1  # 10% medivacs (one per ~10 ground units)


class BioBallBot(BotAI):
    """
    Classic Terran bio ball composition.

    Strategy:
    1. Strong economy (24 workers)
    2. Multiple barracks with mix of tech labs and reactors
    3. Starport for medivacs
    4. 60% Marines, 30% Marauders, 10% Medivacs
    5. Stim + Combat Shields
    6. Attack with large army
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
        await self.build_barracks()
        await self.build_starport()
        await self.upgrade_barracks()
        await self.research_upgrades()
        await self.train_army()
        await self.manage_medivacs()
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
            effective_supply_left < 10  # Large buffer for big army
            and self.supply_cap < 200
            and self.can_afford(UnitTypeId.SUPPLYDEPOT)
            and self.already_pending(UnitTypeId.SUPPLYDEPOT) < 4
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
        """Build multiple barracks for production."""
        if self.structures(UnitTypeId.BARRACKS).amount >= BARRACKS_COUNT:
            return

        if not self.structures(UnitTypeId.SUPPLYDEPOT):
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

    async def build_starport(self):
        """Build starport for medivac production."""
        if self.structures(UnitTypeId.STARPORT).amount >= STARPORT_COUNT:
            return

        # Need barracks first
        if not self.structures(UnitTypeId.BARRACKS).ready:
            return

        if self.can_afford(UnitTypeId.STARPORT):
            workers = self.workers.gathering
            if workers:
                worker = workers.closest_to(self.start_location)
                location = await self.find_placement(
                    UnitTypeId.STARPORT,
                    near=self.start_location.towards(self.game_info.map_center, 10),
                )
                if location:
                    worker.build(UnitTypeId.STARPORT, location)
                    print("[BioBallBot] Building starport for medivacs...")

    async def upgrade_barracks(self):
        """Add tech labs and reactors to barracks."""
        barracks = self.structures(UnitTypeId.BARRACKS).ready

        if not barracks:
            return

        # Count addons
        tech_labs = self.structures(UnitTypeId.BARRACKSTECHLAB).amount
        reactors = self.structures(UnitTypeId.BARRACKSREACTOR).amount

        # Want 2 tech labs (for marauders + research), rest reactors (marines)
        for rax in barracks:
            if rax.has_add_on or not rax.is_idle:
                continue

            if tech_labs < 2 and self.can_afford(UnitTypeId.BARRACKSTECHLAB):
                rax.build(UnitTypeId.BARRACKSTECHLAB)
            elif self.can_afford(UnitTypeId.BARRACKSREACTOR):
                rax.build(UnitTypeId.BARRACKSREACTOR)

    async def research_upgrades(self):
        """Research stim pack and combat shields."""
        tech_labs = self.structures(UnitTypeId.BARRACKSTECHLAB).ready
        if not tech_labs:
            return

        # Stim first (priority)
        if (
            UpgradeId.STIMPACK not in self.state.upgrades
            and self.already_pending_upgrade(UpgradeId.STIMPACK) == 0
            and self.can_afford(UpgradeId.STIMPACK)
        ):
            tech_labs.first.research(UpgradeId.STIMPACK)
            print("[BioBallBot] Researching stim pack!")

        # Combat shields second
        elif (
            UpgradeId.SHIELDWALL not in self.state.upgrades
            and self.already_pending_upgrade(UpgradeId.SHIELDWALL) == 0
            and self.can_afford(UpgradeId.SHIELDWALL)
        ):
            tech_labs.first.research(UpgradeId.SHIELDWALL)
            print("[BioBallBot] Researching combat shields!")

    async def train_army(self):
        """Train marines, marauders based on ratios."""
        marines = self.units(UnitTypeId.MARINE).amount
        marauders = self.units(UnitTypeId.MARAUDER).amount
        total_ground = marines + marauders

        if total_ground == 0:
            target_marines = 1
            target_marauders = 0
        else:
            target_marine_ratio = marines / total_ground
            target_marauders = marauders / total_ground

        # Train from barracks
        for rax in self.structures(UnitTypeId.BARRACKS).ready.idle:
            # Barracks with tech lab: train marauders
            if rax.has_add_on and rax.add_on_tag in [
                addon.tag for addon in self.structures(UnitTypeId.BARRACKSTECHLAB)
            ]:
                if total_ground < 5 or target_marauders < MARAUDER_RATIO:
                    if self.can_afford(UnitTypeId.MARAUDER):
                        rax.train(UnitTypeId.MARAUDER)
                else:
                    if self.can_afford(UnitTypeId.MARINE):
                        rax.train(UnitTypeId.MARINE)

            # Barracks with reactor: train marines (2 at once!)
            elif rax.has_add_on and rax.add_on_tag in [
                addon.tag for addon in self.structures(UnitTypeId.BARRACKSREACTOR)
            ]:
                if self.can_afford(UnitTypeId.MARINE):
                    rax.train(UnitTypeId.MARINE)

            # No addon: train marines
            else:
                if self.can_afford(UnitTypeId.MARINE):
                    rax.train(UnitTypeId.MARINE)

        # Train medivacs from starport
        medivacs = self.units(UnitTypeId.MEDIVAC).amount
        target_medivacs = max(1, total_ground // 10)  # One medivac per 10 units

        if medivacs < target_medivacs:
            for starport in self.structures(UnitTypeId.STARPORT).ready.idle:
                if self.can_afford(UnitTypeId.MEDIVAC):
                    starport.train(UnitTypeId.MEDIVAC)

    async def manage_medivacs(self):
        """Keep medivacs with the army and heal."""
        medivacs = self.units(UnitTypeId.MEDIVAC)
        marines = self.units(UnitTypeId.MARINE)
        marauders = self.units(UnitTypeId.MARAUDER)
        ground_army = marines | marauders

        if not ground_army:
            return

        # Calculate army center
        army_center = ground_army.center

        # Keep medivacs with army
        for medivac in medivacs:
            if medivac.distance_to(army_center) > 8:
                medivac.move(army_center)

    async def attack(self):
        """
        Attack with bio ball.

        - Wait for large army (20+ supply)
        - Use stim in combat
        - Medivacs heal continuously
        """
        marines = self.units(UnitTypeId.MARINE)
        marauders = self.units(UnitTypeId.MARAUDER)
        bio_army = marines | marauders
        enemy_start = self.enemy_start_locations[0]
        has_stim = UpgradeId.STIMPACK in self.state.upgrades

        # Calculate army supply
        army_supply = len(marines) + marauders.amount * 2  # Marauders cost 2 supply

        # Start attack with large army
        if not self.attack_started:
            if army_supply >= ATTACK_THRESHOLD or not self.townhalls:
                self.attack_started = True
                print(f"[BioBallBot] Attack started with {army_supply} supply army!")

        # Attack mode
        if self.attack_started:
            if self.enemy_units or self.enemy_structures:
                all_enemies = self.enemy_units | self.enemy_structures

                for unit in bio_army:
                    # Use stim if available and in combat
                    if (
                        has_stim
                        and unit.health_percentage > 0.5
                        and not unit.has_buff(AbilityId.EFFECT_STIM)
                        and all_enemies.closer_than(10, unit)
                    ):
                        unit(AbilityId.EFFECT_STIM)

                    if unit.is_idle or unit.is_gathering:
                        closest_enemy = all_enemies.closest_to(unit)
                        unit.attack(closest_enemy)
            else:
                # No visible enemies, push forward
                for unit in bio_army:
                    if unit.is_idle:
                        unit.attack(enemy_start)

        # Pre-attack: keep army together
        elif bio_army:
            army_center = bio_army.center
            for unit in bio_army:
                if unit.distance_to(army_center) > 20 and unit.is_idle:
                    unit.move(army_center)
