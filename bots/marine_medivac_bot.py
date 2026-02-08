"""
Marine/Medivac timing attack bot - Pro-style build order.

Follows a common professional Terran build:
1. 1 Barracks expand
2. 3 Barracks
3. Starport for Medivacs
4. Stim + Combat Shields
5. Drop attack at 5 minutes

This is a MUCH stronger opponent than IdleBot/RushBot.
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId


class MarineMedivacBot(BotAI):
    """
    Pro-style Marine/Medivac timing attack.

    Build order:
    - 14 Supply Depot
    - 15 Barracks
    - 16 Refinery
    - 17 Orbital Command
    - 18 Supply Depot
    - Expand to natural
    - 3 more Barracks
    - Tech Lab
    - Stim + Combat Shields
    - Starport + Reactor
    - Medivacs
    - Attack at 5 minutes
    """

    # Build order parameters
    MAX_WORKERS = 50
    TIMING_ATTACK_TIME = 300  # 5 minutes (in-game seconds)
    MARINE_COUNT_TO_ATTACK = 16

    def __init__(self):
        super().__init__()
        self.attack_started = False

    async def on_start(self):
        """Initialize."""
        self.client.game_step = 8

    async def on_step(self, iteration: int):
        """Main bot logic."""
        await self.distribute_workers()

        # Macro (economy + production)
        await self.build_workers()
        await self.build_supply()
        await self.expand()
        await self.build_gas()
        await self.build_production()
        await self.research_upgrades()
        await self.train_units()

        # Micro (attack logic)
        await self.attack_logic()

    async def build_workers(self):
        """Train SCVs up to max."""
        if self.supply_workers < self.MAX_WORKERS and self.supply_left > 0:
            for cc in self.townhalls.idle:
                if self.can_afford(UnitTypeId.SCV):
                    cc.train(UnitTypeId.SCV)

    async def build_supply(self):
        """Build supply depots proactively."""
        # Build multiple depots ahead of time (pro-level macro)
        if (
            self.supply_left < 6
            and self.supply_cap < 200
            and self.already_pending(UnitTypeId.SUPPLYDEPOT) < 2
        ):
            if self.can_afford(UnitTypeId.SUPPLYDEPOT):
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
        """Expand to natural base."""
        if (
            self.townhalls.amount < 2
            and self.can_afford(UnitTypeId.COMMANDCENTER)
            and self.already_pending(UnitTypeId.COMMANDCENTER) == 0
        ):
            # Expand to natural (closest expansion)
            location = await self.get_next_expansion()
            if location:
                workers = self.workers.gathering
                if workers:
                    worker = workers.closest_to(location)
                    worker.build(UnitTypeId.COMMANDCENTER, location)

    async def build_gas(self):
        """Build refineries on geysers."""
        # 2 gas per base
        for cc in self.townhalls.ready:
            vespene_geysers = self.vespene_geyser.closer_than(10, cc)
            for geyser in vespene_geysers:
                # Check if refinery already exists
                if not self.structures(UnitTypeId.REFINERY).closer_than(1, geyser):
                    if self.can_afford(UnitTypeId.REFINERY):
                        workers = self.workers.gathering
                        if workers:
                            worker = workers.closest_to(geyser)
                            worker.build(UnitTypeId.REFINERY, geyser)
                            return

    async def build_production(self):
        """Build barracks and starport."""
        # First barracks (wall off at ramp)
        if (
            self.structures(UnitTypeId.SUPPLYDEPOT).ready
            and self.structures(UnitTypeId.BARRACKS).amount < 1
        ):
            if self.can_afford(UnitTypeId.BARRACKS):
                workers = self.workers.gathering
                if workers:
                    worker = workers.closest_to(self.start_location)
                    location = await self.find_placement(
                        UnitTypeId.BARRACKS,
                        near=self.start_location.towards(self.game_info.map_center, 10),
                    )
                    if location:
                        worker.build(UnitTypeId.BARRACKS, location)

        # 3 more barracks after expand
        if (
            self.townhalls.amount >= 2
            and self.structures(UnitTypeId.BARRACKS).ready.amount < 4
        ):
            if self.can_afford(UnitTypeId.BARRACKS):
                workers = self.workers.gathering
                if workers:
                    worker = workers.closest_to(self.start_location)
                    location = await self.find_placement(
                        UnitTypeId.BARRACKS,
                        near=self.start_location.towards(self.game_info.map_center, 15),
                    )
                    if location:
                        worker.build(UnitTypeId.BARRACKS, location)

        # Tech lab on one barracks for upgrades
        for barracks in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.structures(UnitTypeId.BARRACKSTECHLAB).amount < 1:
                if self.can_afford(UnitTypeId.BARRACKSTECHLAB):
                    # Check if barracks can build addon (has space)
                    if barracks.is_ready and barracks.is_idle:
                        barracks.build(UnitTypeId.BARRACKSTECHLAB)
                        return

        # Starport for medivacs
        if (
            self.structures(UnitTypeId.BARRACKS).ready.amount >= 3
            and self.structures(UnitTypeId.STARPORT).amount < 1
        ):
            if self.can_afford(UnitTypeId.STARPORT):
                workers = self.workers.gathering
                if workers:
                    worker = workers.closest_to(self.start_location)
                    location = await self.find_placement(
                        UnitTypeId.STARPORT,
                        near=self.start_location.towards(self.game_info.map_center, 18),
                    )
                    if location:
                        worker.build(UnitTypeId.STARPORT, location)

        # Reactor on starport for double medivac production
        for starport in self.structures(UnitTypeId.STARPORT).ready.idle:
            if self.structures(UnitTypeId.STARPORTREACTOR).amount < 1:
                if self.can_afford(UnitTypeId.STARPORTREACTOR):
                    # Check if starport can build addon (has space)
                    if starport.is_ready and starport.is_idle:
                        starport.build(UnitTypeId.STARPORTREACTOR)
                        return

    async def research_upgrades(self):
        """Research stim and combat shields."""
        # Research from tech labs
        for techlab in self.structures(UnitTypeId.BARRACKSTECHLAB).ready.idle:
            # Stim
            if UpgradeId.STIMPACK not in self.state.upgrades:
                if self.can_afford(UpgradeId.STIMPACK):
                    techlab.research(UpgradeId.STIMPACK)
                    return

            # Combat shields
            if UpgradeId.SHIELDWALL not in self.state.upgrades:
                if self.can_afford(UpgradeId.SHIELDWALL):
                    techlab.research(UpgradeId.SHIELDWALL)
                    return

    async def train_units(self):
        """Train marines and medivacs."""
        # Train marines from all idle barracks
        for barracks in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.can_afford(UnitTypeId.MARINE) and self.supply_left > 0:
                barracks.train(UnitTypeId.MARINE)

        # Train medivacs from starport
        for starport in self.structures(UnitTypeId.STARPORT).ready.idle:
            if self.can_afford(UnitTypeId.MEDIVAC) and self.supply_left > 0:
                starport.train(UnitTypeId.MEDIVAC)

    async def attack_logic(self):
        """Attack with timing push."""
        marines = self.units(UnitTypeId.MARINE)
        medivacs = self.units(UnitTypeId.MEDIVAC)

        # Start attack when:
        # 1. We have enough marines (16+)
        # 2. OR game time is past timing window (5 minutes)
        if (
            len(marines) >= self.MARINE_COUNT_TO_ATTACK or self.time > self.TIMING_ATTACK_TIME
        ) and not self.attack_started:
            self.attack_started = True
            print(f"[{self.time:.0f}s] TIMING ATTACK! {len(marines)} marines, {len(medivacs)} medivacs")

        if self.attack_started:
            # Load marines into medivacs if available
            for medivac in medivacs:
                if medivac.cargo_used < medivac.cargo_max:
                    # Find nearby marines
                    nearby_marines = marines.closer_than(5, medivac)
                    if nearby_marines:
                        marine = nearby_marines.closest_to(medivac)
                        medivac.smart(marine)

            # Attack with all units
            enemy_start = self.enemy_start_locations[0]
            all_army = marines | medivacs

            if self.enemy_units or self.enemy_structures:
                # Target visible enemies
                all_enemies = self.enemy_units | self.enemy_structures
                for unit in all_army:
                    if unit.is_idle:
                        closest_enemy = all_enemies.closest_to(unit)
                        unit.attack(closest_enemy)
            else:
                # Search for enemies
                for unit in all_army:
                    if unit.is_idle:
                        unit.attack(enemy_start)
        else:
            # Defend before attack starts
            defend_pos = self.start_location.towards(self.game_info.map_center, 15)
            for unit in marines:
                if unit.distance_to(defend_pos) > 10:
                    unit.move(defend_pos)
