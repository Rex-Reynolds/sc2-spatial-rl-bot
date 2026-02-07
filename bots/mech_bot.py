"""Pro-style mech bot with Tanks, Hellions, and Thors based on professional build orders."""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.ability_id import AbilityId


# Tunable constants - based on pro builds
MAX_WORKERS = 28  # Mech needs good economy
FACTORY_COUNT = 3  # Multiple factories for production
STARPORT_COUNT = 1  # For Viking support
TANK_TARGET = 6  # Target number of tanks
HELLION_TARGET = 8  # Target number of hellions
THOR_TARGET = 2  # Target number of thors
ATTACK_SUPPLY_THRESHOLD = 80  # Wait for large mech army


class MechBot(BotAI):
    """
    Professional mech composition bot.

    Build order inspired by pro players:
    1. Fast expand (2 bases)
    2. 3 factories (hellion/tank production)
    3. Armory for upgrades
    4. Composition: Tanks (siege), Hellions (speed), Thors (anti-air/heavy)
    5. Slow push with sieged tanks
    """

    def __init__(self):
        super().__init__()
        self.attack_started = False
        self.expand_started = False
        self.siege_line = None

    async def on_start(self):
        """Initialize bot settings."""
        self.client.game_step = 2
        enemy_start = self.enemy_start_locations[0]
        self.siege_line = self.start_location.towards(enemy_start, 50)

    async def on_step(self, iteration: int):
        """Execute bot logic each game step."""
        await self.distribute_workers()
        await self.train_scvs()
        await self.build_supply_depots()
        await self.expand()
        await self.build_production()
        await self.build_armory()
        await self.research_upgrades()
        await self.train_army()
        await self.manage_hellions()
        await self.manage_tanks()
        await self.attack()

    async def train_scvs(self):
        """Train SCVs up to MAX_WORKERS across all bases."""
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
            effective_supply_left < 12  # Large buffer for mech
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

    async def expand(self):
        """Expand to second base (crucial for mech economy)."""
        if len(self.townhalls) + self.already_pending(UnitTypeId.COMMANDCENTER) >= 2:
            self.expand_started = True
            return

        # Expand when we have 16 workers
        if self.supply_workers >= 16 and self.can_afford(UnitTypeId.COMMANDCENTER):
            location = await self.get_next_expansion()
            if location:
                workers = self.workers.gathering
                if workers:
                    worker = workers.closest_to(location)
                    worker.build(UnitTypeId.COMMANDCENTER, location)
                    print("[MechBot] Expanding to second base...")

    async def build_production(self):
        """Build factories and starport."""
        # Build one barracks (requirement for factory)
        if (
            not self.structures(UnitTypeId.BARRACKS)
            and not self.already_pending(UnitTypeId.BARRACKS)
            and self.structures(UnitTypeId.SUPPLYDEPOT).ready
        ):
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

        # Build factories (main production)
        if self.structures(UnitTypeId.FACTORY).amount < FACTORY_COUNT:
            if (
                self.structures(UnitTypeId.BARRACKS).ready
                and self.can_afford(UnitTypeId.FACTORY)
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

        # Build starport (for Vikings/support)
        if self.structures(UnitTypeId.STARPORT).amount < STARPORT_COUNT:
            if (
                self.structures(UnitTypeId.FACTORY).ready
                and self.can_afford(UnitTypeId.STARPORT)
            ):
                workers = self.workers.gathering
                if workers:
                    worker = workers.closest_to(self.start_location)
                    location = await self.find_placement(
                        UnitTypeId.STARPORT,
                        near=self.start_location.towards(self.game_info.map_center, 8),
                    )
                    if location:
                        worker.build(UnitTypeId.STARPORT, location)

    async def build_armory(self):
        """Build armory for vehicle upgrades."""
        if self.structures(UnitTypeId.ARMORY) or self.already_pending(UnitTypeId.ARMORY):
            return

        # Build after first factory is done
        if self.structures(UnitTypeId.FACTORY).ready:
            if self.can_afford(UnitTypeId.ARMORY):
                workers = self.workers.gathering
                if workers:
                    worker = workers.closest_to(self.start_location)
                    location = await self.find_placement(
                        UnitTypeId.ARMORY,
                        near=self.start_location.towards(self.game_info.map_center, 9),
                    )
                    if location:
                        worker.build(UnitTypeId.ARMORY, location)
                        print("[MechBot] Building armory for upgrades...")

    async def research_upgrades(self):
        """Research vehicle upgrades at armory."""
        armories = self.structures(UnitTypeId.ARMORY).ready
        if not armories:
            return

        armory = armories.first

        # Vehicle weapons upgrade
        if (
            UpgradeId.TERRANVEHICLEWEAPONSLEVEL1 not in self.state.upgrades
            and self.already_pending_upgrade(UpgradeId.TERRANVEHICLEWEAPONSLEVEL1) == 0
            and self.can_afford(UpgradeId.TERRANVEHICLEWEAPONSLEVEL1)
        ):
            armory.research(UpgradeId.TERRANVEHICLEWEAPONSLEVEL1)
            print("[MechBot] Researching +1 vehicle weapons!")

        # Vehicle armor upgrade
        elif (
            UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL1 not in self.state.upgrades
            and self.already_pending_upgrade(UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL1) == 0
            and self.can_afford(UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL1)
        ):
            armory.research(UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL1)
            print("[MechBot] Researching +1 vehicle armor!")

    async def train_army(self):
        """Train mech units: Hellions, Tanks, Thors."""
        tanks = self.units(UnitTypeId.SIEGETANK).amount + self.units(UnitTypeId.SIEGETANKSIEGED).amount
        hellions = self.units(UnitTypeId.HELLION).amount
        thors = self.units(UnitTypeId.THOR).amount

        for factory in self.structures(UnitTypeId.FACTORY).ready.idle:
            # Priority 1: Build thors (need tech lab)
            if thors < THOR_TARGET:
                if factory.has_add_on and self.can_afford(UnitTypeId.THOR):
                    factory.train(UnitTypeId.THOR)
                    continue

            # Priority 2: Build tanks
            if tanks < TANK_TARGET:
                if factory.has_add_on and self.can_afford(UnitTypeId.SIEGETANK):
                    factory.train(UnitTypeId.SIEGETANK)
                    continue

            # Priority 3: Build hellions
            if hellions < HELLION_TARGET:
                if self.can_afford(UnitTypeId.HELLION):
                    factory.train(UnitTypeId.HELLION)
                    continue

            # Default: More hellions or tanks based on what we need
            if hellions < tanks:
                if self.can_afford(UnitTypeId.HELLION):
                    factory.train(UnitTypeId.HELLION)
            else:
                if factory.has_add_on and self.can_afford(UnitTypeId.SIEGETANK):
                    factory.train(UnitTypeId.SIEGETANK)

        # Add tech lab to factories for tanks/thors
        tech_lab_count = self.structures(UnitTypeId.FACTORYTECHLAB).amount
        if tech_lab_count < 2:
            for factory in self.structures(UnitTypeId.FACTORY).ready:
                if not factory.has_add_on and factory.is_idle:
                    if self.can_afford(UnitTypeId.FACTORYTECHLAB):
                        factory.build(UnitTypeId.FACTORYTECHLAB)
                        break

    async def manage_hellions(self):
        """Hellions: fast harassment and screening."""
        hellions = self.units(UnitTypeId.HELLION)

        if not self.attack_started:
            # Pre-attack: keep hellions mobile, look for openings
            for hellion in hellions:
                if hellion.is_idle:
                    # Patrol between bases
                    hellion.move(self.start_location.towards(self.enemy_start_locations[0], 30))
        else:
            # During attack: hellions lead the charge, tanks follow
            if self.enemy_units:
                for hellion in hellions:
                    closest_enemy = self.enemy_units.closest_to(hellion)
                    hellion.attack(closest_enemy)

    async def manage_tanks(self):
        """Tanks: siege and push forward."""
        tanks = self.units(UnitTypeId.SIEGETANK)
        sieged_tanks = self.units(UnitTypeId.SIEGETANKSIEGED)

        if self.attack_started and (self.enemy_units or self.enemy_structures):
            all_enemies = self.enemy_units | self.enemy_structures

            # Siege tanks near enemies
            for tank in tanks:
                if all_enemies.closer_than(13, tank):
                    tank(AbilityId.SIEGEMODE_SIEGEMODE)
                elif tank.is_idle:
                    closest_enemy = all_enemies.closest_to(tank)
                    if tank.distance_to(closest_enemy) > 12:
                        tank.move(closest_enemy.position)

            # Unsiege and advance if enemies are dead nearby
            for tank in sieged_tanks:
                if not all_enemies.closer_than(13, tank):
                    tank(AbilityId.UNSIEGE_UNSIEGE)
        else:
            # Defensive: siege at siege line
            for tank in tanks:
                if tank.distance_to(self.siege_line) < 3:
                    tank(AbilityId.SIEGEMODE_SIEGEMODE)
                elif tank.distance_to(self.siege_line) > 10:
                    tank.move(self.siege_line)

    async def attack(self):
        """
        Attack with mech army once we have critical mass.

        - Wait for 80+ supply (tanks/hellions/thors)
        - Slow push with sieged tanks
        - Hellions screen and harass
        - Thors provide anti-air and heavy firepower
        """
        tanks = self.units(UnitTypeId.SIEGETANK) | self.units(UnitTypeId.SIEGETANKSIEGED)
        hellions = self.units(UnitTypeId.HELLION)
        thors = self.units(UnitTypeId.THOR)
        mech_army = tanks | hellions | thors

        # Calculate mech supply
        army_supply = (
            tanks.amount * 3 +  # Tanks cost 3 supply
            hellions.amount * 2 +  # Hellions cost 2 supply
            thors.amount * 6  # Thors cost 6 supply
        )

        # Start attack with large mech force
        if not self.attack_started:
            if army_supply >= ATTACK_SUPPLY_THRESHOLD or not self.townhalls:
                self.attack_started = True
                print(f"[MechBot] MECH PUSH! {army_supply} supply army moving out!")

        if self.attack_started:
            enemy_start = self.enemy_start_locations[0]

            # Thors: focus on air units if present, otherwise attack ground
            for thor in thors:
                if thor.is_idle:
                    if self.enemy_units.flying:
                        closest_air = self.enemy_units.flying.closest_to(thor)
                        thor.attack(closest_air)
                    elif self.enemy_units or self.enemy_structures:
                        all_enemies = self.enemy_units | self.enemy_structures
                        closest_enemy = all_enemies.closest_to(thor)
                        thor.attack(closest_enemy)
                    else:
                        thor.attack(enemy_start)
