"""Aggressive proxy bot that builds near enemy base."""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2


# Tunable constants
MAX_WORKERS = 12  # Minimal economy for all-in
PROXY_BARRACKS_COUNT = 2  # Barracks built near enemy
HOME_BARRACKS_COUNT = 1  # One barracks at home for safety
ATTACK_MARINE_THRESHOLD = 4  # Attack ASAP with proxy
PROXY_DISTANCE = 30  # How close to enemy to build proxy


class ProxyBot(BotAI):
    """
    Aggressive proxy rush bot.

    Strategy:
    1. Send SCV to enemy base early
    2. Build proxy barracks near enemy
    3. Minimal economy (just enough SCVs)
    4. Rush with first marines
    5. Keep up constant pressure
    6. All-in strategy - win fast or lose
    """

    def __init__(self):
        super().__init__()
        self.proxy_scv = None
        self.proxy_location = None
        self.proxy_established = False
        self.attack_started = False

    async def on_start(self):
        """Initialize bot settings."""
        self.client.game_step = 2

        # Calculate proxy location (near enemy base)
        enemy_start = self.enemy_start_locations[0]
        self.proxy_location = self.start_location.towards(enemy_start, PROXY_DISTANCE)

    async def on_step(self, iteration: int):
        """Execute bot logic each game step."""
        await self.distribute_workers()
        await self.send_proxy_scv()
        await self.train_scvs()
        await self.build_supply_depots()
        await self.build_proxy_barracks()
        await self.build_home_barracks()
        await self.train_marines()
        await self.attack()

    async def send_proxy_scv(self):
        """Send an SCV to build proxy barracks near enemy."""
        # Only send once at the start
        if self.proxy_scv is not None:
            return

        # Send SCV early (iteration ~10-20)
        if self.time < 20 and self.supply_workers >= 13:
            # Select an SCV that's gathering
            workers = self.workers.gathering
            if workers:
                self.proxy_scv = workers.random
                self.proxy_scv.move(self.proxy_location)
                print(f"[ProxyBot] Sending proxy SCV to {self.proxy_location}")

    async def train_scvs(self):
        """Train minimal SCVs - we're all-in on military."""
        if self.supply_workers >= MAX_WORKERS:
            return

        for cc in self.townhalls.idle:
            if self.can_afford(UnitTypeId.SCV):
                cc.train(UnitTypeId.SCV)

    async def build_supply_depots(self):
        """Build supply depots at home base."""
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

    async def build_proxy_barracks(self):
        """Build barracks near enemy base using proxy SCV."""
        # Count proxy barracks (those far from our base)
        proxy_barracks = self.structures(UnitTypeId.BARRACKS).filter(
            lambda b: b.distance_to(self.start_location) > 50
        )

        if len(proxy_barracks) + self.already_pending(UnitTypeId.BARRACKS) >= PROXY_BARRACKS_COUNT:
            self.proxy_established = True
            return

        # Need supply depot first
        if not self.structures(UnitTypeId.SUPPLYDEPOT) and not self.already_pending(
            UnitTypeId.SUPPLYDEPOT
        ):
            return

        # Use proxy SCV if available and near proxy location
        if (
            self.proxy_scv
            and self.proxy_scv.is_alive
            and self.proxy_scv.distance_to(self.proxy_location) < 10
            and self.can_afford(UnitTypeId.BARRACKS)
        ):
            # Build near proxy location
            location = await self.find_placement(
                UnitTypeId.BARRACKS,
                near=self.proxy_location,
                max_distance=15,
                placement_step=2,
            )
            if location:
                self.proxy_scv.build(UnitTypeId.BARRACKS, location)
                print(f"[ProxyBot] Building proxy barracks at {location}")
                # After building, send SCV back or use for next barracks
                self.proxy_scv = None  # Free to use other workers for next barracks

    async def build_home_barracks(self):
        """Build one barracks at home for backup production."""
        # Count home barracks (close to our base)
        home_barracks = self.structures(UnitTypeId.BARRACKS).filter(
            lambda b: b.distance_to(self.start_location) < 50
        )

        if len(home_barracks) >= HOME_BARRACKS_COUNT:
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

    async def train_marines(self):
        """Train marines from all idle barracks."""
        for barracks in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.can_afford(UnitTypeId.MARINE):
                barracks.train(UnitTypeId.MARINE)

    async def attack(self):
        """
        Aggressive early attack strategy.

        - Attack immediately when we have ANY marines from proxy
        - No retreat, constant pressure
        - Pull workers for all-in once marines are at enemy base
        """
        marines = self.units(UnitTypeId.MARINE)
        enemy_start = self.enemy_start_locations[0]

        # Attack with first marine - no waiting!
        if len(marines) >= ATTACK_MARINE_THRESHOLD and not self.attack_started:
            self.attack_started = True
            print(f"[ProxyBot] Proxy rush started with {len(marines)} marines!")

        if marines:
            # Send all marines to enemy base immediately
            for marine in marines:
                if marine.is_idle or marine.distance_to(enemy_start) > 5:
                    marine.attack(enemy_start)

            # All-in: Once we have 6+ marines at enemy, send all workers too
            marines_at_enemy = marines.filter(
                lambda m: m.distance_to(enemy_start) < 30
            )
            if len(marines_at_enemy) >= 6:
                for worker in self.workers:
                    if worker.is_gathering:
                        worker.attack(enemy_start)
