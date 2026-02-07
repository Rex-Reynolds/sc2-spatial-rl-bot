"""Minimal idle bot that only trains SCVs and mines."""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId


class IdleBot(BotAI):
    """
    A do-nothing opponent that serves as a punching bag.

    Strategy:
    - Train SCVs continuously
    - Mine resources automatically
    - No structures, no army, no supply depots
    - Will supply-block itself quickly
    """

    async def on_step(self, iteration: int):
        """Execute bot logic each game step."""
        # Train SCVs from idle command centers
        for cc in self.townhalls.idle:
            if self.can_afford(UnitTypeId.SCV) and self.supply_workers < 15:
                cc.train(UnitTypeId.SCV)
