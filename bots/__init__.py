"""StarCraft II bot implementations."""

from .idle_bot import IdleBot
from .rush_bot import RushBot
from .defense_bot import DefenseBot
from .economy_bot import EconomyBot
from .proxy_bot import ProxyBot
from .stim_bot import StimBot
from .tank_bot import TankBot
from .bioball_bot import BioBallBot
from .mech_bot import MechBot
from .marine_medivac_bot import MarineMedivacBot

__all__ = [
    "IdleBot",
    "RushBot",
    "DefenseBot",
    "EconomyBot",
    "ProxyBot",
    "StimBot",
    "TankBot",
    "BioBallBot",
    "MechBot",
    "MarineMedivacBot",
]
