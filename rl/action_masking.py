"""
Action Masking for Spatial RL Bot

Prevents invalid actions (e.g., training marines without barracks).
Improves learning efficiency by only allowing valid actions.
"""

import numpy as np
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId


def get_available_actions(bot: BotAI) -> np.ndarray:
    """
    Get mask of available actions (1 = available, 0 = not available).

    Returns:
        Boolean array of shape (50,) indicating which actions are valid.
    """
    mask = np.zeros(50, dtype=np.float32)

    # Action 0: no_op - always available
    mask[0] = 1.0

    # Action 1: select_idle_workers - available if have idle workers
    mask[1] = 1.0 if bot.workers.idle else 0.0

    # Action 2: select_army - available if have army
    mask[2] = 1.0 if bot.units.exclude_type([UnitTypeId.SCV]).amount > 0 else 0.0

    # Action 3: select_marines - available if have marines
    mask[3] = 1.0 if bot.units(UnitTypeId.MARINE).amount > 0 else 0.0

    # Action 4: select_tanks - available if have tanks
    mask[4] = 1.0 if bot.units(UnitTypeId.SIEGETANK).amount > 0 else 0.0

    # Action 5: train_scv - available if have townhall, can afford, have supply
    mask[5] = 1.0 if (bot.townhalls.idle and
                      bot.can_afford(UnitTypeId.SCV) and
                      bot.supply_left > 0) else 0.0

    # Action 6: build_supply_depot - available if can afford, have worker, not at cap
    mask[6] = 1.0 if (bot.can_afford(UnitTypeId.SUPPLYDEPOT) and
                      bot.workers and
                      bot.supply_cap < 200) else 0.0

    # Action 7: build_barracks - available if can afford, have worker, have depot
    mask[7] = 1.0 if (bot.can_afford(UnitTypeId.BARRACKS) and
                      bot.workers and
                      bot.structures(UnitTypeId.SUPPLYDEPOT).ready) else 0.0

    # Action 8: build_refinery - available if can afford, have worker, have free geyser
    has_free_geyser = False
    for townhall in bot.townhalls.ready:
        vespenes = bot.vespene_geyser.closer_than(10, townhall)
        for vespene in vespenes:
            if not bot.structures(UnitTypeId.REFINERY).closer_than(1, vespene):
                has_free_geyser = True
                break
    mask[8] = 1.0 if (bot.can_afford(UnitTypeId.REFINERY) and
                      bot.workers and
                      has_free_geyser) else 0.0

    # Action 9: build_factory - available if can afford, have worker, have barracks
    mask[9] = 1.0 if (bot.can_afford(UnitTypeId.FACTORY) and
                      bot.workers and
                      bot.structures(UnitTypeId.BARRACKS).ready) else 0.0

    # Action 10: build_starport - available if can afford, have worker, have factory
    mask[10] = 1.0 if (bot.can_afford(UnitTypeId.STARPORT) and
                       bot.workers and
                       bot.structures(UnitTypeId.FACTORY).ready) else 0.0

    # Action 11: train_marine - available if have barracks, can afford, have supply
    mask[11] = 1.0 if (bot.structures(UnitTypeId.BARRACKS).ready.idle and
                       bot.can_afford(UnitTypeId.MARINE) and
                       bot.supply_left > 0) else 0.0

    # Action 12: train_marauder - available if have barracks w/ tech lab, can afford
    has_tech_lab_barracks = bot.structures(UnitTypeId.BARRACKSTECHLAB).ready.amount > 0
    mask[12] = 1.0 if (has_tech_lab_barracks and
                       bot.can_afford(UnitTypeId.MARAUDER) and
                       bot.supply_left > 0) else 0.0

    # Action 13: train_tank - available if have factory w/ tech lab, can afford
    has_tech_lab_factory = bot.structures(UnitTypeId.FACTORYTECHLAB).ready.amount > 0
    mask[13] = 1.0 if (has_tech_lab_factory and
                       bot.can_afford(UnitTypeId.SIEGETANK) and
                       bot.supply_left > 0) else 0.0

    # Action 14: train_hellion - available if have factory, can afford
    mask[14] = 1.0 if (bot.structures(UnitTypeId.FACTORY).ready.idle and
                       bot.can_afford(UnitTypeId.HELLION) and
                       bot.supply_left > 0) else 0.0

    # Action 15: train_medivac - available if have starport, can afford
    mask[15] = 1.0 if (bot.structures(UnitTypeId.STARPORT).ready.idle and
                       bot.can_afford(UnitTypeId.MEDIVAC) and
                       bot.supply_left > 0) else 0.0

    # Action 16: build_tech_lab_barracks - available if have barracks without add-on
    mask[16] = 1.0 if (bot.structures(UnitTypeId.BARRACKS).ready.idle and
                       bot.can_afford(UnitTypeId.BARRACKSTECHLAB)) else 0.0

    # Action 17: build_reactor_barracks
    mask[17] = 1.0 if (bot.structures(UnitTypeId.BARRACKS).ready.idle and
                       bot.can_afford(UnitTypeId.BARRACKSREACTOR)) else 0.0

    # Action 18: research_stim - available if have tech lab, don't have stim, can afford
    mask[18] = 1.0 if (has_tech_lab_barracks and
                       UpgradeId.STIMPACK not in bot.state.upgrades and
                       bot.can_afford(UpgradeId.STIMPACK)) else 0.0

    # Action 19: research_combat_shields
    mask[19] = 1.0 if (has_tech_lab_barracks and
                       UpgradeId.SHIELDWALL not in bot.state.upgrades and
                       bot.can_afford(UpgradeId.SHIELDWALL)) else 0.0

    # Action 20: research_concussive
    mask[20] = 1.0 if (has_tech_lab_barracks and
                       UpgradeId.PUNISHERGRENADES not in bot.state.upgrades and
                       bot.can_afford(UpgradeId.PUNISHERGRENADES)) else 0.0

    # Action 21: upgrade_weapons - available if have engineering bay, can afford
    mask[21] = 1.0 if (bot.structures(UnitTypeId.ENGINEERINGBAY).ready and
                       bot.can_afford(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1)) else 0.0

    # Action 22: upgrade_armor
    mask[22] = 1.0 if (bot.structures(UnitTypeId.ENGINEERINGBAY).ready and
                       bot.can_afford(UpgradeId.TERRANINFANTRYARMORSLEVEL1)) else 0.0

    # Actions 23-29: Movement/attack commands - always available if have units
    has_army = bot.units.exclude_type([UnitTypeId.SCV]).amount > 0
    for i in range(23, 30):
        mask[i] = 1.0 if has_army else 0.0

    # Action 30: use_stim - available if have stim and marines/marauders
    mask[30] = 1.0 if (UpgradeId.STIMPACK in bot.state.upgrades and
                       bot.units({UnitTypeId.MARINE, UnitTypeId.MARAUDER}).amount > 0) else 0.0

    # Action 31: siege_tanks - available if have unsieged tanks
    mask[31] = 1.0 if bot.units(UnitTypeId.SIEGETANK).amount > 0 else 0.0

    # Action 32: unsiege_tanks - available if have sieged tanks
    mask[32] = 1.0 if bot.units(UnitTypeId.SIEGETANKSIEGED).amount > 0 else 0.0

    # Action 33-34: Medivac actions - available if have medivacs
    mask[33] = 1.0 if bot.units(UnitTypeId.MEDIVAC).amount > 0 else 0.0
    mask[34] = 1.0 if bot.units(UnitTypeId.MEDIVAC).amount > 0 else 0.0

    # Action 35: build_at_location - available if have worker
    mask[35] = 1.0 if bot.workers else 0.0

    # Action 36: expand - available if can afford CC
    mask[36] = 1.0 if bot.can_afford(UnitTypeId.COMMANDCENTER) and bot.workers else 0.0

    # Actions 37-40: Strategic commands - always available
    for i in range(37, 41):
        mask[i] = 1.0

    # Actions 41-49: Various commands - always available if conditions met
    mask[41] = 1.0 if bot.workers else 0.0  # scout
    mask[42] = 1.0 if has_army and bot.enemy_units else 0.0  # focus fire
    mask[43] = 1.0 if has_army else 0.0  # split army
    mask[44] = 1.0 if bot.structures(UnitTypeId.BARRACKS).ready else 0.0  # rally point
    mask[45] = 1.0  # cancel building (always available)
    mask[46] = 1.0 if bot.structures(UnitTypeId.BUNKER) else 0.0  # salvage bunker
    mask[47] = 1.0 if bot.structures else 0.0  # lift building
    mask[48] = 1.0  # land building
    mask[49] = 1.0 if bot.structures(UnitTypeId.ORBITALCOMMAND).ready else 0.0  # scan

    return mask


def apply_action_mask(logits: np.ndarray, mask: np.ndarray, mask_value: float = -1e9) -> np.ndarray:
    """
    Apply action mask to logits.

    Sets logits of invalid actions to very negative value so they won't be sampled.

    Args:
        logits: Action logits from policy (shape: [batch, num_actions])
        mask: Action mask (shape: [num_actions])
        mask_value: Value to set for masked actions

    Returns:
        Masked logits
    """
    # Expand mask if needed
    if logits.ndim > mask.ndim:
        mask = np.expand_dims(mask, 0)

    # Apply mask (0 → mask_value, 1 → keep original)
    masked_logits = np.where(mask > 0.5, logits, mask_value)

    return masked_logits
