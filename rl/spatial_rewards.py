"""
Advanced Reward Shaping for Spatial RL Bot

Includes:
- Milestone rewards (economy, tech, expansion)
- Continuous rewards (army value, resource income)
- Spatial rewards (building placement, positioning)
- Combat rewards (kills, efficiency)
- Penalties (supply blocks, idle production)
"""

import numpy as np
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from typing import Set


class SpatialRewardCalculator:
    """Calculate rewards with spatial awareness."""

    def __init__(self):
        self.milestones_achieved: Set[str] = set()
        self.prev_enemy_units = 0
        self.prev_own_units = 0
        self.prev_army_value = 0
        self.prev_minerals_collected = 0
        self.prev_gas_collected = 0

    def calculate_reward(self, bot: BotAI) -> float:
        """
        Calculate comprehensive reward for current game state.

        Returns:
            Total reward for this step
        """
        reward = 0.0

        # === 1. MILESTONE REWARDS (one-time bonuses) ===
        reward += self._milestone_rewards(bot)

        # === 2. CONTINUOUS REWARDS (every step) ===
        reward += self._continuous_rewards(bot)

        # === 3. SPATIAL REWARDS (positioning & placement) ===
        reward += self._spatial_rewards(bot)

        # === 4. COMBAT REWARDS (kills, efficiency) ===
        reward += self._combat_rewards(bot)

        # === 5. PENALTIES (bad behaviors) ===
        reward += self._penalties(bot)

        return reward

    def _milestone_rewards(self, bot: BotAI) -> float:
        """One-time rewards for achieving milestones."""
        reward = 0.0

        # Economy milestones
        if bot.time >= 120 and bot.supply_workers >= 20:
            if "eco_2min_20workers" not in self.milestones_achieved:
                reward += 1.0
                self.milestones_achieved.add("eco_2min_20workers")

        if bot.time >= 300 and bot.supply_workers >= 40:
            if "eco_5min_40workers" not in self.milestones_achieved:
                reward += 2.0
                self.milestones_achieved.add("eco_5min_40workers")

        # Expansion milestones
        if bot.townhalls.amount >= 2:
            if "expand_natural" not in self.milestones_achieved:
                reward += 2.0
                self.milestones_achieved.add("expand_natural")

        if bot.townhalls.amount >= 3:
            if "expand_third" not in self.milestones_achieved:
                reward += 3.0
                self.milestones_achieved.add("expand_third")

        # Tech milestones
        if bot.structures(UnitTypeId.BARRACKS).ready:
            if "tech_barracks" not in self.milestones_achieved:
                reward += 0.5
                self.milestones_achieved.add("tech_barracks")

        if bot.structures(UnitTypeId.FACTORY).ready:
            if "tech_factory" not in self.milestones_achieved:
                reward += 1.0
                self.milestones_achieved.add("tech_factory")

        if bot.structures(UnitTypeId.STARPORT).ready:
            if "tech_starport" not in self.milestones_achieved:
                reward += 1.5
                self.milestones_achieved.add("tech_starport")

        # Upgrade milestones
        if UpgradeId.STIMPACK in bot.state.upgrades:
            if "upgrade_stim" not in self.milestones_achieved:
                reward += 2.0
                self.milestones_achieved.add("upgrade_stim")

        if UpgradeId.SHIELDWALL in bot.state.upgrades:
            if "upgrade_shields" not in self.milestones_achieved:
                reward += 1.0
                self.milestones_achieved.add("upgrade_shields")

        # Production milestones
        if bot.structures(UnitTypeId.BARRACKS).amount >= 2:
            if "production_2rax" not in self.milestones_achieved:
                reward += 0.5
                self.milestones_achieved.add("production_2rax")

        if bot.structures(UnitTypeId.BARRACKS).amount >= 5:
            if "production_5rax" not in self.milestones_achieved:
                reward += 1.0
                self.milestones_achieved.add("production_5rax")

        # Army milestones
        army_supply = bot.supply_army
        if army_supply >= 50:
            if "army_50supply" not in self.milestones_achieved:
                reward += 1.0
                self.milestones_achieved.add("army_50supply")

        if army_supply >= 100:
            if "army_100supply" not in self.milestones_achieved:
                reward += 2.0
                self.milestones_achieved.add("army_100supply")

        return reward

    def _continuous_rewards(self, bot: BotAI) -> float:
        """Continuous rewards for ongoing good behaviors."""
        reward = 0.0

        # Army value reward (encourage building units)
        army_value = (
            bot.units(UnitTypeId.MARINE).amount * 1 +
            bot.units(UnitTypeId.MARAUDER).amount * 2 +
            bot.units(UnitTypeId.SIEGETANK).amount * 3 +
            bot.units(UnitTypeId.HELLION).amount * 1.5 +
            bot.units(UnitTypeId.MEDIVAC).amount * 2 +
            bot.units(UnitTypeId.VIKINGFIGHTER).amount * 2
        )

        army_value_increase = army_value - self.prev_army_value
        reward += army_value_increase * 0.002
        self.prev_army_value = army_value

        # Resource collection reward
        minerals_collected = bot.state.score.collected_minerals
        gas_collected = bot.state.score.collected_vespene

        minerals_increase = minerals_collected - self.prev_minerals_collected
        gas_increase = gas_collected - self.prev_gas_collected

        reward += (minerals_increase + gas_increase * 1.5) * 0.00001
        self.prev_minerals_collected = minerals_collected
        self.prev_gas_collected = gas_collected

        # Worker saturation reward (encourage good economy)
        optimal_workers_per_base = 16
        total_optimal = bot.townhalls.amount * optimal_workers_per_base
        worker_efficiency = min(bot.supply_workers / max(total_optimal, 1), 1.0)
        reward += worker_efficiency * 0.01

        return reward

    def _spatial_rewards(self, bot: BotAI) -> float:
        """Rewards for good spatial decisions (positioning, placement)."""
        reward = 0.0

        # Building placement reward (close together = efficient)
        if bot.structures.amount >= 5:
            structures = bot.structures.exclude_type([UnitTypeId.SUPPLYDEPOT])
            if structures.amount > 0:
                center = structures.center
                avg_distance = np.mean([s.distance_to(center) for s in structures])

                # Reward compact base (avg distance < 20)
                if avg_distance < 20:
                    reward += 0.05

        # Supply depot placement reward (near base)
        depots = bot.structures(UnitTypeId.SUPPLYDEPOT)
        if depots and bot.townhalls:
            nearest_cc = bot.townhalls.first
            depot_distances = [d.distance_to(nearest_cc) for d in depots]
            avg_depot_distance = np.mean(depot_distances)

            # Reward depots close to base (distance < 15)
            if avg_depot_distance < 15:
                reward += 0.02

        # Army positioning reward (near enemy = aggressive, but not suicidal)
        army = bot.units.exclude_type([UnitTypeId.SCV])
        if army and bot.enemy_start_locations:
            enemy_pos = bot.enemy_start_locations[0]
            army_center = army.center

            distance_to_enemy = army_center.distance_to(enemy_pos)

            # Reward being in middle of map (not at home, not dying at enemy base)
            map_center = bot.game_info.map_center
            distance_to_map_center = army_center.distance_to(map_center)

            if distance_to_map_center < 20:  # Near center = good positioning
                reward += 0.05

        return reward

    def _combat_rewards(self, bot: BotAI) -> float:
        """Rewards for combat efficiency."""
        reward = 0.0

        # Kill rewards
        enemy_units_now = len(bot.enemy_units) + len(bot.enemy_structures)
        own_units_now = len(bot.units) + len(bot.structures)

        units_killed = self.prev_enemy_units - enemy_units_now
        units_lost = self.prev_own_units - own_units_now

        # Reward kills, penalize losses (but kills worth more)
        reward += units_killed * 0.3
        reward -= units_lost * 0.15

        self.prev_enemy_units = enemy_units_now
        self.prev_own_units = own_units_now

        # Combat efficiency (damage dealt vs taken)
        damage_dealt = bot.state.score.total_damage_dealt_life
        damage_taken = bot.state.score.total_damage_taken_life

        if damage_taken > 0:
            efficiency_ratio = damage_dealt / damage_taken
            reward += efficiency_ratio * 0.01

        # Reward attacking (action, not just having units)
        if bot.enemy_units:
            enemy_in_vision = bot.enemy_units.closer_than(50, bot.start_location)
            if enemy_in_vision:
                reward += 0.05  # Reward for engaging nearby enemies

        return reward

    def _penalties(self, bot: BotAI) -> float:
        """Penalties for bad behaviors."""
        penalty = 0.0

        # Supply block penalty
        if bot.supply_left < 3 and bot.supply_used >= 20 and bot.supply_cap < 200:
            penalty -= 0.2

        # Idle production penalty
        idle_barracks = bot.structures(UnitTypeId.BARRACKS).ready.idle.amount
        idle_factories = bot.structures(UnitTypeId.FACTORY).ready.idle.amount
        idle_starports = bot.structures(UnitTypeId.STARPORT).ready.idle.amount

        total_idle_production = idle_barracks + idle_factories + idle_starports

        # Penalize idle production if have resources
        if bot.minerals > 200 and bot.supply_left > 0:
            penalty -= total_idle_production * 0.05

        # Idle worker penalty
        idle_workers = bot.workers.idle.amount
        if idle_workers > 0 and bot.minerals < 500:
            penalty -= idle_workers * 0.01

        # Floating resources penalty (late game)
        if bot.time > 300:  # After 5 minutes
            if bot.minerals > 1000 and bot.vespene > 500:
                penalty -= 0.1  # Not spending resources

        # Over-saturation penalty (too many workers)
        optimal_workers = bot.townhalls.amount * 22  # 16 + 6 on gas
        if bot.supply_workers > optimal_workers + 10:
            penalty -= 0.05  # Too many workers

        return penalty

    def reset(self):
        """Reset tracker for new episode."""
        self.milestones_achieved = set()
        self.prev_enemy_units = 0
        self.prev_own_units = 0
        self.prev_army_value = 0
        self.prev_minerals_collected = 0
        self.prev_gas_collected = 0
