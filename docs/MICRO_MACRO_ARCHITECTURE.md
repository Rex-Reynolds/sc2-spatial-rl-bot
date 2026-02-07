# Micro vs Macro Architecture

## The Problem You Identified

**Current issue**: Bots make macro decisions (build orders, economy) but have poor micro (unit control, positioning, combat).

**Your observation**: Units sit idle during production instead of:
- Scouting
- Positioning defensively
- Practicing splits/formations
- Preparing for engagement

---

## Solution: Separate Micro and Macro

### Architecture Option 1: Micro Manager Class

```python
class MicroManager:
    """Handles all unit-level decisions."""

    def __init__(self, bot):
        self.bot = bot

    async def manage_army(self, army, strategy="defensive"):
        """
        Handle army micro based on strategy.

        Strategies:
        - defensive: Keep units near base
        - aggressive: Push toward enemy
        - harass: Multi-pronged attacks
        - retreat: Pull back injured units
        """
        if strategy == "defensive":
            await self.defensive_positioning(army)
        elif strategy == "aggressive":
            await self.attack_move(army)

    async def defensive_positioning(self, army):
        """Position army defensively."""
        center = self.bot.start_location.towards(
            self.bot.game_info.map_center, 15
        )
        for unit in army:
            if unit.distance_to(center) > 10:
                unit.move(center)

    async def stutter_step(self, unit, target):
        """Kite: attack, move back, attack."""
        if unit.weapon_cooldown == 0:
            unit.attack(target)
        elif unit.distance_to(target) < unit.ground_range:
            # Move back while weapon is cooling
            unit.move(unit.position.towards(target, -2))

    async def focus_fire(self, army, priority="low_hp"):
        """Focus fire on priority targets."""
        if not self.bot.enemy_units:
            return

        if priority == "low_hp":
            target = min(self.bot.enemy_units, key=lambda u: u.health)
        elif priority == "high_value":
            # Prioritize siege tanks, medivacs, etc.
            target = self.get_high_value_target()

        for unit in army:
            unit.attack(target)

    async def split_vs_splash(self, army):
        """Spread units to avoid splash damage."""
        center = army.center
        for unit in army:
            # Move away from center to spread
            if unit.distance_to(center) < 2:
                spread_point = unit.position.towards(center, -3)
                unit.move(spread_point)


class BioBallBot(BotAI):
    def __init__(self):
        super().__init__()
        self.micro = MicroManager(self)  # Micro manager!

    async def attack(self):
        army = self.units(UnitTypeId.MARINE) | self.units(UnitTypeId.MARAUDER)

        if self.attack_started:
            # Use micro manager for combat
            await self.micro.manage_army(army, strategy="aggressive")
            await self.micro.focus_fire(army, priority="high_value")
        else:
            # Use micro manager for positioning
            await self.micro.manage_army(army, strategy="defensive")
```

**Pros**:
- ✅ Clean separation
- ✅ Reusable across bots
- ✅ Easy to test micro independently

**Cons**:
- ❌ Still synchronous with macro
- ❌ Micro decisions at same rate as macro

---

### Architecture Option 2: Async Micro Agent

```python
class MicroAgent:
    """Separate async agent for micro decisions."""

    def __init__(self, bot):
        self.bot = bot
        self.tasks = []

    async def start(self):
        """Run micro loop independently."""
        while True:
            await self.micro_step()
            await asyncio.sleep(0.1)  # Micro at 10 FPS

    async def micro_step(self):
        """High-frequency micro decisions."""
        # This runs EVERY 0.1s, independent of macro
        army = self.bot.units(UnitTypeId.MARINE)

        for marine in army:
            # Check if in combat
            if self.bot.enemy_units.closer_than(15, marine):
                await self.combat_micro(marine)
            else:
                await self.idle_micro(marine)

    async def combat_micro(self, unit):
        """Micro during combat."""
        enemies = self.bot.enemy_units.closer_than(10, unit)
        if not enemies:
            return

        target = enemies.closest_to(unit)

        # Kiting logic
        if unit.weapon_cooldown > 0:
            # Retreat while cooling down
            retreat_point = unit.position.towards(target.position, -2)
            unit.move(retreat_point)
        else:
            # Attack when ready
            unit.attack(target)

        # Stim if needed
        if (
            unit.health_percentage > 0.5
            and not unit.has_buff(AbilityId.EFFECT_STIM)
            and len(enemies) >= 3
        ):
            unit(AbilityId.EFFECT_STIM)

    async def idle_micro(self, unit):
        """Micro when not in combat."""
        # Defensive positioning
        rally_point = self.bot.start_location.towards(
            self.bot.game_info.map_center, 15
        )
        if unit.distance_to(rally_point) > 10:
            unit.move(rally_point)


class BioBallBot(BotAI):
    def __init__(self):
        super().__init__()
        self.micro_agent = MicroAgent(self)

    async def on_start(self):
        # Start micro agent in background
        asyncio.create_task(self.micro_agent.start())

    async def on_step(self, iteration: int):
        # Only handle macro here
        await self.train_scvs()
        await self.build_structures()
        await self.train_army()
        # Micro agent handles all unit control!
```

**Pros**:
- ✅ True separation
- ✅ High-frequency micro (10+ FPS)
- ✅ Macro runs at lower frequency

**Cons**:
- ❌ More complex
- ❌ Potential race conditions
- ❌ Harder to debug

---

### Architecture Option 3: Behavior Trees

```python
from enum import Enum

class BehaviorState(Enum):
    IDLE = 1
    DEFENDING = 2
    ATTACKING = 3
    RETREATING = 4
    HARASSING = 5


class UnitBehavior:
    """Behavior tree for individual units."""

    def __init__(self, unit):
        self.unit = unit
        self.state = BehaviorState.IDLE

    async def update(self, bot):
        """Update behavior based on situation."""
        # Evaluate conditions
        enemies_nearby = bot.enemy_units.closer_than(15, self.unit)
        low_health = self.unit.health_percentage < 0.3
        in_combat = len(enemies_nearby) > 0

        # State machine
        if low_health and in_combat:
            self.state = BehaviorState.RETREATING
            await self.retreat(bot)
        elif in_combat:
            self.state = BehaviorState.ATTACKING
            await self.attack_nearest(enemies_nearby)
        elif bot.attack_started:
            self.state = BehaviorState.ATTACKING
            await self.advance(bot)
        else:
            self.state = BehaviorState.DEFENDING
            await self.defend(bot)

    async def retreat(self, bot):
        """Retreat toward base."""
        self.unit.move(bot.start_location)

    async def attack_nearest(self, enemies):
        """Attack closest enemy."""
        if enemies:
            self.unit.attack(enemies.closest_to(self.unit))

    async def advance(self, bot):
        """Move toward enemy."""
        self.unit.attack(bot.enemy_start_locations[0])

    async def defend(self, bot):
        """Defensive positioning."""
        rally = bot.start_location.towards(bot.game_info.map_center, 15)
        if self.unit.distance_to(rally) > 10:
            self.unit.move(rally)


class BioBallBot(BotAI):
    def __init__(self):
        super().__init__()
        self.unit_behaviors = {}

    async def on_step(self, iteration: int):
        # Macro decisions
        await self.economy()
        await self.production()

        # Micro decisions
        await self.update_unit_behaviors()

    async def update_unit_behaviors(self):
        """Update behavior for each unit."""
        army = self.units(UnitTypeId.MARINE) | self.units(UnitTypeId.MARAUDER)

        for unit in army:
            if unit.tag not in self.unit_behaviors:
                self.unit_behaviors[unit.tag] = UnitBehavior(unit)

            await self.unit_behaviors[unit.tag].update(self)

        # Clean up dead units
        alive_tags = {u.tag for u in army}
        self.unit_behaviors = {
            tag: behavior
            for tag, behavior in self.unit_behaviors.items()
            if tag in alive_tags
        }
```

**Pros**:
- ✅ Individual unit intelligence
- ✅ Scalable to complex behaviors
- ✅ Easy to add new behaviors

**Cons**:
- ❌ Overhead per unit
- ❌ More code to maintain

---

## Recommended Approach

**For your bots, I recommend Option 1: Micro Manager Class**

Why:
1. **Simple to implement** - can add now
2. **Reusable** - one manager for all bots
3. **Testable** - easy to verify micro works
4. **Good enough** - covers 80% of use cases

### Implementation Plan:

1. **Create `utils/micro_manager.py`**
   - Defensive positioning
   - Attack move
   - Focus fire
   - Stutter step (kiting)
   - Retreat logic

2. **Add to existing bots**:
   ```python
   from utils.micro_manager import MicroManager

   class BioBallBot(BotAI):
       def __init__(self):
           super().__init__()
           self.micro = MicroManager(self)

       async def attack(self):
           army = self.get_army()
           if self.attack_started:
               await self.micro.attack_move(army, target)
           else:
               await self.micro.defensive_position(army)
   ```

3. **Test improvement**:
   - Run BioBall vs Mech again
   - Watch for better positioning
   - Measure win rate improvement

---

## Next Steps

Want me to:
1. **Implement MicroManager class** with basic behaviors?
2. **Add it to BioBallBot/MechBot** and test?
3. **Or skip micro for now** and focus on fixing gas first?

The gas issue is critical (MechBot can't function without it), so let's fix that first, then circle back to micro!
