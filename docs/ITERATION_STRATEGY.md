# Bot Iteration Strategy

## Overview

This document outlines approaches for improving bot performance, from simple manual tuning to advanced machine learning.

## Approach 1: Manual Iteration (Current - Week 1)

### What it is
Tune constants in `rush_bot.py` and measure win rates via tournaments.

### Quick experiments to try:
```bash
# In rush_bot.py, try different combinations:
MAX_WORKERS = [12, 14, 16, 18, 20]
BARRACKS_COUNT = [1, 2, 3, 4]
ATTACK_MARINE_THRESHOLD = [6, 8, 10, 12, 15]

# Test each: python scripts/run_tournament.py -n 20
```

### Improvements to add:
1. **Build order stages**: Add time-based decisions
2. **Better targeting**: Attack workers first, then army
3. **Basic micro**: Retreat damaged marines, kite with range advantage
4. **Scouting**: Send SCV to scout enemy at game start

### Pros/Cons
- ✅ Fast to test, easy to understand
- ❌ Limited by human creativity
- ❌ Doesn't scale to complex strategies

---

## Approach 2: Parameter Optimization (Week 2-3)

### What it is
Automatically search for optimal parameter combinations.

### Methods:

#### Grid Search (simplest)
```python
# scripts/optimize_params.py
from itertools import product

params = {
    'MAX_WORKERS': [12, 16, 20],
    'BARRACKS_COUNT': [2, 3, 4],
    'ATTACK_THRESHOLD': [6, 10, 14],
}

best_score = 0
best_params = None

for combo in product(*params.values()):
    # Run tournament with these params
    # Track win rate
    # Save best
```

#### Genetic Algorithm (more efficient)
```python
# Use DEAP library
# Evolve parameters over generations
# Crossover + mutation + selection
# Converges faster than grid search
```

#### Bayesian Optimization (most sample-efficient)
```python
# Use scikit-optimize
# Builds surrogate model of win_rate(params)
# Intelligently explores parameter space
# Best for expensive evaluations (slow games)
```

### Pros/Cons
- ✅ Automated, finds good configs
- ✅ Can optimize 5-10 parameters simultaneously
- ❌ Only tunes existing logic (can't invent new strategies)
- ❌ Computationally expensive (100s of games)

---

## Approach 3: Replay-based Imitation Learning (Advanced)

### What it is
Learn bot behavior by mimicking human/pro players from replays.

### Architecture:

#### Data Collection
```python
import sc2reader

# Parse replays
for replay in replay_files:
    for frame in replay.frames:
        state = extract_state(frame)  # minerals, units, etc.
        action = extract_action(frame)  # what player did
        dataset.append((state, action))
```

#### State Representation (vectorization)
```python
# Option A: Flat vector (~100 features)
state = [
    # Economy (4)
    minerals, gas, supply_used, supply_left,

    # Own units (15+)
    scv_count, marine_count, marauder_count, ...,

    # Own buildings (10+)
    cc_count, barracks_count, factory_count, ...,

    # Enemy intel (20+ if visible)
    enemy_worker_count, enemy_army_count, ...,

    # Spatial (optional - more complex)
    # 84x84 feature maps for minimap-like representation
]

# Option B: Graph representation
# Nodes = units/buildings, edges = relationships
# Better for complex spatial reasoning
```

#### Action Space
```python
# Discrete actions
actions = [
    'train_scv',
    'build_supply_depot',
    'build_barracks',
    'train_marine',
    'attack_enemy_base',
    'attack_enemy_army',
    'defend_base',
    'expand',
    'no_op',
]

# Or continuous: (action_type, target_x, target_y, unit_id)
```

#### Training
```python
# Supervised learning
model = BehaviorCloning(
    state_dim=100,
    action_dim=9,
    hidden_layers=[256, 128],
)

# Train on replay dataset
for epoch in range(100):
    for state, action in dataloader:
        pred_action = model(state)
        loss = cross_entropy(pred_action, action)
        loss.backward()
        optimizer.step()
```

### Challenges:
1. **Data quality**: Need many replays from good players
2. **Action ambiguity**: Hard to extract exact intent from replay frames
3. **Distribution shift**: Replays are vs humans, bot faces different opponents
4. **Generalization**: Model might overfit to specific maps/matchups
5. **Timing**: Frame-by-frame actions don't capture strategic timing

### Pros/Cons
- ✅ Can learn sophisticated strategies
- ✅ Faster than RL (supervised learning)
- ✅ Good for bootstrapping RL agents
- ❌ Requires lots of quality replay data
- ❌ Limited by training data (can't exceed human play)
- ❌ Distribution shift issues

---

## Approach 4: Reinforcement Learning (Your Phase 2 - Long-term)

### What it is
Bot learns through self-play, trial and error, optimizing for wins.

### Why it's better than imitation learning:
1. **No replay data needed**: Learns from scratch
2. **Discovers novel strategies**: Not limited by human play
3. **Adapts to opponents**: Learns what works against specific bots
4. **Continuous improvement**: Gets better over time

### Your planned architecture (from plan):
```python
# Observation space
obs = [
    minerals, gas, supply_used, supply_cap,
    scv_count, marine_count, barracks_count,
    enemy_visible_units, game_time,
    # 11 features total (start simple)
]

# Action space (discrete)
actions = [
    0: train_scv,
    1: build_supply_depot,
    2: build_barracks,
    3: train_marine,
    4: attack,
    5: defend,
    6: no_op,
]

# Reward shaping
reward = (
    +10 if win else -10 if loss else 0
    + 0.1 * enemy_units_killed
    - 0.05 * own_units_lost
    + 0.01 * minerals_mined
)
```

### Training pipeline:
```bash
# 1. Train vs IdleBot (1000 episodes)
# 2. Train vs RushBot (2000 episodes)
# 3. Self-play (5000+ episodes)
# 4. Train vs AlphaStar-inspired bots
```

### Pros/Cons
- ✅ Discovers novel strategies
- ✅ No replay data needed
- ✅ Adapts to opponents
- ✅ Can surpass human-level play (eventually)
- ❌ Very slow (10k-1M games to train)
- ❌ Reward shaping is hard
- ❌ Requires lots of compute

---

## Recommended Path Forward

### Phase 1: Manual Iteration (Now)
- [x] Basic RushBot working
- [ ] Add 2-3 more scripted bots (DefenseBot, EconomyBot, ProxyBot)
- [ ] Tune parameters via tournaments
- [ ] Add basic micro (retreat, focus fire)

### Phase 1.5: Parameter Optimization (1-2 weeks)
- [ ] Implement genetic algorithm for parameter tuning
- [ ] Create bot ladder (Easy → Medium → Hard opponents)
- [ ] Benchmark: RushBot should beat IdleBot 100% of the time

### Phase 2: Reinforcement Learning (Main goal)
- [ ] Implement Gymnasium wrapper
- [ ] Train PPO agent vs IdleBot
- [ ] Train vs scripted bots
- [ ] Self-play training
- [ ] Benchmark vs built-in AI

### Phase 3 (Optional): Hybrid Approach
- [ ] Use imitation learning to bootstrap RL
- [ ] Pretrain on replays, fine-tune with RL
- [ ] Best of both worlds

---

## Replay-based Learning: Implementation Details

If you want to explore replay learning, here's how:

### Tools:
- `sc2reader`: Parse replay files
- `pysc2`: DeepMind's SC2 RL environment (has replay features)
- `burnysc2`: Can load and analyze replays

### Example workflow:
```python
# 1. Collect replays (download from spawningtool.com or record your own)
# 2. Parse replays
import sc2reader

replay = sc2reader.load_replay('replay.SC2Replay')
for event in replay.events:
    if event.name == 'UnitBornEvent':
        # Track what was built when
        pass
    if event.name == 'CommandEvent':
        # Track commands issued
        pass

# 3. Create state-action dataset
# 4. Train behavior cloning model
# 5. Deploy model in bot
```

### Sample sizes needed:
- **Basic imitation**: 100-500 replays
- **Strong imitation**: 1000-5000 replays
- **Human-level**: 10,000+ replays (AlphaStar used 971k)

---

## My Recommendation

**For your goals (RL foundation), I recommend:**

1. **Short-term (this week)**:
   - Create 2-3 more scripted bots with different strategies
   - This gives RL agent varied opponents to train against

2. **Medium-term (2-3 weeks)**:
   - Implement the RL wrapper (your Phase 2)
   - Start with simple observation/action spaces
   - Train vs scripted bots

3. **Long-term (optional)**:
   - Try replay-based imitation learning to bootstrap RL
   - Use imitation learning to get a "warm start" for RL training
   - Fine-tune with RL for superhuman performance

**Skip pure replay learning** unless you specifically want to mimic human play. For discovering novel strategies and building a strong bot, RL is superior (though slower).

---

## Next Steps

Want me to implement:
1. **A genetic algorithm optimizer** for parameter tuning?
2. **2-3 new scripted bots** with different strategies?
3. **The RL environment wrapper** (Phase 2 kickoff)?
4. **A replay parser** to analyze your own bot's games?

Let me know what direction you want to explore!
