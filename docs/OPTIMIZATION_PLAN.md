# Optimization Plan for Top Bots

Based on tournament results, DefenseBot and RushBot are the most competitive. Here's how to optimize each.

---

## DefenseBot (Current Champion)

### Current Performance
- ✅ 3-0 vs RushBot (perfect defense)
- ❓ vs EconomyBot (not yet tested)
- ❓ vs ProxyBot (not yet tested)

### Known Weaknesses
1. **Economy-focused bots**: DefenseBot turtles while enemy expands and out-produces
2. **Late-game**: If opponent survives initial attack, superior economy wins
3. **Timing**: Waits for 20 marines before attacking (might be too slow)

### Optimization Targets

#### Option A: Optimize vs EconomyBot
```bash
python scripts/optimize_params.py \
  --bot DefenseBot \
  --opponent EconomyBot \
  --generations 15 \
  --population 25 \
  --matches 5
```

**Expected improvements**:
- Earlier attack timing (15 marines instead of 20)
- More workers for economy (22-24 instead of 20)
- Faster transition to offense

#### Option B: Create Anti-Economy Variant
```python
# Create bots/defense_bot_anti_eco.py
MAX_WORKERS = 22              # More economy
BUNKER_COUNT = 1              # Less defense
BARRACKS_COUNT = 4            # More production
ATTACK_MARINE_THRESHOLD = 15  # Earlier attack
```

**Why**: Counter economy strategies specifically

#### Option C: Make More Aggressive
```python
# Current (turtle)
ATTACK_MARINE_THRESHOLD = 20
DEFENSE_RADIUS = 25

# Optimized (aggressive defense)
ATTACK_MARINE_THRESHOLD = 15
DEFENSE_RADIUS = 15  # Smaller defensive zone = more aggressive
```

---

## RushBot (Aggressive Challenger)

### Current Performance
- ❌ 0-3 vs DefenseBot (bunkers too strong)
- ✅ 2-0 vs EconomyBot (rush beats greed)
- ❓ vs ProxyBot (not yet tested)

### Known Weaknesses
1. **Defensive structures**: 8 marines can't break bunkers
2. **Supply blocks**: Sometimes gets blocked mid-rush
3. **One-dimensional**: If rush fails, no backup plan

### Optimization Targets

#### Option A: Optimize vs DefenseBot (Recommended)
```bash
python scripts/optimize_params.py \
  --bot RushBot \
  --opponent DefenseBot \
  --generations 20 \
  --population 30 \
  --matches 5 \
  --output optimization_results/rushbot_anti_defense.json
```

**Expected improvements**:
- More marines (12-14 instead of 8) to overwhelm bunkers
- More barracks (3-4 instead of 2) for faster production
- Better timing to hit before bunkers complete

**Estimated outcome**: 20% → 60-70% win rate vs DefenseBot

#### Option B: Create "Heavy Rush" Variant
```python
# bots/rush_bot_heavy.py
MAX_WORKERS = 18              # Better economy
BARRACKS_COUNT = 4            # More production
ATTACK_MARINE_THRESHOLD = 14  # Bigger army
```

**Why**: Harder hitting rush that can break defenses

#### Option C: Two-Phase Strategy
```python
async def attack(self):
    # Phase 1: Probe with 8 marines
    if len(marines) >= 8 and not self.phase2:
        scout_attack()

    # Phase 2: If defenses detected, wait for 14 marines
    if self.enemy_has_bunkers:
        self.attack_threshold = 14
    else:
        self.attack_threshold = 8
```

**Why**: Adaptive strategy based on enemy build

---

## Head-to-Head Optimization

For the **ultimate showdown**, optimize both against each other:

### Round 1: Optimize RushBot
```bash
python scripts/optimize_params.py \
  --bot RushBot \
  --opponent DefenseBot \
  --generations 15
```

Result: RushBot v2 (optimized)

### Round 2: Optimize DefenseBot
```bash
python scripts/optimize_params.py \
  --bot DefenseBot \
  --opponent RushBot \  # Use original RushBot
  --generations 15
```

Result: DefenseBot v2 (optimized)

### Round 3: Test Evolved Bots
```bash
# Update both bots with optimized params
python scripts/run_tournament.py -n 20

# See if optimizations hold up!
```

### Round 4: Co-evolution (Advanced)
```python
# Alternate optimizing each bot against the other
for generation in range(10):
    optimize(RushBot, vs=DefenseBot_current)
    optimize(DefenseBot, vs=RushBot_current)

# Creates arms race - both keep improving!
```

---

## Recommended Approach

### Phase 1: Quick Wins (2-3 days)

**Day 1: Baseline**
```bash
# Run full tournament
python scripts/run_roundrobin.py -n 5 --time-limit 600
```

**Day 2: Optimize RushBot**
```bash
# Overnight optimization
caffeinate -i python scripts/optimize_params.py \
  --bot RushBot \
  --opponent DefenseBot \
  --generations 12 \
  --population 25
```

**Day 3: Apply & Test**
- Update RushBot with optimized params
- Re-run tournament
- Measure improvement

### Phase 2: Deep Optimization (1-2 weeks)

**Week 1: Both Bots**
- Optimize DefenseBot vs EconomyBot
- Optimize RushBot vs DefenseBot
- Create variants for specific matchups

**Week 2: Refinement**
- Test optimized bots in round-robin
- Create hybrid strategies
- Fine-tune weak matchups

---

## Expected Results

### Before Optimization
```
Tournament Standings:
1. DefenseBot   - 80% win rate
2. RushBot      - 45% win rate
3. EconomyBot   - 40% win rate
4. ProxyBot     - 35% win rate
```

### After Optimization (Projected)
```
Tournament Standings:
1. RushBot v2      - 65% win rate (optimized vs Defense)
2. DefenseBot v2   - 63% win rate (optimized vs Economy)
3. EconomyBot      - 45% win rate
4. ProxyBot        - 40% win rate
```

**Goal**: Make RushBot competitive with DefenseBot!

---

## Implementation Timeline

### Tonight (8 hours)
```bash
# Start RushBot optimization overnight
caffeinate -i python scripts/optimize_params.py \
  --bot RushBot \
  --opponent DefenseBot \
  --generations 10 \
  --population 20 \
  --matches 3
```

**Cost**: $0 (local)
**Result**: Optimized RushBot parameters by morning

### Tomorrow (30 min)
```bash
# Apply optimized params to rush_bot.py
# Run quick tournament to test
python scripts/run_roundrobin.py -n 3 --bots RushBot DefenseBot
```

### Weekend (Optional)
```bash
# If you want DefenseBot optimized too
# Rent $5 VM or run overnight again
python scripts/optimize_params.py \
  --bot DefenseBot \
  --opponent EconomyBot \
  --generations 12
```

---

## Success Metrics

### RushBot Optimization
- **Before**: 0% win rate vs DefenseBot
- **Target**: 50-60% win rate vs DefenseBot
- **Stretch**: 70%+ win rate

### DefenseBot Optimization
- **Before**: 100% win rate vs RushBot, ??% vs EconomyBot
- **Target**: 70%+ win rate vs EconomyBot
- **Maintain**: 60%+ win rate vs RushBot

---

## Next Steps

1. **Run full baseline tournament** (30 min)
   ```bash
   python scripts/run_roundrobin.py -n 5 --time-limit 600
   ```

2. **Start RushBot optimization** (overnight)
   ```bash
   caffeinate -i python scripts/optimize_params.py \
     --bot RushBot --opponent DefenseBot \
     --generations 10 --population 20 --matches 3
   ```

3. **Wake up to optimized parameters!**
   - Check `optimization_results/rushbot_vs_defense.json`
   - Apply best params to `bots/rush_bot.py`
   - Test improvement

4. **Repeat for DefenseBot** (next night)

5. **Create final optimized versions**
   - `RushBot v2` (anti-DefenseBot)
   - `DefenseBot v2` (anti-EconomyBot)

Ready to dominate! 🏆
