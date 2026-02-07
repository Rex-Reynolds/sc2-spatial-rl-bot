# Bot Iteration Workflow

Now that you have diverse bot personas, here's how to continuously improve them.

## The Iteration Loop

```
1. Baseline → 2. Optimize → 3. Update → 4. Test → 5. Repeat
     ↑                                              ↓
     └──────────────────────────────────────────────┘
```

---

## Phase 1: Establish Baselines

**Goal**: Understand current bot strengths and weaknesses

### Run Tournament
```bash
cd ~/programming/ai-starcraft
source .venv/bin/activate

# Full tournament (all 5 bots)
python scripts/run_roundrobin.py -n 5 --time-limit 600

# Quick tournament (4 competitive bots)
python scripts/run_roundrobin.py -n 3 --bots RushBot DefenseBot EconomyBot ProxyBot --time-limit 600
```

### Analyze Results

Look at the final standings:
```
Rank Bot          Wins  Losses  Win Rate
1    DefenseBot   12    3       80.0%
2    EconomyBot   10    5       66.7%
3    ProxyBot     8     7       53.3%
4    RushBot      6     9       40.0%
```

**Key insights**:
- DefenseBot is strongest (bunkers OP)
- RushBot needs help vs DefenseBot
- EconomyBot loses to ProxyBot (timing attack)

---

## Phase 2: Optimize Parameters

**Goal**: Find optimal parameters for each bot against specific opponents

### Install Optimizer
```bash
pip install -e ".[optimization]"
```

### Optimize Bots

#### Example 1: RushBot vs DefenseBot
```bash
python scripts/optimize_params.py \
  --bot RushBot \
  --opponent DefenseBot \
  --generations 20 \
  --population 30 \
  --matches 5

# Might discover:
# - MAX_WORKERS: 18 (was 16)
# - BARRACKS_COUNT: 3 (was 2)
# - ATTACK_MARINE_THRESHOLD: 12 (was 8)
# - New win rate: 65% (was 20%)
```

**Why it works**: More marines (12 vs 8) can overwhelm bunkers!

#### Example 2: EconomyBot vs ProxyBot
```bash
python scripts/optimize_params.py \
  --bot EconomyBot \
  --opponent ProxyBot \
  --generations 15 \
  --population 25

# Might discover:
# - MIN_MARINES_BEFORE_EXPAND: 10 (was 6)
# - EXPAND_AT_WORKERS: 16 (was 14)
# - Safer expand timing counters proxy rush
```

### Optimization Strategy

For each bot, optimize against its **weakest matchup**:

| Bot | Weak Against | Optimization Goal |
|-----|--------------|-------------------|
| RushBot | DefenseBot | Bigger army to break bunkers |
| DefenseBot | EconomyBot | Earlier attack before economy snowballs |
| EconomyBot | ProxyBot | Better defense before expanding |
| ProxyBot | DefenseBot | Find weak spots in bunker placement |

---

## Phase 3: Apply Optimizations

### Method A: Update In-Place

Edit the bot file directly:

```python
# In bots/rush_bot.py

# OLD (baseline)
MAX_WORKERS = 16
BARRACKS_COUNT = 2
ATTACK_MARINE_THRESHOLD = 8

# NEW (optimized vs DefenseBot)
MAX_WORKERS = 18
BARRACKS_COUNT = 3
ATTACK_MARINE_THRESHOLD = 12
```

**Pros**: Simple, single version
**Cons**: Loses baseline, might hurt other matchups

### Method B: Create Variants

Create specialized versions:

```bash
# Copy and modify
cp bots/rush_bot.py bots/rush_bot_v2.py

# Edit rush_bot_v2.py with optimized params
# Update class name: class RushBotV2(BotAI)

# Add to bots/__init__.py
from .rush_bot_v2 import RushBotV2
```

**Pros**: Keep baseline, compare versions
**Cons**: More files to manage

### Method C: Parameterized Bots (Advanced)

Refactor bots to accept parameters:

```python
class RushBot(BotAI):
    def __init__(self, max_workers=16, barracks_count=2, attack_threshold=8):
        super().__init__()
        self.max_workers = max_workers
        self.barracks_count = barracks_count
        self.attack_threshold = attack_threshold

# Usage
RushBot()                                    # Baseline
RushBot(max_workers=18, barracks_count=3)   # Optimized
```

**Pros**: Flexible, clean, easy to experiment
**Cons**: Requires refactoring existing code

---

## Phase 4: Test and Compare

### Re-run Tournament

```bash
# Test optimized bots
python scripts/run_roundrobin.py -n 5 --time-limit 600
```

### Compare Results

| Bot | Baseline Win Rate | Optimized Win Rate | Improvement |
|-----|-------------------|-------------------|-------------|
| RushBot | 40% | 58% | +18% ✅ |
| DefenseBot | 80% | 75% | -5% ⚠️ |
| EconomyBot | 67% | 72% | +5% ✅ |

**Insights**:
- RushBot improved significantly vs DefenseBot
- But DefenseBot's optimization hurt its other matchups (trade-off!)
- May need matchup-specific variants

---

## Phase 5: Advanced Strategies

### A. Rock-Paper-Scissors Balance

Create a balanced meta:
- **Anti-Rush Build**: Bunkers, defensive marines
- **Anti-Eco Build**: Fast timing attack
- **Anti-Turtle Build**: Fast expand, out-macro

Each bot has a counter → prevents one dominant strategy

### B. Build Order Variants

Add multiple strategies per bot:

```python
class RushBot(BotAI):
    def __init__(self, strategy="standard"):
        self.strategy = strategy

        if strategy == "standard":
            self.attack_threshold = 8
        elif strategy == "all_in":
            self.attack_threshold = 6  # Faster, riskier
        elif strategy == "safe":
            self.attack_threshold = 12  # Slower, stronger
```

### C. Adaptive Behavior

Make bots scout and adapt:

```python
async def on_step(self, iteration: int):
    # Scout at 1:00
    if self.time == 60:
        await self.scout_enemy()

    # If bunkers detected, build more marines
    if self.enemy_has_bunkers:
        self.attack_threshold = 12  # Wait for bigger army
```

---

## Phase 6: Prepare for RL (Phase 2)

Once scripted bots are strong, they become **training opponents** for RL:

### Create Bot Ladder

```python
# Easy → Hard opponents for RL training
TRAINING_OPPONENTS = [
    ("IdleBot", 0.0),          # Episode 0-1000
    ("RushBot", 0.2),          # Episode 1000-3000
    ("DefenseBot", 0.5),       # Episode 3000-6000
    ("EconomyBot", 0.7),       # Episode 6000-10000
    ("ProxyBot", 0.9),         # Episode 10000+
]
```

RL agent trains against increasingly difficult opponents!

### Hybrid Approach

Use optimized scripted bots to **bootstrap RL**:

1. **Imitation Learning**: RL agent watches optimized RushBot
2. **RL Fine-tuning**: Agent learns to beat RushBot
3. **Self-play**: Agent plays against itself
4. **Curriculum**: Agent faces bot ladder

This is **much faster** than learning from scratch!

---

## Recommended Path Forward

### Week 1: Baseline & Optimization
```bash
# Day 1-2: Baseline tournament
python scripts/run_roundrobin.py -n 10

# Day 3-5: Optimize each bot vs weakest opponent
for bot in RushBot DefenseBot EconomyBot ProxyBot; do
    python scripts/optimize_params.py --bot $bot --opponent <weakest> --generations 20
done

# Day 6-7: Test optimized bots, iterate
```

### Week 2: Variants & Meta
```bash
# Create 2-3 variants per bot
# - Baseline version
# - Anti-rush version
# - Anti-eco version

# Run extended tournament with variants
```

### Week 3-4: Phase 2 - RL
```bash
# Implement Gymnasium wrapper
# Train PPO agent vs bot ladder
# Self-play training
# Compare RL agent vs optimized scripted bots
```

---

## Quick Commands Reference

### Baseline Performance
```bash
python scripts/run_roundrobin.py -n 5 --time-limit 600
```

### Optimize Single Bot
```bash
python scripts/optimize_params.py --bot RushBot --opponent DefenseBot --generations 20
```

### Test Specific Matchup
```bash
python scripts/run_tournament.py -n 10 --map Simple64
# (Modify to use specific bots)
```

### Watch Optimized Bot
```bash
python scripts/run_match.py --realtime
# (Update run_match.py to use optimized bot)
```

---

## Metrics to Track

### Bot Performance
- Overall win rate
- Win rate per matchup
- Average game length
- Resources collected
- Units lost vs killed

### Optimization Progress
- Generation-by-generation fitness
- Best parameters found
- Win rate improvement
- Convergence speed

### Tournament Meta
- Which strategies dominate
- Which matchups are balanced
- Rock-paper-scissors dynamics

---

## Next Phase: Reinforcement Learning

Once you have strong scripted bots (60%+ win rate in bad matchups), you're ready for Phase 2:

### RL Implementation Checklist
- [ ] Gymnasium environment wrapper
- [ ] Observation space (11 features)
- [ ] Action space (7 discrete actions)
- [ ] Reward function
- [ ] PPO training loop
- [ ] Bot ladder for curriculum learning
- [ ] Self-play infrastructure

See `docs/ITERATION_STRATEGY.md` for full RL implementation details.

---

## Summary

The iteration workflow is:

1. **Run tournament** → Identify weaknesses
2. **Optimize parameters** → Find better configs
3. **Update bots** → Apply improvements
4. **Test again** → Measure progress
5. **Repeat** → Continuous improvement

Each iteration makes bots stronger, creating better training opponents for eventual RL agents!
