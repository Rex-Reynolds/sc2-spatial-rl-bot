# Iteration & Improvement Guide

## Current Issues Identified

### 1. **Sparse Rewards** (HIGH PRIORITY)
**Problem:** Bot only gets rewards at end of game (+10 win / -10 loss)
**Impact:** Slow learning, no feedback on what actions were good

**Fix:**
```python
def _calculate_step_reward(self) -> float:
    reward = 0.0

    # Current: Only kill/loss rewards
    # Better: Add intermediate milestones

    # Economy benchmarks (encourage macro)
    if self.time >= 120 and self.supply_workers >= 20:
        reward += 0.5  # Good economy at 2 min
    if self.time >= 300 and self.supply_workers >= 40:
        reward += 1.0  # Excellent economy at 5 min

    # Tech benchmarks (encourage progression)
    if self.structures(UnitTypeId.BARRACKS).amount >= 2:
        reward += 0.2  # Production
    if self.structures(UnitTypeId.FACTORY).ready:
        reward += 0.3  # Tech progression
    if UpgradeId.STIMPACK in self.state.upgrades:
        reward += 0.5  # Key upgrade

    # Expansion rewards (encourage economy)
    if self.townhalls.amount >= 2:
        reward += 0.5 * (self.time / 600.0)  # Scale with game time

    # Supply management (discourage supply blocks)
    if self.supply_left < 3 and self.supply_cap < 200:
        reward -= 0.1  # Penalty for supply block

    # Army value (encourage unit production)
    army_value = (
        self.units(UnitTypeId.MARINE).amount * 1 +
        self.units(UnitTypeId.MARAUDER).amount * 2 +
        self.units(UnitTypeId.SIEGETANK).amount * 3
    )
    reward += army_value * 0.001  # Small continuous reward

    return reward
```

**Expected improvement:** 2-3x faster learning

---

### 2. **Better Observation Features**

**Add spatial/positional information:**
```python
# Current: Global counts only
# Better: Add positioning awareness

def _get_observation(self):
    obs = [
        # ... existing features ...

        # NEW: Spatial awareness
        self._get_army_position_ratio(),  # Where is army? (0=home, 1=enemy)
        self._get_expansion_count_ratio(),  # Bases relative to map size
        self._get_enemy_army_position(),  # Enemy threat location

        # NEW: Threat assessment
        self._get_army_supply_difference(),  # Who has bigger army?
        self._get_tech_advantage(),  # Who has better tech?

        # NEW: Economic efficiency
        self._get_worker_saturation(),  # Are bases saturated?
        self._get_income_rate(),  # Minerals/gas per second
    ]
    return np.array(obs, dtype=np.float32)
```

---

### 3. **Curriculum Learning** (Structured Progression)

**Instead of random training, use a ladder:**

```bash
# Stage 1: Master basics (50 episodes)
python rl/train.py --advanced --opponent IdleBot --episodes 50

# Stage 2: Learn defense (100 episodes)
python rl/train.py --advanced --opponent RushBot --episodes 100 \
    --load-model stage1_final.zip

# Stage 3: Learn macro (100 episodes)
python rl/train.py --advanced --opponent MarineMedivacBot --episodes 100 \
    --load-model stage2_final.zip

# Stage 4: Learn strategy (200 episodes)
python rl/train.py --advanced --self-play --episodes 200 \
    --load-model stage3_final.zip

# Stage 5: Master the game (500 episodes)
python rl/train.py --advanced --self-play --episodes 500 \
    --load-model stage4_final.zip
```

**Expected improvement:** 40-50% higher final win rate

---

## 🔬 Medium Effort (Higher Impact)

### 4. **Hyperparameter Tuning**

**Test different learning configurations:**

```python
# Current settings (in train.py)
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
gamma = 0.99

# Try these variations:
# Fast learner (more aggressive updates)
learning_rate = 1e-3, n_steps = 512, batch_size = 128

# Slow learner (more stable)
learning_rate = 1e-4, n_steps = 4096, batch_size = 32

# Long-term planner (values future more)
gamma = 0.995  # or 0.999
```

**Run experiments:**
```bash
# Experiment 1: Fast learning
python rl/train.py --advanced --self-play --episodes 50 \
    --model-name experiment_fast --lr 0.001

# Experiment 2: Long-term thinking
python rl/train.py --advanced --self-play --episodes 50 \
    --model-name experiment_longterm --gamma 0.995

# Compare results in TensorBoard
tensorboard --logdir=rl/logs/
```

---

### 5. **Better Neural Network Architecture**

**Current:** Small MLP (64-64 hidden units)
**Better:** Deeper network with more capacity

```python
# In train.py, when creating PPO:
policy_kwargs = dict(
    net_arch=[128, 128, 128],  # Deeper network
    activation_fn=nn.ReLU,  # Different activation
)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    # ... other params
)
```

**Expected improvement:** 10-20% better decision quality

---

### 6. **Population-Based Training**

**Train multiple agents simultaneously:**

```bash
# Create 4 agents with different hyperparameters
for i in {1..4}; do
    python rl/train.py --advanced --self-play \
        --episodes 100 \
        --model-name population_agent_$i \
        --lr $(python -c "import random; print(random.uniform(1e-4, 1e-3))") &
done

# Every 25 episodes, have them play each other
# Keep best performers, discard worst
```

**This is how AlphaStar and OpenAI Five reached mastery!**

---

## 🚀 Advanced (Highest Impact, Most Effort)

### 7. **Fix Pro Replay Parsing** (CRITICAL)

**Current issue:** Observations are dummy/random data

**Fix the replay parser to extract REAL game state:**

```python
class GameState:
    """Track actual game state from replay events."""

    def __init__(self):
        self.minerals = 50
        self.gas = 0
        self.units = {}  # {unit_id: Unit}
        self.buildings = {}
        self.upgrades = set()
        self.frame = 0

    def update(self, event):
        """Update state based on replay event."""
        if isinstance(event, UnitBornEvent):
            self.units[event.unit_id] = {
                'type': event.unit_type,
                'position': event.location,
            }
        elif isinstance(event, UnitDiedEvent):
            if event.unit_id in self.units:
                del self.units[event.unit_id]
        elif isinstance(event, PlayerStatsEvent):
            self.minerals = event.minerals_current
            self.gas = event.vespene_current
        # ... handle all event types

    def to_observation(self):
        """Convert to 26-feature vector."""
        # Count units by type
        marines = len([u for u in self.units.values() if u['type'] == 'Marine'])
        # ... count all unit types

        return np.array([
            self.minerals / 2000.0,
            self.gas / 2000.0,
            # ... all 26 features with REAL data
        ], dtype=np.float32)
```

**Impact:** Pro imitation bot will actually work correctly (84.9% → useful opponent)

---

### 8. **Add Convolutional Layers** (Spatial Understanding)

**Current:** Bot doesn't know WHERE things are
**Better:** Add CNN to process spatial information

```python
# Create mini-map representation
def get_spatial_features(self):
    """Create 64x64 feature maps."""
    # Channel 1: Friendly units
    # Channel 2: Enemy units
    # Channel 3: Resources
    # Channel 4: Terrain
    return np.stack([...], axis=0)  # Shape: (4, 64, 64)

# Use CNN policy
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)
```

**Impact:** Bot learns positioning, map control, flanking

---

### 9. **Multi-Task Learning**

**Train on multiple objectives simultaneously:**

```python
# Separate value heads for different goals
class MultiTaskPolicy(ActorCriticPolicy):
    def __init__(self, ...):
        super().__init__(...)
        self.economy_value = nn.Linear(64, 1)
        self.combat_value = nn.Linear(64, 1)
        self.tech_value = nn.Linear(64, 1)

    # Train bot to optimize:
    # 1. Economy (worker count, expansions)
    # 2. Combat (kills, map control)
    # 3. Tech (upgrades, unit variety)
```

---

### 10. **Imitation Pre-training + RL Fine-tuning**

**Fix pro replay parsing, then:**

```bash
# Step 1: Pre-train on pro replays (warm start)
python rl/train_imitation.py \
    --data pro_replays.pkl \
    --output models/pretrained_from_pros \
    --epochs 200

# Step 2: Fine-tune with RL (discover novel strategies)
python rl/train.py --advanced --self-play \
    --load-model models/pretrained_from_pros.zip \
    --episodes 500
```

**This combines human knowledge + RL optimization!**

---

## 📊 Evaluation & Analysis

### 11. **Better Metrics**

**Beyond win rate:**

```python
# Track strategic metrics:
- Average game length (winning faster = better)
- Economy score (workers, expansions, income)
- Tech progression rate (when do upgrades finish?)
- Army efficiency (damage dealt / damage taken)
- Map control (% of map explored/controlled)
- Build order consistency (deviation from optimal timing)
```

### 12. **Head-to-Head Tournament**

```bash
# Compare all your models
python rl/tournament.py \
    --models demo_advanced_final.zip \
             morning_training_final.zip \
             true_self_play_final.zip \
    --games-per-matchup 10
```

---

## 🎯 Recommended Priority Order

**Week 1: Quick Wins**
1. ✅ Fix reward shaping (2-3 hours)
2. ✅ Add curriculum learning (use existing bots)
3. ✅ Run hyperparameter experiments (parallel overnight)

**Week 2: Medium Effort**
4. ✅ Bigger neural network
5. ✅ Better observations (spatial awareness)
6. ✅ Population-based training

**Week 3: Advanced**
7. ✅ Fix replay parser (get real pro data)
8. ✅ Add CNN for spatial reasoning
9. ✅ Imitation pre-training + RL fine-tuning

---

## 💡 Immediate Next Steps

**What I'd do RIGHT NOW:**

1. **Let self-play finish** (100 episodes, ~5 hours)
2. **While it runs:** Implement better reward shaping
3. **Test improved rewards** with 20-episode experiment
4. **Compare:** Old rewards vs new rewards (which learns faster?)
5. **Iterate:** Based on results

**Want me to implement the improved reward function now?** That's the single highest-impact change we can make quickly!
