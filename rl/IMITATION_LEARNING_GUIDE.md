# Imitation Learning from Professional Replays

Train a StarCraft II bot to mimic professional Terran players using behavioral cloning.

## Quick Start

### Step 1: Install Dependencies

```bash
pip install sc2reader
```

### Step 2: Download Professional Replays

Download Terran pro replays from these sources:

1. **spawningtool.com** (recommended - curated replays)
   - Go to: https://lotv.spawningtool.com/replays/
   - Filter: Race = Terran, MMR > 6000
   - Download 50-100 replays

2. **sc2replaystats.com**
   - Filter for high-MMR Terran players
   - Download recent ladder games

3. **Organize replays:**
   ```bash
   mkdir -p rl/data/replays/terran_pro
   # Move all .SC2Replay files here
   ```

### Step 3: Parse Replays into Training Data

```bash
source venv/bin/activate

# Parse all replays in directory
python rl/replay_parser.py rl/data/replays/terran_pro \
    --output rl/data/pro_replays.pkl \
    --max-replays 100

# This will create a .pkl file with (observation, action) pairs
```

### Step 4: Train Imitation Model

```bash
# Train behavioral cloning model on parsed replays
python rl/train_imitation.py \
    --data rl/data/pro_replays.pkl \
    --output rl/models/pro_terran_imitation \
    --epochs 100 \
    --batch-size 64

# Training will show:
# - Training accuracy (how well model fits replay actions)
# - Validation accuracy (generalization)
```

### Step 5: Use Pro Bot as RL Training Opponent

```bash
# Train your RL agent against the pro-mimic bot
python rl/train.py \
    --advanced \
    --self-play \
    --opponent-model rl/models/pro_terran_imitation \
    --episodes 500 \
    --model-name rl_vs_pro

# Your RL agent will now learn by playing against a bot
# that mimics professional Terran strategies!
```

## Architecture

### Replay Parser (`replay_parser.py`)

**Input:** `.SC2Replay` files from pro games

**Processing:**
1. Load replay with `sc2reader`
2. Find Terran player
3. Extract game state every 16 frames
4. Map replay events to our 23-action space
5. Create (observation, action) pairs

**Output:** `.pkl` file with training data

### Imitation Trainer (`train_imitation.py`)

**Input:** Parsed replay data

**Processing:**
1. Behavioral cloning (supervised learning)
2. Neural network learns: `action = policy(observation)`
3. Trained to maximize `P(expert_action | observation)`

**Output:** Trained policy model (SB3-compatible)

### Action Space Mapping

Pro replays have hundreds of low-level commands. We map them to 23 high-level actions:

| Replay Event | Our Action |
|-------------|------------|
| `TrainUnitCommand(SCV)` | `train_scv` (0) |
| `TrainUnitCommand(Marine)` | `train_marine` (9) |
| `BuildCommand(Barracks)` | `build_barracks` (3) |
| `ResearchCommand(Stimpack)` | `research_stim` (14) |
| `AttackCommand(...)` | `attack` (19) |
| Other events | `no_op` (22) |

## Expected Results

### Imitation Model Performance

**Good training indicators:**
- Validation accuracy: 60-80%
- Train/val loss decreasing
- Common actions (train SCVs, build depots) predicted accurately

**Don't expect 100% accuracy** - that's overfitting. 60-70% is realistic.

### RL Agent Performance

After training against the pro-mimic bot:
- Should learn faster than training vs IdleBot
- Learns more sophisticated strategies
- May develop counter-strategies to pro builds

## Limitations & Improvements

### Current Limitations

1. **Simplified state extraction** - Replay parser uses dummy observations
   - Need to properly track all units/buildings throughout replay
   - Requires more complex `GameState` tracking

2. **Action aggregation** - Maps complex micro to high-level actions
   - Loses nuance (unit positioning, precise timings)
   - Pro players do 300+ APM, our bot does ~4 decisions/second

3. **Data requirements** - Needs 50-100+ replays for good coverage
   - More replays = better generalization
   - Different pro styles may conflict

### Improvements to Implement

#### 1. Proper State Tracking

```python
class GameState:
    """Track complete game state from replay events."""

    def __init__(self):
        self.units = {}  # {unit_id: Unit}
        self.buildings = {}  # {building_id: Building}
        self.upgrades = set()
        self.resources = {'minerals': 50, 'gas': 0}
        # ...

    def update(self, event):
        """Update state based on event type."""
        if isinstance(event, UnitBornEvent):
            self.units[event.unit_id] = Unit(event)
        elif isinstance(event, UnitDiedEvent):
            del self.units[event.unit_id]
        # ... handle all event types

    def to_observation(self) -> np.ndarray:
        """Convert current state to 26-feature observation."""
        # Count units by type
        scvs = len([u for u in self.units.values() if u.type == 'SCV'])
        marines = len([u for u in self.units.values() if u.type == 'Marine'])
        # ...
        return np.array([...], dtype=np.float32)
```

#### 2. Data Augmentation

```python
# Flip map (mirror observations/actions)
def augment_replay(obs, actions):
    flipped_obs = flip_observation(obs)
    flipped_actions = flip_actions(actions)
    return [(obs, actions), (flipped_obs, flipped_actions)]
```

#### 3. Curriculum Learning

Train RL agent with increasing difficulty:
1. vs IdleBot (100 episodes) → Learn basics
2. vs Pro-mimic bot (500 episodes) → Learn strategy
3. vs Self-play (1000 episodes) → Refine

## Troubleshooting

### "No module named 'sc2reader'"
```bash
pip install sc2reader
```

### "No Terran player found"
- Replay might be from a different race
- Filter replays before parsing

### Low imitation accuracy (<40%)
- Need more replays
- Check if observations are extracted correctly
- Increase model capacity or training epochs

### Pro bot doesn't play well in-game
- Imitation model only mimics decisions, not execution
- Still uses our simplified action execution
- Consider it a "strategy advisor" not a perfect clone

## Next Steps

1. ✅ Parse 100 pro replays
2. ✅ Train imitation model
3. 🔄 Train RL agent vs pro bot
4. 📊 Compare performance: RL-vs-IdleBot vs RL-vs-Pro
5. 🚀 Implement self-play for emergent strategies

## Resources

- **spawningtool.com** - Pro replays with build orders
- **sc2reader docs** - https://sc2reader.readthedocs.io/
- **Behavioral Cloning paper** - Ross & Bagnell (2010)
