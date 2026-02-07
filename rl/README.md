# Phase 2: Reinforcement Learning

Train an RL agent to play StarCraft II!

## Setup

```bash
# Install RL dependencies
pip install -e ".[rl]"
```

## Quick Start

### 1. Train Against IdleBot (Easy)

```bash
python rl/train.py --opponent IdleBot --episodes 100
```

This trains for 100 episodes (~30-60 min).

### 2. Watch Training Progress

```bash
# In another terminal
tensorboard --logdir=./rl/logs/sc2_ppo
```

Open http://localhost:6006 to see:
- Episode rewards
- Win rate over time
- Learning curves

### 3. Test Trained Agent

```bash
python rl/test.py --model rl/models/sc2_ppo_final.zip --episodes 10
```

## Training Curriculum

Train against progressively harder opponents:

### Stage 1: Learn Basics (vs IdleBot)
```bash
python rl/train.py --opponent IdleBot --episodes 500
```

**Goal**: Learn to build workers, supply depots, barracks, marines, and attack.

**Expected**: 80-100% win rate after 500 episodes.

### Stage 2: Learn Defense (vs RushBot)
```bash
python rl/train.py \
  --opponent RushBot \
  --episodes 1000 \
  --load-model rl/models/sc2_ppo_final.zip
```

**Goal**: Learn to defend early rushes while building economy.

**Expected**: 50-70% win rate after 1000 episodes.

### Stage 3: Advanced Strategy (vs DefenseBot)
```bash
python rl/train.py \
  --opponent DefenseBot \
  --episodes 2000 \
  --load-model rl/models/sc2_ppo_final.zip
```

**Goal**: Learn to break defensive positions with larger armies.

**Expected**: 40-60% win rate after 2000 episodes.

## How It Works

### Observation Space (11 features)
```python
[
    minerals,           # 0-1 (normalized)
    vespene_gas,       # 0-1
    supply_used,       # 0-1
    supply_cap,        # 0-1
    scv_count,         # 0-1
    marine_count,      # 0-1
    barracks_count,    # 0-1
    enemy_units,       # 0-1
    enemy_structures,  # 0-1
    game_time,         # 0-1 (10 min max)
    army_strength,     # 0-1 (relative to enemy)
]
```

### Action Space (7 discrete actions)
```python
0: train_scv          # Build worker
1: build_supply_depot # Build supply
2: build_barracks     # Build production
3: train_marine       # Build army
4: attack             # Offensive
5: defend             # Defensive
6: no_op              # Do nothing
```

### Reward Function
```python
+10.0  - Win game
-10.0  - Lose game
+0.1   - Per enemy unit killed
-0.05  - Per own unit lost
+0.0001 - Per mineral collected
+0.0002 - Per gas collected
```

## Architecture

```
rl/
├── env.py          # Gymnasium environment wrapper
├── rl_bot.py       # Bot controlled by RL agent
├── train.py        # Training script (PPO)
├── test.py         # Testing/evaluation script
├── models/         # Saved models
└── logs/           # TensorBoard logs
```

## Training Tips

### Fast Iteration
- Train on CPU first (no GPU needed for small models)
- Use short episodes (--episodes 10) to test pipeline
- Watch first few episodes to verify agent is learning

### Debugging
- Check TensorBoard for flat learning curves (might need hyperparameter tuning)
- If agent always loses, opponent might be too hard (start with IdleBot)
- If agent always wins, move to harder opponent

### Hyperparameter Tuning
```python
# In train.py, modify PPO parameters:
learning_rate=3e-4   # Lower if unstable, higher if too slow
n_steps=2048         # More steps = more stable but slower
batch_size=64        # Larger = more stable but needs more memory
gamma=0.99           # Discount factor (0.95-0.995 for SC2)
```

## Expected Results

### After 100 episodes vs IdleBot:
- Agent learns to build SCVs
- Agent learns to build supply depots
- Agent learns to build barracks
- **Win rate: 50-70%**

### After 500 episodes vs IdleBot:
- Agent masters basic build order
- Agent learns to attack with marines
- **Win rate: 90-100%**

### After 1000 episodes vs RushBot:
- Agent learns to defend early pressure
- Agent learns timing windows
- **Win rate: 50-60%**

### After 5000+ episodes (all opponents):
- Agent develops adaptive strategies
- Agent learns matchup-specific play
- **Can beat scripted bots consistently**

## Troubleshooting

### "Module 'gymnasium' not found"
```bash
pip install -e ".[rl]"
```

### "SC2 game crashes during training"
Lower the game_step to reduce load:
```python
# In rl_bot.py, line 45:
self.client.game_step = 16  # Was 8, higher = slower but more stable
```

### "Training is too slow"
- Reduce episodes: `--episodes 50`
- Use faster opponent: `--opponent IdleBot`
- Increase game_step (faster but less precise)

### "Agent isn't learning"
- Check reward function (might be too sparse)
- Try lower learning_rate: `learning_rate=1e-4`
- Increase training time (RL needs lots of data!)

## Next Steps

1. **Train baseline agent**: 500 episodes vs IdleBot
2. **Evaluate performance**: Test with rl/test.py
3. **Curriculum learning**: Progress through bot ladder
4. **Add features**: Expand observation space (add unit positions, etc.)
5. **Improve actions**: Add more granular actions (where to build, which unit to attack)
6. **Self-play**: Train agent against itself for emergent strategies

Good luck training! 🤖🎮
