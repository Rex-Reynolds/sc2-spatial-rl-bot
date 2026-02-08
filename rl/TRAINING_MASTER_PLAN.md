
# Training Master Plan - Complete Guide

## 🎯 Training Progression Path

### Phase 1: Validation (✅ IN PROGRESS)
**Goal:** Validate pipeline works end-to-end

```bash
python rl/train_spatial.py --opponent IdleBot --episodes 10
```

**Expected:**
- All episodes complete without crashes
- Reward trending upward
- Bot beats IdleBot consistently
- **Time:** ~30-40 minutes
- **Status:** RUNNING NOW (Episode 2/10)

---

### Phase 2: Extended Training (50-100 episodes)
**Goal:** Achieve consistent performance

```bash
# Option A: Continue vs IdleBot (baseline)
./rl/train_spatial_long.sh  # Interactive, choose opponent

# Option B: Manual command
python rl/train_spatial.py \
    --opponent IdleBot \
    --episodes 100 \
    --model-name spatial_100ep \
    --load-model rl/models/spatial_first_run/final_model.pt \
    --use-lstm
```

**Expected:**
- Win rate vs IdleBot: 95-100%
- Build orders emerging
- Better spatial placement
- **Time:** ~8-10 hours

---

### Phase 3: Curriculum Learning (450 episodes)
**Goal:** Progressive difficulty, master all opponents

```bash
./rl/train_curriculum.sh
```

**Breakdown:**
1. **Stage 1 (50 ep):** IdleBot - Master basics
2. **Stage 2 (100 ep):** RushBot - Learn defense
3. **Stage 3 (100 ep):** MarineMedivacBot - Learn macro
4. **Stage 4 (200 ep):** Self-play - Strategic mastery

**Expected:**
- Win vs IdleBot: 100%
- Win vs RushBot: 70-80%
- Win vs MarineMedivacBot: 50-60%
- Sophisticated self-play strategies
- **Time:** ~30-40 hours

---

### Phase 4: Hyperparameter Optimization
**Goal:** Find best training settings

```bash
./rl/train_population.sh
```

**Trains 4 agents in parallel:**
- Agent 1: Baseline
- Agent 2: Fast learner (high LR)
- Agent 3: Stable learner (low LR)
- Agent 4: Long-term planner (high gamma)

**Analysis:**
```bash
tensorboard --logdir=rl/logs/
# Compare learning curves
# Identify best performer
# Use optimal hyperparameters for final training
```

**Expected:**
- Identify optimal learning rate
- Best batch size for stability
- Optimal decision frequency
- **Time:** ~4-5 hours (parallel)

---

### Phase 5: Final Training (500+ episodes)
**Goal:** Reach competitive level

```bash
# Using best hyperparameters from Phase 4
python rl/train_spatial.py \
    --opponent IdleBot \
    --episodes 500 \
    --model-name spatial_final \
    --load-model rl/models/curriculum_stage4_selfplay/final_model.pt \
    --learning-rate <BEST_LR> \
    --gamma <BEST_GAMMA> \
    --use-lstm \
    --cuda  # If available
```

**Expected:**
- Consistent wins vs all scripted bots
- Advanced micro (kiting, splitting)
- Strategic positioning (walls, high ground)
- Multi-prong attacks
- **Time:** ~40-50 hours
- **Level:** Diamond league equivalent

---

## 📊 Monitoring & Analysis

### TensorBoard
```bash
tensorboard --logdir=rl/logs/
```

**Key metrics to watch:**
- Episode reward (trending up?)
- Episode length (getting faster or slower?)
- Policy loss (stable?)
- Value loss (decreasing?)
- Entropy (maintaining exploration?)

### Manual Testing
```bash
# Test specific model
python rl/test_spatial_game.py \
    --model rl/models/spatial_100ep/final_model.pt

# Tournament evaluation
python rl/tournament.py \
    --models spatial_100ep curriculum_stage4 spatial_final
```

---

## 🔧 Troubleshooting

### Training freezes
- **Solution:** Kill SC2 processes, reduce episodes per run
- **Prevention:** Save checkpoints frequently (`--save-freq 5`)

### Low reward / not learning
- **Check:** TensorBoard entropy (too low = no exploration)
- **Fix:** Increase `--ent-coef` to 0.02 or 0.05
- **Fix:** Lower learning rate

### Reward plateaus
- **Check:** Action masking enabled (prevents invalid actions)
- **Fix:** Switch to harder opponent
- **Fix:** Increase model capacity (deeper network)

### Out of memory
- **Solution:** Reduce batch size (`--batch-size 32`)
- **Solution:** Lower resolution (64×64 instead of 128×128)
- **Solution:** Disable LSTM (`remove --use-lstm`)

---

## 🚀 Advanced Techniques

### 1. Higher Resolution (128×128)
```python
# Edit spatial_features.py
SCREEN_SIZE = 128  # was 64
MINIMAP_SIZE = 128  # was 64

# Update env and policy accordingly
# Requires GPU with 8GB+ VRAM
```

### 2. Multi-Agent League Training
```bash
# Train population
./rl/train_population.sh

# Best agents play each other
# Continuously evolve strategies
# This is how AlphaStar reached Grandmaster!
```

### 3. Imitation Pre-training
```bash
# First: Train on pro replays (if parser fixed)
python rl/train_imitation.py --data pro_replays.pkl

# Then: Fine-tune with RL
python rl/train_spatial.py --load-model imitation_model.pt
```

### 4. Reward Curriculum
```python
# Start with sparse rewards (win/loss only)
# Gradually add shaped rewards
# This prevents reward hacking
```

---

## 📈 Expected Performance Timeline

| Training Time | Episodes | Win Rate | League Level | Key Abilities |
|---------------|----------|----------|--------------|---------------|
| **10-30 min** | 10 | 80% vs Idle | Bronze | Basic macro |
| **8-10 hrs** | 100 | 95% vs Idle | Silver | Build orders |
| **30-40 hrs** | 450 | 70% vs Rush | Gold | Defense, positioning |
| **40-50 hrs** | 500+ | 50% vs Macro | Platinum | Advanced micro |
| **100+ hrs** | 1000+ | Self-play master | Diamond | Strategic depth |

---

## 💾 Checkpointing Strategy

### Frequent Checkpoints (Phase 1-2)
```bash
--save-freq 5  # Every 5 episodes
```
**Why:** Early training is unstable, save often

### Less Frequent (Phase 3+)
```bash
--save-freq 20  # Every 20 episodes
```
**Why:** Stable training, less disk space

### Always Save
- After each curriculum stage
- Before switching opponents
- Before major hyperparameter changes
- Best performing models (based on TensorBoard)

---

## 🎓 Learning Resources

### Understanding Your Bot
```bash
# Visualize feature maps
python rl/visualize_features.py --model <MODEL>

# Analyze action distributions
python rl/analyze_actions.py --model <MODEL>

# Watch replays
# SC2 saves replays to ~/Documents/StarCraft II/Accounts/*/Replays/
```

### Performance Analysis
- **TensorBoard:** Learning curves, losses
- **Manual observation:** Watch games, identify mistakes
- **Ablation studies:** Remove features, see impact
- **A/B testing:** Compare models head-to-head

---

## 🏆 Milestone Checklist

- [ ] **Validation:** 10 episodes complete, bot learns
- [ ] **Extended:** 100 episodes, consistent vs IdleBot
- [ ] **Curriculum Stage 1:** Beat IdleBot 100%
- [ ] **Curriculum Stage 2:** Beat RushBot 70%+
- [ ] **Curriculum Stage 3:** Beat MarineMedivacBot 50%+
- [ ] **Curriculum Stage 4:** Sophisticated self-play
- [ ] **Optimization:** Identify best hyperparameters
- [ ] **Final Training:** 500+ episodes, Diamond level
- [ ] **Tournament:** Beat all scripted bots reliably

---

## 📞 Quick Commands Reference

```bash
# Current training status
tail -f /private/tmp/.../b889c29.output

# Start validation (10 ep)
python rl/train_spatial.py --opponent IdleBot --episodes 10

# Long training (100 ep)
./rl/train_spatial_long.sh

# Full curriculum (450 ep)
./rl/train_curriculum.sh

# Population training (parallel)
./rl/train_population.sh

# TensorBoard
tensorboard --logdir=rl/logs/

# Test model
python rl/test_spatial_game.py

# Make scripts executable
chmod +x rl/*.sh
```

---

**You're on the path to world-class SC2 AI!** 🚀

Current progress: Episode 2/10 (Validation phase)
