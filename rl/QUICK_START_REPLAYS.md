# Quick Start: Pro Replay Imitation Learning

## Step 1: Download Replays (Manual - Do This Now!)

### Best Sources

**Option 1: spawningtool.com (EASIEST)**
- Direct link: https://lotv.spawningtool.com/replays/?p=Maru
- Click each replay → "Download Replay" button
- Save to: `/Users/rreynolds/programming/ai-starcraft/rl/data/replays/terran_pro/`

**Pro player links:**
- Maru: https://lotv.spawningtool.com/replays/?p=Maru
- Clem: https://lotv.spawningtool.com/replays/?p=Clem
- ByuN: https://lotv.spawningtool.com/replays/?p=ByuN
- Cure: https://lotv.spawningtool.com/replays/?p=Cure

### How Many?
- **Minimum:** 20 replays (good start)
- **Recommended:** 50 replays (better coverage)
- **Ideal:** 100+ replays (excellent training data)

### What to Look For
✅ Recent (2024-2025)
✅ High MMR (6000+)
✅ Standard games (not cheese)
✅ Terran player wins (better strategy examples)
✅ Mix of matchups (TvZ, TvT, TvP)

---

## Step 2: Verify Downloads

Once you've downloaded replays:

```bash
cd /Users/rreynolds/programming/ai-starcraft

# Count replays
ls rl/data/replays/terran_pro/*.SC2Replay | wc -l

# Check file sizes (should be 50KB-500KB each)
ls -lh rl/data/replays/terran_pro/
```

---

## Step 3: Parse Replays (Run This After Downloading)

```bash
source venv/bin/activate

# Parse all replays into training data
python rl/replay_parser.py rl/data/replays/terran_pro \
    --output rl/data/maru_pro_replays.pkl \
    --max-replays 50

# This will:
# - Read each .SC2Replay file
# - Extract (observation, action) pairs
# - Save to .pkl file for training
```

**Expected output:**
```
Found 50 replay files
[1/50] Parsing: game1.SC2Replay
  ✓ Extracted 120 (obs, action) pairs
[2/50] Parsing: game2.SC2Replay
  ✓ Extracted 95 (obs, action) pairs
...
✓ Total training samples: 5000+
```

---

## Step 4: Train Imitation Model

```bash
# Train behavioral cloning model on pro replays
python rl/train_imitation.py \
    --data rl/data/maru_pro_replays.pkl \
    --output rl/models/maru_imitation \
    --epochs 100 \
    --batch-size 64

# Training will show:
# Epoch 1/100
#   Train Loss: 2.45  |  Train Acc: 25%
#   Val Loss: 2.30    |  Val Acc: 28%
# ...
# Epoch 100/100
#   Train Loss: 1.45  |  Train Acc: 65%
#   Val Loss: 1.52    |  Val Acc: 62%
```

**Target accuracy:** 50-70% (don't expect 100%)

---

## Step 5: Use Pro Bot as Training Opponent

```bash
# Train your RL agent against the pro-mimic bot
python rl/train.py \
    --advanced \
    --self-play \
    --opponent-model rl/models/maru_imitation \
    --episodes 100 \
    --model-name rl_vs_maru_imitation \
    --load-model rl/models/demo_advanced_final.zip
```

---

## Timeline

- **Download replays:** 15-30 minutes (manual)
- **Parse replays:** 5-10 minutes (50 replays)
- **Train imitation:** 5-15 minutes (100 epochs)
- **Train RL vs pro:** 3-5 hours (100 episodes)

---

## Troubleshooting

### "No replays found"
Make sure files end with `.SC2Replay` (capital letters)

### "Parse error: No Terran player"
Some replays don't have Terran - parser will skip automatically

### "Imitation accuracy is low (<40%)"
Normal if you only have 10-20 replays. Try downloading more.

### "sc2reader not found"
Already installed! But if needed: `pip install sc2reader`

---

## What You'll Get

After completing all steps:

**Files:**
- `rl/data/maru_pro_replays.pkl` - Parsed training data
- `rl/models/maru_imitation.zip` - Pro-mimic bot

**Capabilities:**
- Bot that mimics Maru's decision-making
- Much stronger opponent than scripted bots
- Your RL agent will learn advanced strategies

---

## Next Steps After Imitation Training

1. **Compare performance:**
   - vs IdleBot: 100% win (easy)
   - vs MarineMedivacBot: ~40-50% win (medium)
   - vs Maru Imitation: ~20-30% win (hard)

2. **Curriculum learning:**
   ```bash
   # Progressive training
   python rl/train.py --opponent IdleBot --episodes 50
   python rl/train.py --opponent MarineMedivacBot --episodes 100
   python rl/train.py --self-play --opponent-model maru_imitation --episodes 200
   ```

3. **Self-play final stage:**
   ```bash
   # Agent vs itself for emergent strategies
   python rl/train.py --advanced --self-play --episodes 500
   ```

---

## Current Status Reminder

**Background:** Your morning_training (50 episodes vs MarineMedivacBot) is still running
**Your task:** Download 20-50 pro replays from spawningtool.com
**Save to:** `/Users/rreynolds/programming/ai-starcraft/rl/data/replays/terran_pro/`

**Once you have replays, run Steps 3-5 above!**
