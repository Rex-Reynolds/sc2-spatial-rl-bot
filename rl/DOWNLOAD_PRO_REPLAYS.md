# How to Download Professional StarCraft II Replays

Complete guide to finding, downloading, and organizing pro Terran replays for imitation learning.

## Best Sources for Pro Replays

### 1. spawningtool.com (RECOMMENDED)
**Best for:** Curated, high-quality replays with build orders

**URL:** https://lotv.spawningtool.com/replays/

**Steps:**
1. Go to https://lotv.spawningtool.com/replays/
2. **Filter replays:**
   - Race: Select "Terran"
   - Patch: Select recent patches (6.0+)
   - Player: Can search for specific pros (ByuN, Maru, Clem, etc.)
3. **Download:**
   - Click on a replay title
   - Click "Download Replay" button
   - Save to `rl/data/replays/terran_pro/`
4. **Repeat** for 50-100 replays

**Pro tip:** Look for replays with:
- High-rated players (Pro/Top GM)
- Recent patches (6.0.x)
- Standard builds (not all-ins or cheese)

### 2. sc2replaystats.com
**Best for:** Large volume, ladder replays

**URL:** https://sc2replaystats.com

**Steps:**
1. Go to https://sc2replaystats.com/replays/
2. **Filter:**
   - League: Master/Grandmaster
   - Race: Terran
   - Game Type: 1v1
   - Patch: Latest
3. **Bulk download:**
   - Can download replay packs
   - Save to `rl/data/replays/terran_pro/`

### 3. AlphaStar Replay Pack (Research Dataset)
**Best for:** Large-scale dataset (if available)

DeepMind released training replays for AlphaStar research. Check:
- https://github.com/deepmind/pysc2
- Research paper supplementary materials

**Note:** May be harder to access/organize

### 4. YouTube VOD Replay Packs
**Best for:** Specific pro players

Many YouTube channels post replay packs:
- Search: "Maru replay pack"
- Search: "Clem replay pack"
- Download from description links

## Organizing Your Replays

### Directory Structure

```bash
rl/data/replays/
├── terran_pro/          # Main directory for training
│   ├── maru_tvz_1.SC2Replay
│   ├── clem_tvt_2.SC2Replay
│   ├── byun_marine_3.SC2Replay
│   └── ...
├── terran_bio/          # Optional: organize by style
│   └── ...
└── terran_mech/         # Optional: organize by style
    └── ...
```

### Create the directory:
```bash
mkdir -p rl/data/replays/terran_pro
```

## Replay Selection Criteria

### ✅ Good Replays for Training

**Characteristics:**
- **High MMR** (6000+, preferably pro players)
- **Recent patches** (6.0+)
- **Standard play** (macro games, not cheese/all-ins)
- **Terran perspective** (we only extract Terran player data)
- **Clean games** (not pauses, disconnects, or bugs)

**Recommended pro players:**
- **Maru** (Multitasking master, macro god)
- **Clem** (Aggressive bio, multitasking)
- **ByuN** (Marine control, harassment)
- **Cure** (Solid macro, timing attacks)
- **TY** (Late game, mech)

### ❌ Avoid These Replays

- **Cheese/all-ins** - Not representative of standard play
- **Old patches** (pre-6.0) - Balance different
- **Low MMR** (<5000) - Poor decision-making
- **Non-Terran** - Won't have Terran player data
- **Team games** (2v2, 3v3) - Different dynamics

## Quick Start: Download Your First 20 Replays

### Step-by-Step

**1. Go to spawningtool.com**
```
https://lotv.spawningtool.com/replays/?search=&tag=&p=Maru&order=-created
```

**2. Download 20 replays:**
- Click each replay → Download
- Save all to: `/Users/rreynolds/programming/ai-starcraft/rl/data/replays/terran_pro/`

**3. Verify downloads:**
```bash
cd rl/data/replays/terran_pro
ls -lh *.SC2Replay | wc -l
# Should show: 20
```

**4. Parse the replays:**
```bash
cd /Users/rreynolds/programming/ai-starcraft
source venv/bin/activate
python rl/replay_parser.py rl/data/replays/terran_pro \
    --output rl/data/maru_replays.pkl \
    --max-replays 20
```

**5. Train imitation model:**
```bash
python rl/train_imitation.py \
    --data rl/data/maru_replays.pkl \
    --output rl/models/maru_imitation \
    --epochs 50
```

**6. Train RL agent against it:**
```bash
python rl/train.py \
    --advanced \
    --self-play \
    --opponent-model rl/models/maru_imitation \
    --episodes 100 \
    --model-name rl_vs_maru
```

## Expected Results by Replay Count

| # Replays | Training Samples | Imitation Accuracy | RL Performance |
|-----------|------------------|-------------------|----------------|
| 10 | ~2,000 | 30-40% | Learns basic patterns |
| 20 | ~4,000 | 40-50% | Recognizes build orders |
| 50 | ~10,000 | 50-60% | Good strategy mimic |
| 100 | ~20,000 | 60-70% | Strong opponent |
| 200+ | ~40,000+ | 70%+ | Pro-level decisions |

**Start with 20-50 replays**, then scale up if results are good.

## Troubleshooting

### "Download is slow"
- spawningtool downloads one at a time
- Consider sc2replaystats for bulk packs
- Or use wget/curl to automate (see below)

### "Can't find good replays"
Search terms that work well on spawningtool:
- Player: "Maru"
- Player: "Clem"
- Tag: "TvZ" (Terran vs Zerg)
- Tag: "Bio"

### "Replays won't parse"
- Check file extension is `.SC2Replay`
- Ensure replays are from recent patches (6.0+)
- Old replay format may not be compatible with sc2reader

## Advanced: Automated Download Script

Create `download_replays.sh`:
```bash
#!/bin/bash
# Automated replay downloader for spawningtool.com

PLAYER="Maru"
OUTPUT_DIR="rl/data/replays/terran_pro"
COUNT=50

# Note: This requires spawningtool API access or web scraping
# For now, manual download is recommended

echo "Please download replays manually from:"
echo "https://lotv.spawningtool.com/replays/?p=$PLAYER"
echo "Save to: $OUTPUT_DIR"
```

## Replay Quality Checklist

Before parsing, verify:
- [ ] At least 20 replay files
- [ ] All files are `.SC2Replay` format
- [ ] Recent patches (6.0+)
- [ ] Mix of matchups (TvZ, TvT, TvP)
- [ ] Pro players (6000+ MMR)
- [ ] Standard games (not all cheese)

## Next Steps

After downloading replays:

1. **Parse replays:**
   ```bash
   python rl/replay_parser.py rl/data/replays/terran_pro
   ```

2. **Train imitation model:**
   ```bash
   python rl/train_imitation.py --data rl/data/pro_replays.pkl
   ```

3. **Use as RL opponent:**
   ```bash
   python rl/train.py --self-play --opponent-model rl/models/pro_imitation
   ```

4. **Compare performance:**
   - vs IdleBot: Easy, learns basics
   - vs MarineMedivacBot: Medium, learns tactics
   - vs Pro Imitation: Hard, learns strategy

## Resources

- **spawningtool.com** - https://lotv.spawningtool.com/replays/
- **sc2replaystats.com** - https://sc2replaystats.com
- **Liquipedia pro players** - https://liquipedia.net/starcraft2/Portal:Players
- **Maru's replays** - https://lotv.spawningtool.com/replays/?p=Maru
- **Clem's replays** - https://lotv.spawningtool.com/replays/?p=Clem

---

**Ready to start?** Download 20 replays from spawningtool.com and run the quick start steps above!
