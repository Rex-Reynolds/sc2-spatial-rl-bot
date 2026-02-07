# StarCraft II AI vs AI Framework

A Python framework for building and testing StarCraft II bots, designed to serve as a foundation for reinforcement learning experiments.

## Prerequisites

Before running the bots, you need to complete these manual setup steps:

### 1. Install StarCraft II

Download and install the free version of StarCraft II via Battle.net:
- Download from: https://www.blizzard.com/apps/battle.net/desktop
- StarCraft II will be installed to `/Applications/StarCraft II/`

### 2. Download Maps

Download the Melee map pack:
```bash
curl -O https://blzdistsc2-a.akamaihd.net/MapPacks/Melee.zip
# Password when extracting: iagreetotheeula
unzip Melee.zip
# Copy Simple64.SC2Map to StarCraft II Maps directory
cp "Melee/Simple64.SC2Map" "/Applications/StarCraft II/Maps/"
```

## Installation

```bash
# Clone/navigate to the project directory
cd ~/programming/ai-starcraft

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

## Project Structure

```
ai-starcraft/
├── bots/
│   ├── __init__.py
│   ├── idle_bot.py       # Minimal opponent bot
│   └── rush_bot.py       # Aggressive marine rush bot
├── scripts/
│   ├── run_match.py      # Single match runner
│   └── run_tournament.py # Tournament with statistics
├── replays/              # Saved game replays (gitignored)
├── pyproject.toml        # Project configuration
└── README.md
```

## Usage

### Run a Single Match

```bash
# Basic usage (fast simulation mode)
python scripts/run_match.py

# Watch in real-time
python scripts/run_match.py --realtime

# Custom map and replay location
python scripts/run_match.py --map Simple64 --replay replays/my_match.SC2Replay

# Set time limit (default 300s)
python scripts/run_match.py --time-limit 600
```

### Run a Tournament

```bash
# Run 10 matches and show win statistics
python scripts/run_tournament.py -n 10

# Run 5 matches and save all replays
python scripts/run_tournament.py -n 5 --save-replays

# Custom map
python scripts/run_tournament.py -n 10 --map Simple64
```

## Bots

### RushBot

An aggressive Terran bot that executes a marine rush strategy:
- Trains SCVs up to 12 workers
- Builds 2 barracks
- Trains marines continuously
- Attacks when 8+ marines are ready (all-in with workers)

**Tunable constants** (in `bots/rush_bot.py`):
- `MAX_WORKERS` (default: 12)
- `BARRACKS_COUNT` (default: 2)
- `ATTACK_MARINE_THRESHOLD` (default: 8)

### IdleBot

A passive opponent that only trains SCVs and mines. Serves as a practice target.

## Iterating on Bot Strategies

To tune the RushBot performance:

1. Edit constants in `bots/rush_bot.py`
2. Run a tournament: `python scripts/run_tournament.py -n 10`
3. Analyze win rate and adjust parameters

Example experiments:
- Try `ATTACK_MARINE_THRESHOLD` at 6, 8, 10, 12
- Try `BARRACKS_COUNT` at 2 vs 3
- Build barracks closer to enemy (proxy strategy)

## Viewing Replays

Saved replays can be opened in StarCraft II:
1. Open StarCraft II
2. Go to Replays
3. Navigate to your replay files
4. Watch the game playback

## Troubleshooting

**Map not found error:**
- Ensure `Simple64.SC2Map` is in `/Applications/StarCraft II/Maps/`
- Path is case-sensitive

**SC2 path issues:**
- burnysc2 auto-detects macOS installation
- If non-standard location, set `SC2PATH` environment variable

**Python version:**
- Requires Python 3.10+
- If using Python 3.14 causes issues, try Python 3.12 via pyenv

## Future: Reinforcement Learning

Phase 2 will add an `rl/` directory with:
- Gymnasium environment wrapper
- RL agent bot integration
- Training scripts using Stable-Baselines3 PPO
- Observation space: 11 features (resources, unit counts, etc.)
- Action space: 7 discrete actions (train, build, attack, etc.)

To install RL dependencies:
```bash
pip install -e ".[rl]"
```

## License

This is a learning project. Feel free to use and modify.
