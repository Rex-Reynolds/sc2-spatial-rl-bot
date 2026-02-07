# Bot Strategies Guide

This document describes each bot's strategy and tunable parameters.

## Bot Overview

| Bot | Strategy | Difficulty | Best Against |
|-----|----------|------------|--------------|
| **IdleBot** | Passive, mines only | Easy | Testing |
| **RushBot** | Early marine rush | Medium | IdleBot, EconomyBot |
| **DefenseBot** | Turtle with bunkers | Medium | RushBot, ProxyBot |
| **EconomyBot** | Fast expand, macro | Hard | DefenseBot |
| **ProxyBot** | Aggressive proxy | High Risk | EconomyBot, slow bots |

---

## IdleBot

**Strategy**: Do nothing except train SCVs and mine.

**Purpose**: Testing opponent, ensures other bots can win.

**Parameters**: None (intentionally weak)

---

## RushBot

**Strategy**: Fast marine rush with minimal economy.

### Build Order:
1. Train SCVs to 16 workers
2. Build 2 barracks
3. Mass marines
4. Attack at 8+ marines
5. All-in with workers

### Tunable Parameters:
```python
MAX_WORKERS = 16             # Economic cap (start with 12)
BARRACKS_COUNT = 2           # Production buildings
ATTACK_MARINE_THRESHOLD = 8  # Marines before all-in
```

### Strengths:
- Fast timing attack
- Catches greedy bots off-guard
- Simple and reliable

### Weaknesses:
- Weak to bunker defense
- No backup plan if rush fails
- Limited economy

### Good Against: IdleBot, EconomyBot (if timed well)
### Bad Against: DefenseBot, ProxyBot

---

## DefenseBot

**Strategy**: Defensive turtle, mass army, counter-attack.

### Build Order:
1. Train SCVs to 20 workers
2. Build 2 bunkers at defensive position
3. Build 3 barracks
4. Mass marines and load bunkers
5. Defend until 20+ marines
6. Counter-attack with overwhelming force

### Tunable Parameters:
```python
MAX_WORKERS = 20               # More economy than rush
BUNKER_COUNT = 2               # Defensive structures
BARRACKS_COUNT = 3             # Production
ATTACK_MARINE_THRESHOLD = 20   # Large army before push
DEFENSE_RADIUS = 25            # How close enemy must be to trigger defense
```

### Strengths:
- Bunkers hard-counter early rushes
- Can defend and counter-attack
- Strong mid-game army

### Weaknesses:
- Slow to attack
- Vulnerable to fast expand
- Can be out-macroed

### Good Against: RushBot, ProxyBot
### Bad Against: EconomyBot (gets out-macroed)

---

## EconomyBot

**Strategy**: Macro-focused with fast expansion.

### Build Order:
1. Train SCVs to 14 workers
2. Build minimal defense (6 marines)
3. Expand to second base
4. Saturate both bases (32 workers)
5. Build 4 barracks
6. Mass huge army (25+ marines)
7. Overwhelm opponent with numbers

### Tunable Parameters:
```python
MAX_WORKERS_PER_BASE = 16      # Saturation per base
EXPAND_AT_WORKERS = 14         # When to take expansion
BARRACKS_COUNT = 4             # Production once economy is strong
ATTACK_MARINE_THRESHOLD = 25   # Wait for huge army
MIN_MARINES_BEFORE_EXPAND = 6  # Defense before expanding
```

### Strengths:
- Superior economy
- Can replace losses
- Overwhelming late-game army
- Multiple bases = resilience

### Weaknesses:
- Vulnerable early game
- Slow to build up
- Can lose to timing attacks

### Good Against: DefenseBot, late-game bots
### Bad Against: RushBot, ProxyBot (timing attacks)

---

## ProxyBot

**Strategy**: Ultra-aggressive proxy barracks near enemy.

### Build Order:
1. Train to 13 SCVs
2. Send 1 SCV to enemy base (~20 seconds)
3. Build 2 proxy barracks near enemy
4. Build 1 barracks at home (backup)
5. Rush with first 4 marines
6. All-in with workers at 6+ marines
7. Constant reinforcements

### Tunable Parameters:
```python
MAX_WORKERS = 12             # Minimal economy
PROXY_BARRACKS_COUNT = 2     # Barracks near enemy
HOME_BARRACKS_COUNT = 1      # Backup production
ATTACK_MARINE_THRESHOLD = 4  # Attack immediately
PROXY_DISTANCE = 30          # How close to build proxy
```

### Strengths:
- Extremely fast attack timing
- Marines arrive before enemy is ready
- Short reinforcement distance
- High pressure, no downtime

### Weaknesses:
- All-in (if it fails, lose)
- Proxy can be scouted and killed
- Minimal economy (can't recover)
- High execution risk

### Good Against: EconomyBot, greedy builds
### Bad Against: DefenseBot (bunkers hard-counter), early scouts

---

## Parameter Tuning Tips

### For RushBot:
- **Lower ATTACK_THRESHOLD (6-8)**: Faster attack, more risk
- **Higher ATTACK_THRESHOLD (10-12)**: Safer, but slower
- **More BARRACKS (3-4)**: Faster army, more cost
- **More WORKERS (18-20)**: Better economy, slower attack

### For DefenseBot:
- **More BUNKERS (3)**: Stronger defense, more cost
- **Lower ATTACK_THRESHOLD (15)**: Earlier counter-attack
- **More BARRACKS (4-5)**: Faster army production

### For EconomyBot:
- **Earlier EXPAND (12 workers)**: Riskier but faster economy
- **More MIN_MARINES (8-10)**: Safer expand, slower
- **More BARRACKS (5-6)**: Leverage superior economy

### For ProxyBot:
- **Closer PROXY (20-25)**: Faster arrival, easier to scout
- **Farther PROXY (35-40)**: Harder to scout, slower arrival

---

## Matchup Guide

### RushBot vs DefenseBot
- DefenseBot favored (bunkers counter rush)
- RushBot needs to attack before bunkers complete
- RushBot can try proxy variation

### RushBot vs EconomyBot
- RushBot favored if attack is fast enough
- EconomyBot wins if it survives to late game
- Critical timing: before expansion pays off

### DefenseBot vs EconomyBot
- EconomyBot favored (out-macros DefenseBot)
- DefenseBot needs earlier attack
- EconomyBot's 2-base economy > DefenseBot's 1-base

### ProxyBot vs EconomyBot
- ProxyBot heavily favored (punishes greed)
- EconomyBot expansion timing too slow
- Marines arrive before defense is ready

### ProxyBot vs DefenseBot
- DefenseBot heavily favored (bunkers counter proxy)
- Proxy marines can't break bunkers
- ProxyBot needs to scout and adapt

### ProxyBot vs RushBot
- Coin flip (both all-in)
- Whoever attacks first usually wins
- Proxy has slight edge (faster timing)

---

## Next Steps

1. **Run round-robin tournament** to see matchup win rates:
   ```bash
   python scripts/run_roundrobin.py -n 10
   ```

2. **Optimize parameters** using genetic algorithm:
   ```bash
   pip install -e ".[optimization]"
   python scripts/optimize_params.py --bot RushBot --opponent DefenseBot --generations 10
   ```

3. **Experiment with variations**:
   - Edit bot files directly
   - Create new bots by copying and modifying
   - Test specific matchups

4. **Prepare for RL (Phase 2)**:
   - These bots serve as training opponents
   - Diverse strategies help RL agent generalize
   - Optimized bots provide stronger challenge
