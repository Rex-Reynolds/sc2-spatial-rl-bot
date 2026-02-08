# Training Against Professional StarCraft II Replays

## Overview

We can create a bot trained on professional Terran replays to serve as a strong opponent for RL training. This uses **Imitation Learning** (Behavioral Cloning).

## Approach 1: Imitation Learning Bot

### Step 1: Collect Professional Replays

```bash
# Download pro replays from spawningtool.com
# Example sources:
# - https://lotv.spawningtool.com/replays/
# - Filter for: Terran players, high MMR (6000+), recent patches
```

Replay sources:
- **spawningtool.com** - Curated pro replays with build orders
- **sc2replaystats.com** - Large replay database
- **AlphaStar replay pack** - DeepMind's training data (if available)

### Step 2: Parse Replays Into Training Data

Use `sc2reader` library to extract game states and actions:

```python
import sc2reader

def parse_replay(replay_path):
    """Extract observations and actions from a replay."""
    replay = sc2reader.load_replay(replay_path)

    # Find Terran player
    terran_player = None
    for player in replay.players:
        if player.play_race == 'Terran':
            terran_player = player
            break

    if not terran_player:
        return None

    # Extract game events
    trajectories = []
    for event in replay.events:
        if event.player == terran_player:
            # Extract game state at this moment
            obs = extract_observation(replay, event.frame)

            # Map event to our action space
            action = map_event_to_action(event)

            if action is not None:
                trajectories.append((obs, action))

    return trajectories
```

### Step 3: Train Imitation Learning Model

```python
from stable_baselines3 import PPO
from stable_baselines3.common.behavioral_cloning import BehavioralCloning
import numpy as np

# Collect all pro replay data
expert_observations = []
expert_actions = []

for replay_file in pro_replay_files:
    trajectories = parse_replay(replay_file)
    if trajectories:
        for obs, action in trajectories:
            expert_observations.append(obs)
            expert_actions.append(action)

# Convert to numpy arrays
expert_obs = np.array(expert_observations)
expert_actions = np.array(expert_actions)

# Train behavioral cloning model
bc_trainer = BehavioralCloning(
    observation_space=env.observation_space,
    action_space=env.action_space,
    expert_data=(expert_obs, expert_actions),
)

bc_trainer.train(n_epochs=100)

# Save the pro-mimic model
bc_trainer.policy.save("models/pro_terran_clone")
```

### Step 4: Use Pro Bot as RL Training Opponent

```python
# Load the pro-trained model
from stable_baselines3 import PPO

pro_model = PPO.load("models/pro_terran_clone")

def pro_policy(obs):
    """Policy that mimics pro players."""
    return pro_model.predict(obs, deterministic=False)

# Train against the pro bot
env = make_env(
    opponent="SelfPlay",
    opponent_policy=pro_policy,
    advanced=True
)

# Train your RL agent
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=1000000)
```

## Approach 2: Replay-Guided Curriculum

Instead of full imitation, use pro replays to guide training:

1. **Extract "milestone" states** from pro replays:
   - 5-min mark: "Should have 40 SCVs, 5 barracks, 20 marines"
   - 10-min mark: "Should have 3 bases, +1/+1 upgrades, 60 supply army"

2. **Reward shaping based on pro patterns**:
   ```python
   def calculate_reward_with_pro_guidance(self):
       reward = base_reward

       # Bonus for matching pro benchmarks
       if self.time >= 300:  # 5 minutes
           if self.supply_workers >= 40:
               reward += 1.0  # Pro-level economy

       # Penalty for deviating too far from pro timings
       if self.time >= 180 and self.supply_army < 10:
           reward -= 0.5  # Should have army by 3 minutes

       return reward
   ```

3. **Use pro replays as starting positions**:
   - Load game state from replay at minute 5
   - Let RL agent take over from there
   - Learns "late game" without grinding early game

## Implementation Plan

### Phase 1: Simple Pro Bot (Scripted)
**Easiest** - Create a scripted bot that follows a pro build order:
- Marine/Medivac drop timing attack
- Barracks/Factory/Starport opening
- Hardcoded build order from a specific pro replay

```bash
# Create ProTerranBot (manually coded)
python rl/train.py --advanced --opponent ProTerranBot --episodes 100
```

### Phase 2: Behavioral Cloning Bot
**Medium difficulty** - Train on 100+ pro replays:
1. Download 100 Terran pro replays
2. Parse into (observation, action) pairs
3. Train behavioral cloning model
4. Use as opponent

### Phase 3: Replay-Augmented RL
**Advanced** - Combine RL with replay data:
- Pre-train on pro replays (warm start)
- Fine-tune with RL against various opponents
- Use replay states as curriculum checkpoints

## Challenges

1. **Action Space Mismatch**:
   - Pro replays have hundreds of low-level actions (move each unit)
   - Our bot has 23 high-level actions (train marine, attack)
   - Need to **aggregate** replay actions into our action space

2. **Observation Extraction**:
   - Replays don't directly give us our 26-feature observation vector
   - Need to reconstruct game state at each frame
   - sc2reader can help but requires careful parsing

3. **Data Quality**:
   - Need many replays (100+) for good coverage
   - Must filter for: correct race, high skill, recent patch
   - Different pro styles might conflict (aggressive vs macro)

## Next Steps

**Want me to implement:**
1. ✅ Simple pro-style scripted bot (fastest to test)
2. 🔧 Replay parser + behavioral cloning pipeline
3. 📊 Replay analysis tool (show pro benchmarks)

Which approach interests you most?
