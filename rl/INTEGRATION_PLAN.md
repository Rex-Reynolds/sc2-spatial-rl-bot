# Integration Plan - Action Masking & Spatial Rewards

## 🎯 What Needs Integration

### 1. **Action Masking** (High Priority)
**Status:** Code written (`action_masking.py`) but not integrated

**Integration Steps:**

#### A. Update `spatial_bot.py` to provide masks:
```python
# In SpatialRLBot._get_spatial_observation()
from rl.action_masking import get_available_actions

def _get_spatial_observation(self) -> Dict[str, np.ndarray]:
    features = self.feature_extractor.extract_features(self)

    # ADD: Get action mask
    action_mask = get_available_actions(self)
    features['action_mask'] = action_mask

    return features
```

#### B. Update `spatial_env.py` observation space:
```python
# Add to __init__
self.observation_space = spaces.Dict({
    'screen': spaces.Box(0, 1, (20, 64, 64), dtype=np.float32),
    'minimap': spaces.Box(0, 1, (11, 64, 64), dtype=np.float32),
    'scalars': spaces.Box(0, 1, (90,), dtype=np.float32),
    'action_mask': spaces.Box(0, 1, (50,), dtype=np.float32),  # NEW
})
```

#### C. Update `spatial_policy.py` to use masks:
```python
# In SpatialActorCriticPolicy.get_action_and_value()
def get_action_and_value(self, observations, ...):
    outputs = self.forward(observations, lstm_states)

    action_type_logits = outputs['action_type_logits']

    # APPLY ACTION MASK
    if 'action_mask' in observations:
        mask = observations['action_mask']
        # Set invalid actions to -inf
        action_type_logits = torch.where(
            mask > 0.5,
            action_type_logits,
            torch.tensor(-1e9).to(action_type_logits.device)
        )

    # Rest of sampling...
```

#### D. Update `train_spatial.py`:
```python
# In convert_obs_to_torch()
def convert_obs_to_torch(obs_dict, device):
    return {
        'screen': torch.from_numpy(obs_dict['screen']).unsqueeze(0).to(device),
        'minimap': torch.from_numpy(obs_dict['minimap']).unsqueeze(0).to(device),
        'scalars': torch.from_numpy(obs_dict['scalars']).unsqueeze(0).to(device),
        'action_mask': torch.from_numpy(obs_dict['action_mask']).unsqueeze(0).to(device),  # NEW
    }
```

**Expected Impact:** 30-40% faster learning, cleaner policies

---

### 2. **Spatial Rewards** (Medium Priority)
**Status:** Code written (`spatial_rewards.py`) but not integrated

**Integration Steps:**

#### A. Update `spatial_bot.py` to use new reward calculator:
```python
# In __init__
from rl.spatial_rewards import SpatialRewardCalculator

def __init__(self, env, player_id=1, policy=None):
    super().__init__()
    # ... existing code ...

    # ADD: Use spatial reward calculator
    self.reward_calculator = SpatialRewardCalculator()

# In _calculate_reward()
def _calculate_reward(self) -> float:
    # REPLACE simple reward calculation with:
    return self.reward_calculator.calculate_reward(self)

# In on_start() or reset
async def on_start(self):
    self.client.game_step = 4
    self.reward_calculator.reset()  # Reset for new episode
```

**Expected Impact:** 2-3x faster learning, better strategies

---

## 📅 Implementation Timeline

### **Tomorrow Morning (2 hours)**

**Task 1: Integrate Action Masking**
- Update spatial_bot.py (30 min)
- Update spatial_env.py (15 min)
- Update spatial_policy.py (30 min)
- Update train_spatial.py (15 min)
- Test with 1 episode (30 min)

**Task 2: Integrate Spatial Rewards**
- Update spatial_bot.py (15 min)
- Test with 1 episode (15 min)

**Task 3: Validation Run**
```bash
# Test integrated version
python rl/train_spatial.py \
    --opponent IdleBot \
    --episodes 10 \
    --model-name spatial_integrated_test \
    --use-lstm
```

**Expected:** Same win rate but faster learning (fewer episodes needed)

---

## 🧪 Testing Plan

### **A/B Comparison:**

**Test 1: Without Integration** (baseline)
- Use current spatial bot
- 20 episodes vs IdleBot
- Measure: episodes to 90% win rate

**Test 2: With Action Masking Only**
- Integrated action masking
- 20 episodes vs IdleBot
- Measure: episodes to 90% win rate
- Expected: 30% faster

**Test 3: With Action Masking + Spatial Rewards**
- Both integrated
- 20 episodes vs IdleBot
- Measure: episodes to 90% win rate
- Expected: 50% faster

**Test 4: Full Integration vs Curriculum**
- Best configuration
- Full curriculum (IdleBot → RushBot → Macro → Self-play)
- Measure: final win rates

---

## 📊 Success Metrics

### **After Integration:**
- ✅ No invalid actions attempted (check logs)
- ✅ Reward variance decreases (more stable)
- ✅ Learning curve steeper (faster convergence)
- ✅ Win rate reaches 90% in fewer episodes

### **Red Flags:**
- ⚠️ Action masking too restrictive (all actions masked)
- ⚠️ Rewards exploding (bad normalization)
- ⚠️ Training unstable (oscillating losses)

---

## 🚀 After Integration

Once integrated and tested:

1. **Re-run Population Training**
   - With action masking + spatial rewards
   - Find optimal hyperparameters for integrated version
   - Expected: Even better performance

2. **Re-run Curriculum**
   - With all improvements
   - Expected: Reach Diamond level faster

3. **Advanced Features**
   - Proceed to next phase (see below)
