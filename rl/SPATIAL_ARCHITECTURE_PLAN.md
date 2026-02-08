# Spatial RL Bot - AlphaStar-Level Architecture

## Implementation Plan

### Phase 3: Full Spatial Intelligence

Building a world-class SC2 bot with:
- ✅ Spatial observations (feature maps)
- ✅ Spatial actions (WHERE to build/attack)
- ✅ CNN policy (convolutional processing)
- ✅ Unit-level control (micro)
- ✅ Proper multi-headed action space

---

## Architecture Overview

```
Game State → Feature Extraction → CNN Encoder → Multiple Action Heads
                                                    ├─ Action Type (discrete)
                                                    ├─ Screen Location (spatial)
                                                    ├─ Minimap Location (spatial)
                                                    └─ Unit Selection (discrete)
```

---

## 1. Spatial Observations

### Feature Maps (Screen: 64x64)

**Channels (20 total):**
```python
Screen Features [64x64 x 20]:
  0. Player relative (self=1, enemy=2, neutral=3)
  1. Unit type (encoded)
  2. Selected units
  3. Unit hit points (ratio)
  4. Unit shields (ratio)
  5. Unit energy (ratio)
  6. Unit density
  7. Friendly unit density
  8. Enemy unit density
  9. Height map
 10. Visibility (visible/hidden)
 11. Creep
 12. Buildable
 13. Pathable
 14. Unit hit points (normalized)
 15. Unit shields (normalized)
 16. Unit energy (normalized)
 17. Selected units (density)
 18. Cargo
 19. Cargo size
```

### Feature Maps (Minimap: 64x64)

**Channels (11 total):**
```python
Minimap Features [64x64 x 11]:
  0. Height map
  1. Visibility
  2. Creep
  3. Camera position
  4. Player relative
  5. Player ID
  6. Selected units
  7. Unit type
  8. Alerts
  9. Buildable
 10. Pathable
```

### Scalar Features

```python
Scalars [100+ features]:
  - Player info (minerals, gas, supply, etc.)
  - Unit counts by type
  - Building counts by type
  - Upgrade status
  - Available actions
  - Game time
  - etc.
```

---

## 2. Spatial Action Space

### Action Components

```python
Action = {
    'function': int,        # What to do (0-573 in full SC2 API)
    'screen': (x, y),      # WHERE on screen (0-63, 0-63)
    'minimap': (x, y),     # WHERE on minimap (0-63, 0-63)
    'queued': bool,        # Add to unit queue?
    'select_unit': int,    # Which unit to control (0-N)
    'select_add': bool,    # Add to selection?
    'control_group': int,  # Control group (0-9)
    'select_point': (x,y), # Point selection
    'select_rect': (x1,y1,x2,y2)  # Rectangle selection
}
```

### Simplified Action Set (for training)

**We'll start with ~50 key actions:**
```python
Simplified Actions:
  0. No-op
  1. Select army
  2. Select all marines
  3. Select point (requires screen x,y)
  4. Move to location (requires screen/minimap x,y)
  5. Attack location (requires screen/minimap x,y)
  6. Train SCV
  7. Build supply depot (requires screen x,y)
  8. Build barracks (requires screen x,y)
  9. Build refinery (on geyser, requires screen x,y)
 10. Build factory (requires screen x,y)
 11. Train marine
 12. Train marauder
 13. Research stim
 14. Use stim
 15. Siege tank
 16. Unsiege tank
 ... (expand to ~50 actions)
```

---

## 3. CNN Policy Architecture

### Network Structure

```python
class SpatialPolicy(nn.Module):
    """
    AlphaStar-inspired CNN policy for SC2.

    Architecture:
      Screen + Minimap → CNN Encoders → Concatenate → LSTM → Action Heads
    """

    def __init__(self):
        # Screen encoder (64x64x20 → 256)
        self.screen_encoder = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=8, stride=4),  # → 15x15x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → 6x6x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → 4x4x64
            nn.ReLU(),
            nn.Flatten(),  # → 1024
            nn.Linear(1024, 256)
        )

        # Minimap encoder (64x64x11 → 128)
        self.minimap_encoder = nn.Sequential(
            nn.Conv2d(11, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(480, 128)
        )

        # Scalar encoder (100+ → 64)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(num_scalars, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Combine all features (256 + 128 + 64 = 448)
        self.combined_size = 256 + 128 + 64

        # LSTM for temporal reasoning
        self.lstm = nn.LSTM(self.combined_size, 256, num_layers=1)
        self.lstm_hidden = None

        # Action heads
        self.action_type_head = nn.Linear(256, num_actions)

        # Spatial heads (output 64x64 heatmaps)
        self.screen_location_head = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Unflatten(1, (16, 8, 8)),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2),  # → 18x18
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2),   # → 38x38
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),  # → 64x64
        )

        self.minimap_location_head = similar_to_screen()

        # Value head
        self.value_head = nn.Linear(256, 1)

    def forward(self, screen, minimap, scalars, lstm_state=None):
        # Encode observations
        screen_features = self.screen_encoder(screen)
        minimap_features = self.minimap_encoder(minimap)
        scalar_features = self.scalar_encoder(scalars)

        # Combine
        combined = torch.cat([screen_features, minimap_features, scalar_features], dim=-1)

        # LSTM for temporal context
        lstm_out, lstm_state = self.lstm(combined.unsqueeze(0), lstm_state)
        features = lstm_out.squeeze(0)

        # Action heads
        action_logits = self.action_type_head(features)
        screen_heatmap = self.screen_location_head(features).squeeze(1)  # 64x64
        minimap_heatmap = self.minimap_location_head(features).squeeze(1)  # 64x64
        value = self.value_head(features)

        return {
            'action_logits': action_logits,
            'screen_heatmap': screen_heatmap,
            'minimap_heatmap': minimap_heatmap,
            'value': value,
            'lstm_state': lstm_state
        }
```

---

## 4. Implementation Steps

### Step 1: Feature Extraction ✅
- Create `spatial_features.py` to convert game state → feature maps
- Implement screen, minimap, and scalar extraction
- Test feature visualization

### Step 2: Spatial Environment ✅
- Create `spatial_env.py` extending SC2Env
- Implement spatial observation space
- Implement spatial action space
- Handle action masking (invalid actions)

### Step 3: CNN Policy ✅
- Create `spatial_policy.py` with PyTorch model
- Implement screen/minimap encoders
- Implement spatial decoders (heatmaps)
- Add LSTM for temporal reasoning

### Step 4: Custom PPO ✅
- Modify PPO to handle spatial actions
- Multi-headed loss function
- Action masking during training
- Entropy bonuses for exploration

### Step 5: Training Script ✅
- Create `train_spatial.py`
- Curriculum learning (start simple, increase complexity)
- Reward shaping for spatial actions
- TensorBoard logging for spatial outputs

### Step 6: Testing & Iteration ✅
- Visualize learned policies (heatmaps)
- Test against scripted bots
- Debug and iterate

---

## 5. Challenges & Solutions

### Challenge 1: Huge Action Space
**Problem:** 50 action types × 64×64 screen × 64×64 minimap = massive space

**Solution:** Action masking - only allow valid actions per state

### Challenge 2: Sparse Spatial Rewards
**Problem:** Hard to learn WHERE to build without dense feedback

**Solution:**
- Reward good building placement explicitly
- Imitation learning from pro replays (spatial)
- Shaped rewards for positioning

### Challenge 3: Training Time
**Problem:** Spatial processing is slow, training takes longer

**Solution:**
- Start with low resolution (32×32, scale up later)
- Parallel environments (vectorized)
- GPU acceleration for CNN

### Challenge 4: Action Coordination
**Problem:** Need to select action type AND location together

**Solution:** Hierarchical action sampling:
1. Sample action type
2. If spatial action, sample location from heatmap
3. If unit selection, sample unit ID

---

## 6. Expected Results

### After Training:

**Build Orders:**
- ✅ Optimal building placement (walls, grids)
- ✅ Efficient production flow
- ✅ Proper tech timings

**Micro:**
- ✅ Focus fire (target selection)
- ✅ Kiting and splitting
- ✅ Retreating damaged units
- ✅ Multi-prong attacks

**Macro:**
- ✅ Map awareness
- ✅ Expansion timing
- ✅ Resource management

**Strategic:**
- ✅ Scouting
- ✅ Harassment
- ✅ Positioning
- ✅ Engagement decisions

---

## 7. Timeline

**Week 1:** Feature extraction + spatial environment
**Week 2:** CNN policy + custom PPO
**Week 3:** Training infrastructure + curriculum
**Week 4:** Initial training + debugging
**Week 5-8:** Iteration and improvement

**Total: ~2 months to world-class bot**

---

## Let's Build It! 🚀

Starting with feature extraction...
