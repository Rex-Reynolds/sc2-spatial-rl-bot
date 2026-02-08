# Visualization & Analysis Tools

## 🎯 Goal: Understand What the Bot Learned

### 1. **Feature Map Visualization**
**Purpose:** See what the CNN "sees"

```python
# Create: rl/visualize_features.py
import matplotlib.pyplot as plt
import numpy as np

def visualize_screen_features(obs_dict):
    """Show all 20 screen channels as images."""
    screen = obs_dict['screen']  # (20, 64, 64)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    channel_names = [
        "Player Relative", "Unit Type", "Selected", "HP",
        "Shields", "Energy", "Unit Density", "Friendly Density",
        "Enemy Density", "Height", "Visibility", "Creep",
        "Buildable", "Pathable", "HP Absolute", "Shields Absolute",
        "Energy Absolute", "Selected Density", "Cargo", "Cargo Size"
    ]

    for i in range(20):
        ax = axes[i // 5, i % 5]
        ax.imshow(screen[i], cmap='viridis')
        ax.set_title(channel_names[i], fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('feature_maps.png')
    plt.show()
```

**Use:**
```bash
python rl/visualize_features.py --model <MODEL> --episode 1
```

---

### 2. **Action Heatmaps**
**Purpose:** See WHERE the bot decides to act

```python
# Create: rl/visualize_actions.py
def create_action_heatmap(episode_data):
    """Show spatial action distribution."""
    screen_actions = []

    for obs, action, reward, done, info in episode_data:
        screen_idx = action['screen_idx']
        x = screen_idx % 64
        y = screen_idx // 64
        screen_actions.append((x, y))

    # Create heatmap
    heatmap = np.zeros((64, 64))
    for x, y in screen_actions:
        heatmap[y, x] += 1

    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Action Frequency')
    plt.title('Spatial Action Distribution')
    plt.savefig('action_heatmap.png')
```

---

### 3. **Strategy Analysis**
**Purpose:** Understand learned strategies

```python
# Create: rl/analyze_strategy.py
def analyze_build_order(episode):
    """Extract and visualize build order."""
    timeline = []

    for step, (obs, action, reward, done, info) in enumerate(episode):
        action_type = action['action_type']
        action_name = ACTION_NAMES[action_type]
        game_time = obs['scalars'][TIME_INDEX] * 1800  # Denormalize

        if 'build' in action_name or 'train' in action_name:
            timeline.append({
                'time': game_time,
                'action': action_name,
                'resources': obs['scalars'][MINERALS_INDEX] * 5000
            })

    # Plot timeline
    times = [t['time'] for t in timeline]
    actions = [t['action'] for t in timeline]

    plt.figure(figsize=(12, 6))
    plt.scatter(times, range(len(actions)), s=100)
    plt.yticks(range(len(actions)), actions)
    plt.xlabel('Game Time (seconds)')
    plt.title('Build Order Timeline')
    plt.grid(True, alpha=0.3)
    plt.savefig('build_order.png')
```

---

### 4. **Performance Metrics Dashboard**

```python
# Create: rl/dashboard.py
import streamlit as st
import pandas as pd

def create_dashboard():
    """Interactive dashboard for model analysis."""

    st.title("SC2 Bot Performance Dashboard")

    # Load training data
    models = ["baseline", "integrated", "curriculum_final"]

    # Win rate over time
    st.subheader("Win Rate Progression")
    win_rates = load_win_rates(models)
    st.line_chart(win_rates)

    # Action distribution
    st.subheader("Action Distribution")
    action_counts = load_action_counts(selected_model)
    st.bar_chart(action_counts)

    # Spatial heatmaps
    st.subheader("Spatial Action Heatmap")
    heatmap = load_action_heatmap(selected_model)
    st.image(heatmap)

    # Build order analysis
    st.subheader("Average Build Order")
    build_order = analyze_build_orders(selected_model)
    st.dataframe(build_order)

# Run:
# streamlit run rl/dashboard.py
```

---

## 📊 Analysis Goals

### **What to Look For:**

1. **Feature Maps**
   - Are enemy units being detected?
   - Is terrain information captured?
   - Is fog of war represented?

2. **Action Heatmaps**
   - Are actions concentrated in smart locations?
   - Building placement patterns?
   - Army positioning strategies?

3. **Build Orders**
   - Consistent openings?
   - Optimal timings?
   - Adapting to opponents?

4. **Spatial Patterns**
   - Wall-ins at natural?
   - Organized base layouts?
   - Strategic positioning?

---

## 🎯 Expected Insights

After visualization, you should be able to answer:
- ✅ What does the bot "see"?
- ✅ Where does it prefer to build?
- ✅ What's its go-to strategy?
- ✅ Is it learning spatial patterns?
- ✅ How does it differ from scripted bots?

---

## 🚀 Implementation Priority

**High Priority (Do First):**
1. Action heatmaps (easiest, most informative)
2. Build order timeline (understand strategy)

**Medium Priority:**
3. Feature map visualization (debugging)
4. Strategy analysis (deeper understanding)

**Low Priority (Nice to Have):**
5. Interactive dashboard (polish)

**Time: ~1-2 days to implement all**
