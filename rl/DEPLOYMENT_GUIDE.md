# Deployment Guide: SC2 Spatial RL Bot

## 🎯 Overview

Your trained bot can be deployed in 4 main scenarios:

1. **Local Play** - Human vs AI (testing/fun)
2. **Ladder/Tournament** - Competitive deployment
3. **API Server** - Remote control via REST API
4. **Demo/Showcase** - Public demonstrations

---

## 1️⃣ Local Play: Human vs AI

**Purpose:** Play against your trained bot to test its skill level

**File:** `rl/play_vs_bot.py` (already created)

**Usage:**
```bash
# Play against your trained bot
python rl/play_vs_bot.py rl/models/spatial_100ep_extended/final_model.pt

# Or against curriculum-trained bot
python rl/play_vs_bot.py rl/models/curriculum_stage4_selfplay/final_model.pt
```

**What happens:**
- StarCraft 2 launches in real-time mode
- You control Player 1 (Terran)
- AI controls Player 2 (Terran)
- Play normally with mouse/keyboard
- Game ends when someone wins

**Tips:**
- Start with bots trained on fewer episodes to see progression
- Play multiple games to assess consistency
- Watch for specific behaviors (build order, army composition, micro)

---

## 2️⃣ Ladder/Tournament: Competitive Deployment

**Purpose:** Deploy bot to play automatically in tournaments or ladder systems

### **A. SC2AI Ladder Manager**

SC2AI Ladder Manager is the standard tournament infrastructure.

**Setup:**
```bash
# 1. Create standalone bot script
# File: rl/ladder_bot.py
import sys
import torch
from sc2 import run_game, maps, Race
from sc2.player import Bot
from rl.play_vs_bot import PlayableRLBot, load_trained_model

def main():
    # Load your best model
    model_path = "rl/models/curriculum_stage4_selfplay/final_model.pt"
    policy = load_trained_model(model_path)
    bot = PlayableRLBot(policy)

    # Run game (ladder manager handles opponent)
    run_game(
        maps.get(sys.argv[1] if len(sys.argv) > 1 else "Simple64"),
        [Bot(Race.Terran, bot, name="SpatialRLBot")],
        realtime=False,
        save_replay_as="replay.SC2Replay"
    )

if __name__ == "__main__":
    main()
```

**Package for submission:**
```bash
# 2. Create requirements.txt
cat > requirements.txt << EOF
torch>=2.0.0
burnysc2>=6.5.0
numpy>=1.24.0
gymnasium>=0.29.0
EOF

# 3. Create bot metadata
cat > ladderbots.json << EOF
{
    "Bots": {
        "SpatialRLBot": {
            "Race": "Terran",
            "Type": "Python",
            "RootPath": "./",
            "FileName": "ladder_bot.py",
            "Debug": false
        }
    }
}
EOF

# 4. Bundle everything
tar -czf spatial_rl_bot.tar.gz \
    ladder_bot.py \
    ladderbots.json \
    requirements.txt \
    rl/ \
    --exclude=rl/models/*/episode_*.pt \
    --exclude=rl/logs/
```

**Submit to:**
- SC2AI Discord tournaments
- AI Arena (https://aiarena.net)
- Custom ladder competitions

### **B. AI Arena Deployment**

AI Arena is the largest StarCraft II bot competition platform.

**Steps:**
1. Create account at https://aiarena.net
2. Upload `spatial_rl_bot.tar.gz`
3. Configure bot settings:
   - Name: SpatialRLBot
   - Race: Terran
   - Type: Python
   - Entry point: ladder_bot.py
4. Submit to ladder queue
5. Watch replays and track win rate

**Monitoring:**
- Win rate against different skill levels
- APM (Actions Per Minute)
- Common failure modes
- Matchup statistics (TvT, TvP, TvZ)

---

## 3️⃣ API Server: Remote Control

**Purpose:** Run bot as a service that accepts commands via REST API

### **Why API Server?**
- Control bot remotely
- Integrate with web apps
- Run multiple bots on server
- Easier monitoring/logging

### **Implementation:**

**File:** `rl/api_server.py`
```python
#!/usr/bin/env python3
"""
API Server for SC2 Bot
Runs bot as a service, accepts REST API commands
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import asyncio
from threading import Thread
from typing import Optional
import uvicorn

from sc2 import maps, run_game, Race
from sc2.player import Bot, Computer, AIBuild
from rl.play_vs_bot import PlayableRLBot, load_trained_model

app = FastAPI(title="SC2 Spatial RL Bot API")

# Global bot instance
current_bot = None
current_game = None
game_results = []


class GameRequest(BaseModel):
    opponent: str = "Computer"  # "Computer", "Easy", "Medium", "Hard"
    map_name: str = "Simple64"
    realtime: bool = False


class BotStatus(BaseModel):
    loaded: bool
    model_path: Optional[str]
    games_played: int
    win_rate: float


@app.post("/load_model")
async def load_model(model_path: str):
    """Load a trained model."""
    global current_bot
    try:
        policy = load_trained_model(model_path)
        current_bot = PlayableRLBot(policy)
        return {"status": "success", "model": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/play_game")
async def play_game(request: GameRequest):
    """Start a new game."""
    global current_bot, game_results

    if current_bot is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    # Map opponent string to difficulty
    difficulty_map = {
        "Easy": (Computer, Race.Random, AIBuild.Rush),
        "Medium": (Computer, Race.Random, AIBuild.Macro),
        "Hard": (Computer, Race.Random, AIBuild.Timing),
    }

    # Run game in background thread
    def run_game_sync():
        result = run_game(
            maps.get(request.map_name),
            [
                Bot(Race.Terran, current_bot, name="SpatialRLBot"),
                Computer(Race.Random, AIBuild.Rush)
            ],
            realtime=request.realtime,
        )
        game_results.append(result)

    thread = Thread(target=run_game_sync, daemon=True)
    thread.start()

    return {"status": "game_started", "map": request.map_name}


@app.get("/status")
async def get_status() -> BotStatus:
    """Get bot status."""
    wins = sum(1 for r in game_results if r == 1)
    win_rate = wins / len(game_results) if game_results else 0.0

    return BotStatus(
        loaded=current_bot is not None,
        model_path="loaded" if current_bot else None,
        games_played=len(game_results),
        win_rate=win_rate
    )


@app.get("/results")
async def get_results():
    """Get game results history."""
    return {
        "total_games": len(game_results),
        "wins": sum(1 for r in game_results if r == 1),
        "losses": sum(1 for r in game_results if r == 2),
        "results": game_results[-10:]  # Last 10 games
    }


if __name__ == "__main__":
    print("=" * 70)
    print("SC2 SPATIAL RL BOT API SERVER")
    print("=" * 70)
    print("")
    print("Endpoints:")
    print("  POST /load_model?model_path=<path>  - Load trained model")
    print("  POST /play_game                      - Start new game")
    print("  GET  /status                         - Get bot status")
    print("  GET  /results                        - Get game results")
    print("")
    print("API docs: http://localhost:8000/docs")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Usage:**

```bash
# 1. Install FastAPI
pip install fastapi uvicorn pydantic

# 2. Start server
python rl/api_server.py

# Server runs on http://localhost:8000
```

**API Examples:**

```bash
# Load model
curl -X POST "http://localhost:8000/load_model?model_path=rl/models/curriculum_stage4_selfplay/final_model.pt"

# Start game
curl -X POST "http://localhost:8000/play_game" \
  -H "Content-Type: application/json" \
  -d '{"opponent": "Medium", "map_name": "Simple64", "realtime": false}'

# Check status
curl "http://localhost:8000/status"

# Get results
curl "http://localhost:8000/results"
```

**Interactive Docs:**
Visit http://localhost:8000/docs for Swagger UI

**Production Deployment:**
```bash
# Use gunicorn for production
pip install gunicorn

gunicorn rl.api_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## 4️⃣ Demo/Showcase: Public Demonstrations

**Purpose:** Show off your bot to others (streams, videos, presentations)

### **A. Replay Recording**

**Record games automatically:**
```python
# File: rl/record_demo_games.py
import torch
from sc2 import maps, run_game, Race
from sc2.player import Bot, Computer, AIBuild
from rl.play_vs_bot import PlayableRLBot, load_trained_model
import datetime

def record_demo_games(model_path: str, num_games: int = 5):
    """Record demonstration games against various opponents."""

    policy = load_trained_model(model_path)
    bot = PlayableRLBot(policy)

    opponents = [
        ("Easy", AIBuild.Rush),
        ("Medium", AIBuild.Macro),
        ("Hard", AIBuild.Timing),
        ("VeryHard", AIBuild.Air),
    ]

    for i, (difficulty, build) in enumerate(opponents * num_games):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        replay_name = f"demo_{difficulty}_{i}_{timestamp}.SC2Replay"

        print(f"Recording game {i+1}/{num_games*len(opponents)}: vs {difficulty}...")

        result = run_game(
            maps.get("Simple64"),
            [
                Bot(Race.Terran, bot, name="SpatialRLBot"),
                Computer(Race.Random, build)
            ],
            realtime=False,
            save_replay_as=f"replays/{replay_name}"
        )

        print(f"  Result: {'WIN' if result == 1 else 'LOSS'}")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "rl/models/final_model.pt"
    record_demo_games(model_path, num_games=5)
```

**Usage:**
```bash
# Create replays directory
mkdir -p replays

# Record demo games
python rl/record_demo_games.py rl/models/curriculum_stage4_selfplay/final_model.pt

# Replays saved to replays/ directory
```

### **B. Live Streaming Setup**

**Stream bot playing live games:**

```bash
# 1. Use OBS Studio to capture SC2 window
# 2. Run bot in realtime mode
# 3. Add overlays showing:
#    - Current action
#    - Decision confidence
#    - Resource counts
#    - APM
```

**Create overlay script:**
```python
# File: rl/stream_overlay.py
"""
Real-time overlay for streaming
Shows bot's current state and decisions
"""

import tkinter as tk
from threading import Thread
import time

class BotOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Bot Overlay")
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.8)  # Semi-transparent

        # Stats display
        self.stats_label = tk.Label(
            self.root,
            text="Initializing...",
            font=("Courier", 14),
            bg="black",
            fg="lime",
            justify=tk.LEFT
        )
        self.stats_label.pack(padx=10, pady=10)

    def update_stats(self, stats: dict):
        """Update overlay with current bot stats."""
        text = f"""
═══════════════════════════════
🤖 SPATIAL RL BOT
═══════════════════════════════
Time:     {stats['game_time']:6.1f}s
Supply:   {stats['supply_used']:3d} / {stats['supply_cap']:3d}
Minerals: {stats['minerals']:6d}
Gas:      {stats['vespene']:6d}
Workers:  {stats['workers']:3d}
Army:     {stats['army_value']:6d}
APM:      {stats['apm']:6.1f}

Last Action: {stats['last_action']}
Confidence:  {stats['confidence']:5.1f}%
═══════════════════════════════
        """.strip()
        self.stats_label.config(text=text)

    def run(self):
        self.root.mainloop()

# Use in your bot code:
# overlay = BotOverlay()
# overlay.update_stats({...})
```

### **C. Highlight Reel Generator**

**Automatically create best moments:**
```python
# File: rl/create_highlights.py
"""
Extract highlights from replays
- Best micro moments
- Large battles
- Game-winning plays
"""

import sc2reader
from moviepy.editor import VideoFileClip, concatenate_videoclips

def find_battle_timestamps(replay_path: str):
    """Find timestamps where battles occurred."""
    replay = sc2reader.load_replay(replay_path)

    battle_times = []
    for event in replay.events:
        if event.name == "UnitDiedEvent":
            # Cluster deaths within 10 seconds
            battle_times.append(event.second)

    # Cluster close timestamps
    battles = []
    current_battle = []
    for t in sorted(battle_times):
        if not current_battle or t - current_battle[-1] < 10:
            current_battle.append(t)
        else:
            if len(current_battle) > 5:  # Significant battle
                battles.append((min(current_battle), max(current_battle)))
            current_battle = [t]

    return battles

def create_highlight_reel(video_path: str, battles: list):
    """Create highlight video from battle timestamps."""
    video = VideoFileClip(video_path)

    clips = []
    for start, end in battles:
        # Add 5 seconds before and after
        clip_start = max(0, start - 5)
        clip_end = min(video.duration, end + 5)
        clips.append(video.subclip(clip_start, clip_end))

    final = concatenate_videoclips(clips)
    final.write_videofile("highlights.mp4")

# Usage:
# battles = find_battle_timestamps("replays/demo_Hard_0.SC2Replay")
# create_highlight_reel("full_game.mp4", battles)
```

### **D. Web Demo Interface**

**Create interactive web demo:**
```html
<!-- File: web_demo/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>SC2 Spatial RL Bot Demo</title>
    <style>
        body {
            font-family: 'Courier New', monospace;
            background: #1a1a1a;
            color: #00ff00;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .stats-box {
            background: #000;
            border: 2px solid #00ff00;
            padding: 20px;
            margin: 20px 0;
        }
        button {
            background: #00ff00;
            color: #000;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover { background: #00cc00; }
        #replay-list { list-style: none; padding: 0; }
        #replay-list li {
            padding: 10px;
            border-bottom: 1px solid #333;
            cursor: pointer;
        }
        #replay-list li:hover { background: #222; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 SC2 SPATIAL RL BOT - LIVE DEMO</h1>

        <div class="stats-box">
            <h2>Bot Status</h2>
            <div id="status">Loading...</div>
        </div>

        <div class="stats-box">
            <h2>Start New Game</h2>
            <button onclick="startGame('Easy')">vs Easy AI</button>
            <button onclick="startGame('Medium')">vs Medium AI</button>
            <button onclick="startGame('Hard')">vs Hard AI</button>
            <button onclick="startGame('VeryHard')">vs Very Hard AI</button>
        </div>

        <div class="stats-box">
            <h2>Recent Games</h2>
            <div id="results">No games yet...</div>
        </div>

        <div class="stats-box">
            <h2>Watch Replays</h2>
            <ul id="replay-list"></ul>
        </div>
    </div>

    <script>
        // Connect to API server
        const API_URL = 'http://localhost:8000';

        async function startGame(difficulty) {
            const response = await fetch(`${API_URL}/play_game`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    opponent: difficulty,
                    map_name: 'Simple64',
                    realtime: false
                })
            });
            const data = await response.json();
            alert(`Game started! ${data.status}`);
            refreshResults();
        }

        async function refreshStatus() {
            const response = await fetch(`${API_URL}/status`);
            const data = await response.json();
            document.getElementById('status').innerHTML = `
                <strong>Model Loaded:</strong> ${data.loaded ? 'Yes' : 'No'}<br>
                <strong>Games Played:</strong> ${data.games_played}<br>
                <strong>Win Rate:</strong> ${(data.win_rate * 100).toFixed(1)}%
            `;
        }

        async function refreshResults() {
            const response = await fetch(`${API_URL}/results`);
            const data = await response.json();
            document.getElementById('results').innerHTML = `
                <strong>Total Games:</strong> ${data.total_games}<br>
                <strong>Wins:</strong> ${data.wins}<br>
                <strong>Losses:</strong> ${data.losses}<br>
                <strong>Win Rate:</strong> ${(data.wins / data.total_games * 100).toFixed(1)}%
            `;
        }

        // Auto-refresh every 5 seconds
        setInterval(() => {
            refreshStatus();
            refreshResults();
        }, 5000);

        // Initial load
        refreshStatus();
        refreshResults();
    </script>
</body>
</html>
```

**Run web demo:**
```bash
# 1. Start API server
python rl/api_server.py

# 2. Serve web demo
cd web_demo
python -m http.server 8080

# 3. Open browser
# http://localhost:8080
```

---

## 📊 Deployment Checklist

### **Before Deploying:**

- [ ] **Test thoroughly**
  - Play 20+ games against various opponents
  - Check for crashes or errors
  - Verify consistent performance
  - Test on different maps

- [ ] **Optimize model**
  - Remove unnecessary checkpoints
  - Compress model file (optional)
  - Test loading speed
  - Verify memory usage

- [ ] **Document performance**
  - Win rate vs each difficulty
  - Average game length
  - Common strategies used
  - Known weaknesses

- [ ] **Prepare assets**
  - Save final trained model
  - Generate example replays
  - Create gameplay videos
  - Write bot description

### **Deployment-Specific:**

**For Tournaments:**
- [ ] Test ladder manager compatibility
- [ ] Verify all dependencies included
- [ ] Test on clean environment
- [ ] Create backup of model

**For API Server:**
- [ ] Set up monitoring (health checks)
- [ ] Configure logging
- [ ] Set up SSL/HTTPS (if public)
- [ ] Test concurrent requests
- [ ] Document API endpoints

**For Demos:**
- [ ] Record high-quality gameplay
- [ ] Create presentation slides
- [ ] Prepare explanation of approach
- [ ] Set up overlay graphics
- [ ] Test streaming setup

---

## 🎯 What to Expect

### **Local Play Results:**
- Games feel challenging but fair
- Bot makes human-like decisions
- Occasional weird behaviors
- Fun to watch and play against

### **Tournament Performance:**
- Win rate depends on training quality
- Curriculum-trained bots: 60-70% vs Medium AI
- Spatial rewards: Better micro and positioning
- Action masking: Fewer stupid mistakes

### **API Server Usage:**
- ~2-3 games per minute (non-realtime)
- Low latency (<100ms response time)
- Scales to multiple concurrent games
- Easy to integrate with other systems

### **Demo Reception:**
- People impressed by spatial awareness
- Questions about how CNN works
- Interest in training process
- Requests to play against it

---

## 🚀 Next Steps After Deployment

### **Monitor & Improve:**
1. Collect replay data from deployment
2. Analyze failure cases
3. Fine-tune on problematic scenarios
4. Re-deploy improved version

### **Scale Up:**
1. Train on more diverse maps
2. Add more sophisticated strategies
3. Implement multi-race support
4. Enter larger tournaments

### **Research Extensions:**
1. Publish results (blog post, paper)
2. Open source your approach
3. Compare to other bots
4. Contribute to SC2 AI community

---

## 📚 Resources

**Tournament Platforms:**
- AI Arena: https://aiarena.net
- SC2AI Discord: https://discord.gg/sc2ai
- Ladder Manager: https://github.com/Cryptyc/Sc2LadderServer

**Community:**
- SC2 AI subreddit: r/sc2ai
- python-sc2 Discord: https://discord.gg/burnysc2
- TwitchPlaysPokemon SC2 AI streams

**Documentation:**
- python-sc2 docs: https://burnysc2.github.io/python-sc2
- SC2 API docs: https://github.com/Blizzard/s2client-proto

---

## 🎉 Congratulations!

You've built a complete spatial RL bot from scratch. Now go show it to the world! 🚀
