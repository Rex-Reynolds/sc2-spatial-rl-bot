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
from typing import Optional, List
import uvicorn
from datetime import datetime

from sc2 import maps, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.data import Difficulty, AIBuild

# Import our bot components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.play_vs_bot import PlayableRLBot, load_trained_model

app = FastAPI(title="SC2 Spatial RL Bot API", version="1.0.0")

# Global state
current_bot = None
current_model_path = None
game_results = []
game_in_progress = False


class GameRequest(BaseModel):
    opponent: str = "Medium"  # "Easy", "Medium", "Hard", "VeryHard"
    map_name: str = "Simple64"
    realtime: bool = False


class BotStatus(BaseModel):
    loaded: bool
    model_path: Optional[str]
    games_played: int
    wins: int
    losses: int
    win_rate: float
    game_in_progress: bool


class GameResult(BaseModel):
    game_id: int
    opponent: str
    map_name: str
    result: str  # "Win", "Loss"
    timestamp: str


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SC2 Spatial RL Bot API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This help message",
            "POST /load_model": "Load a trained model",
            "POST /play_game": "Start a new game",
            "GET /status": "Get bot status",
            "GET /results": "Get game results history",
            "GET /results/{game_id}": "Get specific game result"
        },
        "docs": "/docs"
    }


@app.post("/load_model")
async def load_model(model_path: str):
    """Load a trained model."""
    global current_bot, current_model_path

    if game_in_progress:
        raise HTTPException(status_code=400, detail="Game in progress, cannot load model")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    try:
        print(f"Loading model: {model_path}")
        policy = load_trained_model(model_path)
        current_bot = PlayableRLBot(policy)
        current_model_path = model_path
        print("✓ Model loaded successfully!")
        return {
            "status": "success",
            "model_path": model_path,
            "message": "Model loaded successfully"
        }
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/play_game")
async def play_game(request: GameRequest):
    """Start a new game."""
    global current_bot, game_results, game_in_progress

    if current_bot is None:
        raise HTTPException(
            status_code=400,
            detail="No model loaded. Use POST /load_model first."
        )

    if game_in_progress:
        raise HTTPException(
            status_code=400,
            detail="Game already in progress"
        )

    # Map opponent string to difficulty
    difficulty_map = {
        "VeryEasy": Difficulty.VeryEasy,
        "Easy": Difficulty.Easy,
        "Medium": Difficulty.Medium,
        "MediumHard": Difficulty.MediumHard,
        "Hard": Difficulty.Hard,
        "Harder": Difficulty.Harder,
        "VeryHard": Difficulty.VeryHard,
        "CheatVision": Difficulty.CheatVision,
        "CheatMoney": Difficulty.CheatMoney,
        "CheatInsane": Difficulty.CheatInsane,
    }

    if request.opponent not in difficulty_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid opponent. Must be one of: {', '.join(difficulty_map.keys())}"
        )

    difficulty = difficulty_map[request.opponent]
    game_id = len(game_results)

    # Run game in background thread
    def run_game_sync():
        global game_in_progress
        game_in_progress = True

        try:
            print(f"Starting game #{game_id}: vs {request.opponent} on {request.map_name}")
            result = run_game(
                maps.get(request.map_name),
                [
                    Bot(Race.Terran, current_bot, name="SpatialRLBot"),
                    Computer(Race.Random, difficulty)
                ],
                realtime=request.realtime,
            )

            # Store result
            game_results.append({
                "game_id": game_id,
                "opponent": request.opponent,
                "map_name": request.map_name,
                "result": "Win" if result[0] == 1 else "Loss",
                "timestamp": datetime.now().isoformat()
            })

            result_str = "WIN" if result[0] == 1 else "LOSS"
            print(f"✓ Game #{game_id} finished: {result_str}")

        except Exception as e:
            print(f"✗ Game #{game_id} error: {e}")
            game_results.append({
                "game_id": game_id,
                "opponent": request.opponent,
                "map_name": request.map_name,
                "result": "Error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
        finally:
            game_in_progress = False

    thread = Thread(target=run_game_sync, daemon=True)
    thread.start()

    return {
        "status": "game_started",
        "game_id": game_id,
        "opponent": request.opponent,
        "map": request.map_name,
        "message": f"Game #{game_id} started"
    }


@app.get("/status")
async def get_status() -> BotStatus:
    """Get bot status."""
    wins = sum(1 for r in game_results if r.get("result") == "Win")
    losses = sum(1 for r in game_results if r.get("result") == "Loss")
    total = len(game_results)
    win_rate = wins / total if total > 0 else 0.0

    return BotStatus(
        loaded=current_bot is not None,
        model_path=current_model_path,
        games_played=total,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        game_in_progress=game_in_progress
    )


@app.get("/results")
async def get_results(limit: int = 10):
    """Get game results history."""
    wins = sum(1 for r in game_results if r.get("result") == "Win")
    losses = sum(1 for r in game_results if r.get("result") == "Loss")
    total = len(game_results)

    return {
        "total_games": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total if total > 0 else 0.0,
        "recent_games": game_results[-limit:] if game_results else []
    }


@app.get("/results/{game_id}")
async def get_game_result(game_id: int):
    """Get specific game result."""
    if game_id < 0 or game_id >= len(game_results):
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    return game_results[game_id]


@app.delete("/results")
async def clear_results():
    """Clear all game results."""
    global game_results

    if game_in_progress:
        raise HTTPException(status_code=400, detail="Cannot clear results while game in progress")

    count = len(game_results)
    game_results = []

    return {
        "status": "success",
        "message": f"Cleared {count} game results"
    }


if __name__ == "__main__":
    print("=" * 70)
    print("SC2 SPATIAL RL BOT - API SERVER")
    print("=" * 70)
    print("")
    print("API Endpoints:")
    print("  POST /load_model?model_path=<path>  - Load trained model")
    print("  POST /play_game                      - Start new game")
    print("  GET  /status                         - Get bot status")
    print("  GET  /results                        - Get game results")
    print("  GET  /results/{game_id}              - Get specific result")
    print("  DELETE /results                      - Clear all results")
    print("")
    print("Interactive API docs: http://localhost:8000/docs")
    print("=" * 70)
    print("")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
