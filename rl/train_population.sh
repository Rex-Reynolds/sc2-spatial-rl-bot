#!/bin/bash
# Population-Based Training
# Train multiple agents in parallel with different hyperparameters

set -e

echo "======================================================================"
echo "POPULATION-BASED TRAINING"
echo "======================================================================"
echo ""
echo "This will train 4 agents in parallel with different hyperparameters:"
echo "  Agent 1: Baseline (lr=3e-4, batch=64)"
echo "  Agent 2: Fast learner (lr=1e-3, batch=128)"
echo "  Agent 3: Stable learner (lr=1e-4, batch=32)"
echo "  Agent 4: Long-term planner (gamma=0.995)"
echo ""
echo "Each agent: 50 episodes (~4-5 hours per agent)"
echo "Total time: ~4-5 hours (parallel execution)"
echo ""
read -p "Press Enter to start population training..."

# Create logs directory
mkdir -p rl/population_logs

# Agent 1: Baseline
echo ""
echo "🤖 Starting Agent 1: Baseline"
python rl/train_spatial.py \
    --opponent IdleBot \
    --episodes 50 \
    --model-name population_agent1_baseline \
    --learning-rate 0.0003 \
    --ppo-epochs 4 \
    --use-lstm \
    > rl/population_logs/agent1.log 2>&1 &
PID1=$!
echo "   PID: $PID1"

# Agent 2: Fast learner
echo "🤖 Starting Agent 2: Fast Learner"
python rl/train_spatial.py \
    --opponent IdleBot \
    --episodes 50 \
    --model-name population_agent2_fast \
    --learning-rate 0.001 \
    --step-interval 8 \
    --ppo-epochs 4 \
    --use-lstm \
    > rl/population_logs/agent2.log 2>&1 &
PID2=$!
echo "   PID: $PID2"

# Agent 3: Stable learner
echo "🤖 Starting Agent 3: Stable Learner"
python rl/train_spatial.py \
    --opponent IdleBot \
    --episodes 50 \
    --model-name population_agent3_stable \
    --learning-rate 0.0001 \
    --ppo-epochs 6 \
    --use-lstm \
    > rl/population_logs/agent3.log 2>&1 &
PID3=$!
echo "   PID: $PID3"

# Agent 4: Long-term planner
echo "🤖 Starting Agent 4: Long-term Planner"
python rl/train_spatial.py \
    --opponent IdleBot \
    --episodes 50 \
    --model-name population_agent4_longterm \
    --learning-rate 0.0003 \
    --gamma 0.995 \
    --ppo-epochs 4 \
    --use-lstm \
    > rl/population_logs/agent4.log 2>&1 &
PID4=$!
echo "   PID: $PID4"

echo ""
echo "======================================================================"
echo "ALL AGENTS STARTED!"
echo "======================================================================"
echo ""
echo "Monitor progress:"
echo "  tail -f rl/population_logs/agent1.log"
echo "  tail -f rl/population_logs/agent2.log"
echo "  tail -f rl/population_logs/agent3.log"
echo "  tail -f rl/population_logs/agent4.log"
echo ""
echo "Check status:"
echo "  ps -p $PID1 $PID2 $PID3 $PID4"
echo ""
echo "Waiting for all agents to complete..."
echo ""

# Wait for all agents
wait $PID1
echo "✓ Agent 1 complete"

wait $PID2
echo "✓ Agent 2 complete"

wait $PID3
echo "✓ Agent 3 complete"

wait $PID4
echo "✓ Agent 4 complete"

echo ""
echo "======================================================================"
echo "POPULATION TRAINING COMPLETE!"
echo "======================================================================"
echo ""
echo "Models saved:"
echo "  rl/models/population_agent1_baseline/"
echo "  rl/models/population_agent2_fast/"
echo "  rl/models/population_agent3_stable/"
echo "  rl/models/population_agent4_longterm/"
echo ""
echo "Next steps:"
echo "  1. Compare performance with TensorBoard:"
echo "     tensorboard --logdir=rl/logs/"
echo ""
echo "  2. Identify best performer"
echo ""
echo "  3. Use best hyperparameters for final training"
echo ""
echo "  4. Optional: Tournament between agents"
echo ""
