#!/bin/bash
# Run hyperparameter experiments in parallel
# Tests different learning configurations to find optimal settings

set -e

echo "======================================================================"
echo "HYPERPARAMETER TUNING EXPERIMENTS"
echo "======================================================================"
echo "Running 4 experiments in parallel (20 episodes each)"
echo "This will take ~2-3 hours"
echo ""
echo "Experiments:"
echo "  1. Baseline (default hyperparameters)"
echo "  2. Fast learning (high LR, small batch)"
echo "  3. Large batch (more stable gradients)"
echo "  4. Long-term planning (high gamma)"
echo ""
echo "======================================================================"
echo ""

# Create output directory
mkdir -p rl/experiment_logs

# Experiment 1: Baseline (default hyperparameters)
echo "🧪 Starting Experiment 1: Baseline..."
python rl/train.py \
    --advanced \
    --use-improved-rewards \
    --self-play \
    --episodes 20 \
    --model-name exp_baseline \
    > rl/experiment_logs/exp_baseline.log 2>&1 &
PID1=$!
echo "   PID: $PID1"

# Experiment 2: Fast learning
echo "🧪 Starting Experiment 2: Fast Learning..."
python rl/train.py \
    --advanced \
    --use-improved-rewards \
    --self-play \
    --episodes 20 \
    --model-name exp_fast_learning \
    --learning-rate 0.001 \
    --n-steps 512 \
    --batch-size 128 \
    > rl/experiment_logs/exp_fast_learning.log 2>&1 &
PID2=$!
echo "   PID: $PID2"

# Experiment 3: Large batch (more stable)
echo "🧪 Starting Experiment 3: Large Batch..."
python rl/train.py \
    --advanced \
    --use-improved-rewards \
    --self-play \
    --episodes 20 \
    --model-name exp_large_batch \
    --batch-size 256 \
    --n-steps 4096 \
    > rl/experiment_logs/exp_large_batch.log 2>&1 &
PID3=$!
echo "   PID: $PID3"

# Experiment 4: Long-term planning
echo "🧪 Starting Experiment 4: Long-term Planning..."
python rl/train.py \
    --advanced \
    --use-improved-rewards \
    --self-play \
    --episodes 20 \
    --model-name exp_long_term \
    --gamma 0.995 \
    > rl/experiment_logs/exp_long_term.log 2>&1 &
PID4=$!
echo "   PID: $PID4"

echo ""
echo "======================================================================"
echo "All experiments started!"
echo "======================================================================"
echo ""
echo "Monitor progress:"
echo "  tail -f rl/experiment_logs/exp_baseline.log"
echo "  tail -f rl/experiment_logs/exp_fast_learning.log"
echo "  tail -f rl/experiment_logs/exp_large_batch.log"
echo "  tail -f rl/experiment_logs/exp_long_term.log"
echo ""
echo "Check status with:"
echo "  ps -p $PID1 -p $PID2 -p $PID3 -p $PID4"
echo ""
echo "View results with TensorBoard:"
echo "  tensorboard --logdir=rl/logs/"
echo ""

# Wait for all to complete
echo "Waiting for experiments to finish..."
wait $PID1
echo "✓ Experiment 1 complete"
wait $PID2
echo "✓ Experiment 2 complete"
wait $PID3
echo "✓ Experiment 3 complete"
wait $PID4
echo "✓ Experiment 4 complete"

echo ""
echo "======================================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  - rl/models/exp_baseline/"
echo "  - rl/models/exp_fast_learning/"
echo "  - rl/models/exp_large_batch/"
echo "  - rl/models/exp_long_term/"
echo ""
echo "Next steps:"
echo "  1. Launch TensorBoard to compare learning curves:"
echo "     tensorboard --logdir=rl/logs/"
echo ""
echo "  2. Look for:"
echo "     - Which converges fastest?"
echo "     - Which achieves highest reward?"
echo "     - Which is most stable (least variance)?"
echo ""
echo "  3. Use the best hyperparameters for your final training run!"
echo ""
