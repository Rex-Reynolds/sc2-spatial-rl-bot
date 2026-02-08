#!/bin/bash
# Spatial Bot Quick Start
# Run this to validate and start training the spatial bot

set -e

echo "======================================================================"
echo "SPATIAL RL BOT - QUICK START"
echo "======================================================================"
echo ""

# Step 1: Test components
echo "Step 1: Testing components..."
echo "----------------------------------------------------------------------"
python rl/test_spatial_components.py
if [ $? -eq 0 ]; then
    echo "✓ Components test passed"
else
    echo "✗ Components test failed"
    exit 1
fi
echo ""

# Step 2: Test game
echo "Step 2: Testing game (this takes ~1-2 minutes)..."
echo "----------------------------------------------------------------------"
python rl/test_spatial_game.py
if [ $? -eq 0 ]; then
    echo "✓ Game test passed"
else
    echo "✗ Game test failed"
    exit 1
fi
echo ""

# Step 3: Training
echo "======================================================================"
echo "ALL TESTS PASSED! STARTING TRAINING"
echo "======================================================================"
echo ""
echo "Running 10 episodes vs IdleBot (will take ~30-40 minutes)..."
echo ""

python rl/train_spatial.py \
    --opponent IdleBot \
    --episodes 10 \
    --model-name spatial_first_run \
    --step-interval 16 \
    --ppo-epochs 4 \
    --learning-rate 0.0003

echo ""
echo "======================================================================"
echo "TRAINING COMPLETE!"
echo "======================================================================"
echo ""
echo "Model saved to: rl/models/spatial_first_run/"
echo ""
echo "View training progress:"
echo "  tensorboard --logdir=rl/logs/spatial_first_run"
echo ""
echo "Next steps:"
echo "  1. Visualize learned policies (heatmaps)"
echo "  2. Train longer (50-100 episodes)"
echo "  3. Try harder opponents (RushBot, self-play)"
echo "  4. Add action masking and better rewards"
echo ""
