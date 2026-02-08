#!/bin/bash
# Test script to compare old rewards vs improved rewards
# Quick 20-episode test to see which learns faster

set -e

echo "======================================================================"
echo "REWARD SHAPING COMPARISON TEST"
echo "======================================================================"
echo "Running 20 episodes with each reward function to compare learning speed"
echo ""

# Test 1: Old rewards (AdvancedRLBot)
echo "📊 Test 1: Original Rewards (AdvancedRLBot)"
echo "----------------------------------------------------------------------"
python rl/train.py \
    --advanced \
    --opponent IdleBot \
    --episodes 20 \
    --model-name test_old_rewards \
    --no-tensorboard

echo ""
echo "✓ Test 1 complete!"
echo ""

# Test 2: Improved rewards (ImprovedRLBot)
echo "📊 Test 2: Improved Rewards (ImprovedRLBot)"
echo "----------------------------------------------------------------------"
python rl/train.py \
    --advanced \
    --use-improved-rewards \
    --opponent IdleBot \
    --episodes 20 \
    --model-name test_improved_rewards \
    --no-tensorboard

echo ""
echo "✓ Test 2 complete!"
echo ""

echo "======================================================================"
echo "COMPARISON COMPLETE"
echo "======================================================================"
echo ""
echo "Models saved to:"
echo "  - rl/models/test_old_rewards/"
echo "  - rl/models/test_improved_rewards/"
echo ""
echo "Next steps:"
echo "  1. Compare final models:"
echo "     python rl/inspect_model.py rl/models/test_old_rewards/sc2_agent_*.zip"
echo "     python rl/inspect_model.py rl/models/test_improved_rewards/sc2_agent_*.zip"
echo ""
echo "  2. Watch both play against IdleBot to see behavioral differences"
echo "  3. If improved rewards work better, use --use-improved-rewards for all future training!"
echo ""
