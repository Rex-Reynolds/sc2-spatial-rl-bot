#!/bin/bash
# Curriculum Learning - Progressive Training
# Train against increasingly difficult opponents

set -e

echo "======================================================================"
echo "CURRICULUM LEARNING - PROGRESSIVE TRAINING"
echo "======================================================================"
echo ""
echo "This script will train your bot through a curriculum:"
echo "  Stage 1: IdleBot (50 episodes)       - Learn basics"
echo "  Stage 2: RushBot (100 episodes)      - Learn defense"
echo "  Stage 3: MarineMedivacBot (100 ep)   - Learn macro"
echo "  Stage 4: Self-play (200 episodes)    - Master strategy"
echo ""
echo "Total: 450 episodes (~30-40 hours)"
echo ""
read -p "Press Enter to start curriculum training..."

# Stage 1: Basics vs IdleBot
echo ""
echo "======================================================================"
echo "STAGE 1: MASTERING BASICS (vs IdleBot)"
echo "======================================================================"
python rl/train_spatial.py \
    --opponent IdleBot \
    --episodes 50 \
    --model-name curriculum_stage1_idle \
    --learning-rate 0.0003 \
    --step-interval 16 \
    --ppo-epochs 4 \
    --use-lstm

if [ $? -ne 0 ]; then
    echo "✗ Stage 1 failed"
    exit 1
fi
echo "✓ Stage 1 complete!"

# Stage 2: Defense vs RushBot
echo ""
echo "======================================================================"
echo "STAGE 2: LEARNING DEFENSE (vs RushBot)"
echo "======================================================================"
python rl/train_spatial.py \
    --opponent RushBot \
    --episodes 100 \
    --model-name curriculum_stage2_defense \
    --load-model rl/models/curriculum_stage1_idle/final_model.pt \
    --learning-rate 0.0002 \
    --step-interval 16 \
    --ppo-epochs 4 \
    --use-lstm

if [ $? -ne 0 ]; then
    echo "✗ Stage 2 failed"
    exit 1
fi
echo "✓ Stage 2 complete!"

# Stage 3: Macro vs MarineMedivacBot
echo ""
echo "======================================================================"
echo "STAGE 3: LEARNING MACRO (vs MarineMedivacBot)"
echo "======================================================================"
python rl/train_spatial.py \
    --opponent MarineMedivacBot \
    --episodes 100 \
    --model-name curriculum_stage3_macro \
    --load-model rl/models/curriculum_stage2_defense/final_model.pt \
    --learning-rate 0.0002 \
    --step-interval 16 \
    --ppo-epochs 4 \
    --use-lstm

if [ $? -ne 0 ]; then
    echo "✗ Stage 3 failed"
    exit 1
fi
echo "✓ Stage 3 complete!"

# Stage 4: Mastery via Self-Play
echo ""
echo "======================================================================"
echo "STAGE 4: MASTERING STRATEGY (Self-Play)"
echo "======================================================================"
python rl/train_spatial.py \
    --opponent IdleBot \
    --episodes 200 \
    --model-name curriculum_stage4_selfplay \
    --load-model rl/models/curriculum_stage3_macro/final_model.pt \
    --learning-rate 0.0001 \
    --step-interval 8 \
    --ppo-epochs 6 \
    --use-lstm

if [ $? -ne 0 ]; then
    echo "✗ Stage 4 failed"
    exit 1
fi
echo "✓ Stage 4 complete!"

# Summary
echo ""
echo "======================================================================"
echo "CURRICULUM COMPLETE! 🎓"
echo "======================================================================"
echo ""
echo "Your bot has completed the full curriculum:"
echo "  ✓ Stage 1: Basics (50 episodes)"
echo "  ✓ Stage 2: Defense (100 episodes)"
echo "  ✓ Stage 3: Macro (100 episodes)"
echo "  ✓ Stage 4: Strategy (200 episodes)"
echo ""
echo "Final model: rl/models/curriculum_stage4_selfplay/final_model.pt"
echo ""
echo "Expected performance:"
echo "  - Beats IdleBot: 100%"
echo "  - Beats RushBot: 70-80%"
echo "  - Beats MarineMedivacBot: 50-60%"
echo "  - Self-play: Sophisticated strategies"
echo ""
echo "Next steps:"
echo "  1. Test against all opponents"
echo "  2. Analyze learned behaviors"
echo "  3. Continue self-play training (500+ episodes)"
echo "  4. Tournament evaluation"
echo ""
