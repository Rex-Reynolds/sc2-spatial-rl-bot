#!/bin/bash
# Long Training Run (50-100 episodes)
# Use after validating with 10-episode test

set -e

echo "======================================================================"
echo "SPATIAL RL BOT - LONG TRAINING RUN"
echo "======================================================================"
echo ""

# Check if test model exists
if [ ! -f "rl/models/spatial_first_run/final_model.pt" ]; then
    echo "⚠️  No test model found!"
    echo "Run the 10-episode test first:"
    echo "  python rl/train_spatial.py --opponent IdleBot --episodes 10"
    exit 1
fi

echo "Starting 100-episode training run..."
echo "This will take ~8-10 hours"
echo ""

# Choice of opponents
echo "Select opponent:"
echo "  1) IdleBot (easy, for baseline)"
echo "  2) RushBot (medium, learn defense)"
echo "  3) MarineMedivacBot (hard, learn macro)"
echo ""
read -p "Enter choice (1-3): " opponent_choice

case $opponent_choice in
    1)
        OPPONENT="IdleBot"
        MODEL_NAME="spatial_100ep_idle"
        ;;
    2)
        OPPONENT="RushBot"
        MODEL_NAME="spatial_100ep_rush"
        ;;
    3)
        OPPONENT="MarineMedivacBot"
        MODEL_NAME="spatial_100ep_macro"
        ;;
    *)
        echo "Invalid choice, using IdleBot"
        OPPONENT="IdleBot"
        MODEL_NAME="spatial_100ep_idle"
        ;;
esac

echo ""
echo "Training configuration:"
echo "  Opponent: $OPPONENT"
echo "  Episodes: 100"
echo "  Model name: $MODEL_NAME"
echo "  Load from: spatial_first_run"
echo ""
read -p "Press Enter to start training..."

python rl/train_spatial.py \
    --opponent $OPPONENT \
    --episodes 100 \
    --model-name $MODEL_NAME \
    --load-model rl/models/spatial_first_run/final_model.pt \
    --learning-rate 0.0003 \
    --step-interval 16 \
    --ppo-epochs 4 \
    --save-freq 10 \
    --use-lstm

echo ""
echo "======================================================================"
echo "TRAINING COMPLETE!"
echo "======================================================================"
echo ""
echo "Model saved to: rl/models/$MODEL_NAME/"
echo ""
echo "View training curves:"
echo "  tensorboard --logdir=rl/logs/$MODEL_NAME"
echo ""
echo "Next steps:"
echo "  1. Analyze performance (TensorBoard)"
echo "  2. Test against different opponents"
echo "  3. Run curriculum learning"
echo ""
