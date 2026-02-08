#!/bin/bash
# Process Pro Replays - All-in-One Script
# Run this AFTER downloading replays to rl/data/replays/terran_pro/

set -e  # Exit on error

echo "========================================"
echo "PRO REPLAY PROCESSING PIPELINE"
echo "========================================"
echo ""

# Check if replays exist
REPLAY_DIR="rl/data/replays/terran_pro"
REPLAY_COUNT=$(ls $REPLAY_DIR/*.SC2Replay 2>/dev/null | wc -l | tr -d ' ')

if [ "$REPLAY_COUNT" -eq 0 ]; then
    echo "❌ No replays found in $REPLAY_DIR"
    echo ""
    echo "Please download replays first:"
    echo "  1. Visit: https://lotv.spawningtool.com/replays/?p=Maru"
    echo "  2. Download 20-50 replays"
    echo "  3. Save to: $REPLAY_DIR/"
    echo ""
    exit 1
fi

echo "✓ Found $REPLAY_COUNT replay files"
echo ""

# Activate virtual environment
source venv/bin/activate

# Step 1: Parse replays
echo "Step 1: Parsing replays..."
echo "---"
python rl/replay_parser.py $REPLAY_DIR \
    --output rl/data/pro_replays.pkl \
    --max-replays $REPLAY_COUNT

if [ $? -ne 0 ]; then
    echo "❌ Replay parsing failed"
    exit 1
fi

echo ""
echo "✓ Replays parsed successfully"
echo ""

# Step 2: Train imitation model
echo "Step 2: Training imitation model..."
echo "---"
python rl/train_imitation.py \
    --data rl/data/pro_replays.pkl \
    --output rl/models/pro_imitation \
    --epochs 50 \
    --batch-size 64

if [ $? -ne 0 ]; then
    echo "❌ Imitation training failed"
    exit 1
fi

echo ""
echo "✓ Imitation model trained"
echo ""

# Success!
echo "========================================"
echo "✓ COMPLETE!"
echo "========================================"
echo ""
echo "Your pro imitation bot is ready:"
echo "  Model: rl/models/pro_imitation.zip"
echo ""
echo "Next step - Train RL agent against it:"
echo "  python rl/train.py \\"
echo "      --advanced \\"
echo "      --self-play \\"
echo "      --opponent-model rl/models/pro_imitation \\"
echo "      --episodes 100 \\"
echo "      --model-name rl_vs_pro"
echo ""
