#!/usr/bin/env bash
# run_all_models.sh - Run all 5 VLMs on DisasterM3 sequentially
# Usage:
#   Full dataset: bash run_all_models.sh
#   50 samples:   bash run_all_models.sh 50
#
# Runs models from smallest to largest to minimize OOM risk.
# Each model is loaded, runs inference, then VRAM is cleared before the next.

set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
MAX_SAMPLES=${1:-0}

# Activate venv
source "$EVAL_DIR/vlm_env/bin/activate"

# Models ordered by VRAM usage (smallest first)
# moondream2 is DISABLED — incompatible with transformers>=5.x / PyTorch 2.5
MODELS=(
    "kimi-vl-a3b"      # ~7 GB
    "phi-3.5-vision"   # ~9 GB
    "llava-1.5-7b"     # ~14 GB
    "qwen2.5-vl-7b"    # ~16 GB
)

start_time=$(date +%s)

for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Starting: $model"
    echo "  Time: $(date '+%H:%M:%S')"
    echo "========================================"

    if [ "$MAX_SAMPLES" -gt 0 ]; then
        python "$EVAL_DIR/run_test.py" \
            --model "$model" \
            --max_samples "$MAX_SAMPLES" \
            --stratified \
            --output_prefix "eval"
    else
        python "$EVAL_DIR/run_test.py" \
            --model "$model" \
            --stratified \
            --output_prefix "eval"
    fi

    if [ $? -ne 0 ]; then
        echo "[WARN] $model failed"
        echo "Continuing with next model..."
    else
        echo "[OK] $model completed successfully."
    fi
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo ""
echo "========================================"
echo "  All models complete!"
printf '  Total time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
echo "  Results in: $EVAL_DIR/results/"
echo "========================================"
