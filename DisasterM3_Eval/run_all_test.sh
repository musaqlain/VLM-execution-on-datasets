#!/usr/bin/env bash
# run_all_test.sh — Run all 8 VLMs on N stratified samples from DisasterM3
# Usage: bash run_all_test.sh [NUM_SAMPLES]
#   Default: 50 samples, stratified sampling
set -e

EVAL_DIR="/home/aipmu/Datasets for VLM/DisasterM3_Eval"
NUM_SAMPLES="${1:-50}"

# ── Load HF_TOKEN from .env ──
if [ -f "$EVAL_DIR/.env" ]; then
    export $(grep -v '^#' "$EVAL_DIR/.env" | xargs)
    echo "✅ Loaded HF_TOKEN from .env"
fi

# ── Activate the isolated environment ──
source "$EVAL_DIR/vlm_env_disasterm3/bin/activate"
cd "$EVAL_DIR"

echo "Python: $(which python3)"
echo "PyTorch CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "Samples per model: $NUM_SAMPLES (stratified)"
echo ""

MODELS=(
    "moondream2"
    "blip2-opt-2.7b"
    "llava-1.5-7b"
    "qwen-vl-chat"
    "instructblip-vicuna"
    "idefics2-8b"
    "internvl2-4b"
    "llava-next-llama3"
)

TOTAL=${#MODELS[@]}
PASSED=0
FAILED_MODELS=()

echo "Starting ${NUM_SAMPLES}-sample stratified evaluation for $TOTAL VLMs..."
echo ""

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    NUM=$((i + 1))
    echo "==========================================="
    echo " [$NUM/$TOTAL] Evaluating: $MODEL"
    echo "==========================================="
    
    if python3 run_test.py \
        --model "$MODEL" \
        --max_samples "$NUM_SAMPLES" \
        --stratified \
        --output_prefix "eval"; then
        PASSED=$((PASSED + 1))
        echo "  🧹 VRAM cleared."
        echo "✅ $MODEL passed"
    else
        FAILED_MODELS+=("$MODEL")
        echo "❌ $MODEL FAILED"
    fi
    
    echo ""
    echo "Waiting 5 seconds for VRAM cleanup..."
    sleep 5
done

echo ""
echo "==========================================="
echo " SUMMARY: $PASSED/$TOTAL models passed"
echo "==========================================="

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "❌ Failed models: ${FAILED_MODELS[*]}"
    exit 1
else
    echo "✅ All $TOTAL models evaluated successfully!"
    echo ""
    echo "Generating consolidated report..."
    python3 generate_report.py || echo "⚠ Report generation skipped (script may not exist yet)"
fi
