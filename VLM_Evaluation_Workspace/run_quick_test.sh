#!/bin/bash
# ================================================================
# run_quick_test.sh  –  Sequential smoke test (~5-10 minutes)
# ================================================================
# WHAT THIS DOES:
#   Runs the SINGLE FASTEST model (moondream2) on ALL 4 datasets,
#   3 samples each, SEQUENTIALLY. This tests the full pipeline
#   without parallel download races or GPU contention.
#
# WHY SEQUENTIAL (not parallel):
#   The parallel runner requires all models to already be cached
#   in ~/.cache/huggingface/hub. If models are not cached, 32
#   parallel downloads deadlock the network and HF rate-limiter.
#   Run predownload_models.sh FIRST to cache all models, then
#   run the full parallel run_all.sh.
#
# HOW TO RUN:
#   cd VLM_Evaluation_Workspace
#   bash run_quick_test.sh 2>&1 | tee test_log.txt
#
# AFTER IT FINISHES:
#   git add . && git commit -m "quick test results" && git push
# ================================================================

set -e   # Exit on any error — catches wrong paths early

# ── Environment ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-load .env if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -o allexport
    source "$SCRIPT_DIR/.env"
    set +o allexport
fi

export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1   # <-- CRITICAL: prevents Python output buffering

MAX_SAMPLES=3
MODEL="moondream2"          # Fastest model: ~3.6GB, ~30s per sample
RESULTS_DIR="results_test"

echo "========================================================"
echo "  SEQUENTIAL SMOKE TEST  •  $(date)"
echo "  Model: $MODEL (fastest, ~3.6 GB)"
echo "  Datasets: 4  |  Samples per dataset: $MAX_SAMPLES"
echo "  Output: $RESULTS_DIR/"
echo "========================================================"

# Activate env (only if not already active)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔄 Activating vlm_env_main"
    source "$SCRIPT_DIR/vlm_env_main/bin/activate"
else
    echo "✅ Already in venv: $VIRTUAL_ENV"
fi

mkdir -p "$RESULTS_DIR"

# ── Run 1: RSVLM-QA ─────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────"
echo "  [1/4] RSVLM-QA  •  $(date)"
echo "──────────────────────────────────────────────────────"
python -u run_rsvlmqa.py --model "$MODEL" --max_samples $MAX_SAMPLES --results_dir "$RESULTS_DIR"
echo "  DONE: $RESULTS_DIR/rsvlmqa_${MODEL}.json"

# ── Run 2: DisasterM3 ────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────"
echo "  [2/4] DisasterM3  •  $(date)"
echo "──────────────────────────────────────────────────────"
python -u run_disasterm3.py --model "$MODEL" --max_samples $MAX_SAMPLES --results_dir "$RESULTS_DIR"
echo "  DONE: $RESULTS_DIR/disasterm3_${MODEL}.json"

# ── Run 3: RSVQA-HR ─────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────"
echo "  [3/4] RSVQA-HR  •  $(date)"
echo "──────────────────────────────────────────────────────"
python -u run_rsvqa_hr.py --model "$MODEL" --max_samples $MAX_SAMPLES --results_dir "$RESULTS_DIR"
echo "  DONE: $RESULTS_DIR/rsvqa_hr_${MODEL}.json"

# ── Run 4: EarthVQA ─────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────"
echo "  [4/4] EarthVQA  •  $(date)"
echo "──────────────────────────────────────────────────────"
python -u run_earthvqa.py --model "$MODEL" --max_samples $MAX_SAMPLES --results_dir "$RESULTS_DIR"
echo "  DONE: $RESULTS_DIR/earthvqa_${MODEL}.json"

echo ""
echo "========================================================"
echo "  SMOKE TEST COMPLETE  •  $(date)"
echo "  Results in: $RESULTS_DIR/"
echo ""
echo "  Files produced:"
ls -lh "$RESULTS_DIR/"*.json 2>/dev/null || echo "  (no JSON files found — check for errors above)"
echo ""
echo "  Next steps:"
echo "    git add . && git commit -m 'smoke test results' && git push"
echo "========================================================"
