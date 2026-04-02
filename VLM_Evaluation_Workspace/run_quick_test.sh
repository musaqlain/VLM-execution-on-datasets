#!/bin/bash
# ================================================================
# run_quick_test.sh  –  Fast end-to-end smoke test (~5-10 minutes)
# ================================================================
# Runs ALL 8 models × ALL 4 datasets with only 3 samples each.
# Uses the multi-GPU runner to run jobs in parallel.
#
# Purpose: verify the full pipeline works before the final
# production run (run_all.sh with 5000 samples).
#
# Test results go into results_test/ (separate from production results/).
# Individual logs for each job also go there.
#
# HOW TO RUN:
#   cd "/home/aiserver/Documents/opensource/VLM-execution-on-datasets/VLM_Evaluation_Workspace"
#   bash run_quick_test.sh 2>&1 | tee test_log.txt
#
# After it finishes:
#   1) Check test_log.txt for the summary table
#   2) Check results_test/*.json for individual model outputs
#   3) Check results_test/log_*.txt for per-job logs
#   4) git add . && git commit -m "quick test results" && git push
#   5) Pull on your local machine and share with me for analysis
# ================================================================

# Auto-load .env (this file is ignored in git, so it's safe to store tokens here)
if [ -f "$(dirname "$0")/.env" ]; then
    export $(grep -v '^#' "$(dirname "$0")/.env" | xargs)
fi

export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=1

WORKSPACE="/home/aiserver/Documents/opensource/VLM-execution-on-datasets/VLM_Evaluation_Workspace"
MAX_SAMPLES=3
NUM_GPUS=16

echo "========================================================"
echo "  QUICK TEST  •  $(date)"
echo "  Models: 8  |  Datasets: 4  |  Samples/dataset: $MAX_SAMPLES"
echo "  GPUs: $NUM_GPUS"
echo "  Datasets: RSVLM-QA, DisasterM3, RSVQA-HR, EarthVQA"
echo "  Output: results_test/"
echo "========================================================"

# Activate the unified environment
echo "🔄 Activating vlm_env_main"
source "$WORKSPACE/vlm_env_main/bin/activate"

mkdir -p results_test

# Run ALL models × ALL datasets with 3 samples each
python multi_gpu_runner.py \
    --num_gpus $NUM_GPUS \
    --max_samples $MAX_SAMPLES \
    --results_dir results_test

deactivate

echo ""
echo "========================================================"
echo "  ✅  QUICK TEST DONE  •  $(date)"
echo "  Check results_test/ for output files"
echo ""
echo "  Next steps:"
echo "    1) Review results_test/*.json"
echo "    2) git add . && git commit -m 'quick test results' && git push"
echo "    3) Share results with Claude for analysis"
echo "========================================================"
