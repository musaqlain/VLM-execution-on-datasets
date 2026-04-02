#!/bin/bash
# ================================================================
# run_all.sh  –  Multi-GPU VLM Evaluation (targets 20-30 hours)
# ================================================================
# Uses a unified virtual environment and the multi-GPU orchestrator
# to run all 8 models × 4 datasets across 16 GPUs in parallel.
#
# PREREQUISITES:
#   cd "/home/aiserver/Documents/opensource/VLM-execution-on-datasets/VLM_Evaluation_Workspace"
#   bash setup_envs.sh          # one-time: creates vlm_env_main
#
# HOW TO RUN:
#   cd "/home/aiserver/Documents/opensource/VLM-execution-on-datasets/VLM_Evaluation_Workspace"
#   nohup bash run_all.sh > master_log.txt 2>&1 &
#
# HOW TO WATCH PROGRESS (from another terminal):
#   tail -f "/home/aiserver/Documents/opensource/VLM-execution-on-datasets/VLM_Evaluation_Workspace/master_log.txt"
# ================================================================

# Auto-load .env (this file is ignored in git, so it's safe to store tokens here)
if [ -f "$(dirname "$0")/.env" ]; then
    export $(grep -v '^#' "$(dirname "$0")/.env" | xargs)
fi

export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=1

WORKSPACE="/home/aiserver/Documents/opensource/VLM-execution-on-datasets/VLM_Evaluation_Workspace"
MAX_SAMPLES=5000
NUM_GPUS=16

echo "========================================================"
echo "  VLM BENCHMARK (MULTI-GPU)  •  $(date)"
echo "  Models: 8  |  Datasets: 4  |  Samples/dataset: $MAX_SAMPLES"
echo "  GPUs: $NUM_GPUS  |  Datasets: RSVLM-QA, DisasterM3, RSVQA-HR, EarthVQA"
echo "  Mode: Parallel (multi_gpu_runner.py)"
echo "========================================================"

# Activate the unified environment
echo "🔄 Activating vlm_env_main"
source "$WORKSPACE/vlm_env_main/bin/activate"

mkdir -p results

# ── Multi-GPU parallel mode (default) ──────────────────────
python multi_gpu_runner.py \
    --num_gpus $NUM_GPUS \
    --max_samples $MAX_SAMPLES \
    --results_dir results

deactivate

echo ""
echo "========================================================"
echo "  ✅  ALL DONE  •  $(date)"
echo "  Results saved in ./results/"
echo "========================================================"
