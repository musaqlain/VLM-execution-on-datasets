#!/bin/bash
# ================================================================
# run_all.sh  –  Multi-GPU VLM Evaluation (targets 20-30 hours)
# ================================================================
# Uses a unified virtual environment and the multi-GPU orchestrator
# to run all 8 models × 4 datasets across 16 GPUs in parallel.
#
# PREREQUISITES:
#   cd "/home/aipmu/Datasets for VLM/VLM_Evaluation_Workspace"
#   bash setup_envs.sh          # one-time: creates vlm_env_main
#
# HOW TO RUN:
#   cd "/home/aipmu/Datasets for VLM/VLM_Evaluation_Workspace"
#   nohup bash run_all.sh > master_log.txt 2>&1 &
#
# HOW TO WATCH PROGRESS (from another terminal):
#   tail -f "/home/aipmu/Datasets for VLM/VLM_Evaluation_Workspace/master_log.txt"
# ================================================================

export HF_TOKEN=""
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=1

WORKSPACE="/home/aipmu/Datasets for VLM/VLM_Evaluation_Workspace"
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
