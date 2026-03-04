#!/bin/bash
# ================================================================
# run_all.sh  –  Overnight VLM Evaluation (targets 15-24 hours)
# ================================================================
# Uses a unified virtual environment for all models.
# Output goes to BOTH the terminal AND log files.
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

# NOTE: No 'set -e'. If one model fails, the rest continue.

export HF_TOKEN=""
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=1

WORKSPACE="/home/aipmu/Datasets for VLM/VLM_Evaluation_Workspace"
MAX_SAMPLES=700

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

echo "========================================================"
echo "  VLM BENCHMARK  •  $(date)"
echo "  Models: ${#MODELS[@]}  |  Samples/dataset: $MAX_SAMPLES"
echo "  Datasets: RSVLM-QA, DisasterM3"
echo "  Envs: vlm_env_main (all models)"
echo "========================================================"

mkdir -p results

# Activate the unified environment 
echo "🔄 Activating vlm_env_main"
source "$WORKSPACE/vlm_env_main/bin/activate"

for model in "${MODELS[@]}"; do

  echo "────────────────────────────────────────────────────────"
  echo "  MODEL: $model  •  DATASET: RSVLM-QA  •  ENV: vlm_env_main  •  $(date)"
  echo "────────────────────────────────────────────────────────"
  python run_rsvlmqa.py --model "$model" --max_samples $MAX_SAMPLES 2>&1 | tee "results/log_rsvlmqa_${model}.txt"

  echo ""
  echo "────────────────────────────────────────────────────────"
  echo "  MODEL: $model  •  DATASET: DisasterM3  •  ENV: vlm_env_main  •  $(date)"
  echo "────────────────────────────────────────────────────────"
  python run_disasterm3.py --model "$model" --max_samples $MAX_SAMPLES 2>&1 | tee "results/log_disasterm3_${model}.txt"

done

deactivate

echo ""
echo "========================================================"
echo "  ✅  ALL DONE  •  $(date)"
echo "  Results saved in ./results/"
echo "========================================================"
