#!/bin/bash
# ================================================================
# setup_envs.sh  –  Create a unified virtual environment for VLMs
# ================================================================
# Creates one venv (keeps existing vlm_eval_env as backup):
#   vlm_env_main  – transformers==4.44.2 (all 8 models)
#
# HOW TO RUN:
#   cd "/home/aipmu/Datasets for VLM/VLM_Evaluation_Workspace"
#   bash setup_envs.sh
# ================================================================

set -e
WORKSPACE="/home/aipmu/Datasets for VLM/VLM_Evaluation_Workspace"
cd "$WORKSPACE"

echo "========================================================"
echo "  VLM Environment Setup  •  $(date)"
echo "========================================================"
echo ""

# ── Unified Environment ─────────────────
ENV_MAIN="$WORKSPACE/vlm_env_main"
if [ -d "$ENV_MAIN" ]; then
    echo "⚠  vlm_env_main already exists – skipping creation."
else
    echo "📦 Creating vlm_env_main (transformers==4.44.2) …"
    python3 -m venv "$ENV_MAIN"
fi

echo "📥 Installing packages …"
source "$ENV_MAIN/bin/activate"
pip install --upgrade pip setuptools wheel -q
pip install -r "$WORKSPACE/requirements.txt" -q

echo "🔧 Patching transformers_stream_generator to support Qwen-VL-Chat on transformers 4.44.2 …"
PATCH_FILE="$ENV_MAIN/lib/python3.12/site-packages/transformers_stream_generator/main.py"
if [ -f "$PATCH_FILE" ]; then
    # Remove the incompatible imports
    sed -i "s/BeamSearchScorer,//g" "$PATCH_FILE"
    sed -i "s/ConstrainedBeamSearchScorer,//g" "$PATCH_FILE"
    echo "✅ Patch applied."
else
    echo "⚠ Could not find transformers_stream_generator to patch!"
fi

echo "✅ vlm_env_main ready  (transformers==$(python -c 'import transformers; print(transformers.__version__)'))"
deactivate

echo ""
echo "========================================================"
echo "  ✅  ENVIRONMENT READY"
echo "  vlm_eval_env  → backup (transformers==4.57.6)"
echo "  vlm_env_main  → ALL 8 MODELS"
echo "========================================================"
