#!/bin/bash
# ================================================================
# setup_envs.sh  –  Create a unified virtual environment for VLMs
# ================================================================
# Creates one venv:
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
PATCH_FILE=$(find "$ENV_MAIN" -path "*/transformers_stream_generator/main.py" 2>/dev/null | head -1)
if [ -n "$PATCH_FILE" ] && [ -f "$PATCH_FILE" ]; then
    # Remove the incompatible imports
    sed -i "s/BeamSearchScorer,//g" "$PATCH_FILE"
    sed -i "s/ConstrainedBeamSearchScorer,//g" "$PATCH_FILE"
    echo "✅ Patch applied."
else
    echo "⚠ Could not find transformers_stream_generator to patch!"
fi

# Download NLTK data for METEOR metric
echo "📥 Downloading NLTK data for METEOR …"
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo "✅ vlm_env_main ready  (transformers==$(python -c 'import transformers; print(transformers.__version__)'))"
deactivate

echo ""
echo "========================================================"
echo "  ✅  ENVIRONMENT READY"
echo "  vlm_env_main  → ALL 8 MODELS + 8 METRICS"
echo "========================================================"
