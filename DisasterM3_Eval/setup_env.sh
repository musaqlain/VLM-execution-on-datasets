#!/usr/bin/env bash
# setup_env.sh — Creates an isolated Python venv for DisasterM3 evaluation
set -e

EVAL_DIR="/home/aipmu/Datasets for VLM/DisasterM3_Eval"
VENV_NAME="vlm_env_disasterm3"
VENV_PATH="$EVAL_DIR/$VENV_NAME"

# ── Load HF_TOKEN from .env if present ──
if [ -f "$EVAL_DIR/.env" ]; then
    export $(grep -v '^#' "$EVAL_DIR/.env" | xargs)
    echo "✅ Loaded HF_TOKEN from .env"
fi

# ── Create venv ──
echo "Creating isolated virtual environment at: $VENV_PATH"
python3 -m venv "$VENV_PATH"

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ── Install PyTorch first (CUDA 12.1 — matches your CUDA 13.0 driver) ──
echo "Installing PyTorch with CUDA support..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# ── Install everything else ──
echo "Installing remaining packages from requirements.txt..."
pip install -r "$EVAL_DIR/requirements.txt"

# ── Download NLTK data needed for BLEU scoring ──
echo "Downloading NLTK punkt tokenizer..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ── Patch transformers_stream_generator for Qwen-VL compatibility ──
echo "Checking for transformers_stream_generator patch..."
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
GENERATOR_UTILS="$SITE_PACKAGES/transformers_stream_generator/main.py"

if [ -f "$GENERATOR_UTILS" ]; then
    # The real fix: patch the import that breaks with newer transformers
    sed -i 's/from transformers.generation_stopping_criteria/from transformers.generation.stopping_criteria/g' "$GENERATOR_UTILS" 2>/dev/null || true
    echo "✅ Patched transformers_stream_generator"
else
    echo "⚠  transformers_stream_generator main.py not found — Qwen-VL may still work without it"
fi

# ── Verify setup ──
echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:           {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:          {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import transformers
print(f'  Transformers:  {transformers.__version__}')
from dotenv import load_dotenv
import os
load_dotenv('$EVAL_DIR/.env')
token = os.environ.get('HF_TOKEN', '')
print(f'  HF_TOKEN:      {\"set (\" + token[:8] + \"...)\" if token else \"NOT SET\"}')
"

echo ""
echo "================================================================"
echo "✅ Environment setup complete!"
echo "To activate manually:"
echo "  source '$VENV_PATH/bin/activate'"
echo "================================================================"
