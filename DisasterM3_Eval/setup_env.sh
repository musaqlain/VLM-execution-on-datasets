#!/usr/bin/env bash
# setup_env.sh — Creates an isolated Python venv for DisasterM3 evaluation
# Usage: bash setup_env.sh
set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_NAME="vlm_env"
VENV_PATH="$EVAL_DIR/$VENV_NAME"

echo "================================================"
echo "  DisasterM3 VLM Evaluation - Environment Setup"
echo "================================================"
echo ""

# ── Load HF_TOKEN from .env if present ──
if [ -f "$EVAL_DIR/.env" ]; then
    export $(grep -v '^#' "$EVAL_DIR/.env" | xargs)
    echo "[OK] Loaded .env"
fi

# ── Create venv ──
echo "Creating virtual environment at: $VENV_PATH"
python3 -m venv "$VENV_PATH"

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ── Install PyTorch with CUDA ──
echo ""
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── Install remaining packages ──
echo ""
echo "Installing remaining packages..."
pip install -r "$EVAL_DIR/requirements.txt"

# ── Download NLTK data for METEOR ──
echo ""
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# ── Verify ──
echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:           {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info()
    print(f'  VRAM:          {free/1e9:.1f} GB free / {total/1e9:.1f} GB total')
import transformers
print(f'  Transformers:  {transformers.__version__}')
import os
token = os.environ.get('HF_TOKEN', '')
print(f'  HF_TOKEN:      {\"set\" if token else \"NOT SET\"}')
"

echo ""
echo "================================================"
echo "  Environment setup complete!"
echo "  To activate manually:"
echo "    source $VENV_PATH/bin/activate"
echo ""
echo "  Quick test:"
echo "    python run_test.py --model moondream2 --max_samples 5 --stratified"
echo "================================================"
