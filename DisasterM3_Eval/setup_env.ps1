# setup_env.ps1 - Windows PowerShell setup for DisasterM3 VLM evaluation
# Run from the DisasterM3_Eval directory on the Windows server (RTX A4500)
#
# Usage:  powershell -ExecutionPolicy Bypass -File setup_env.ps1

$ErrorActionPreference = "Stop"
$EVAL_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_NAME = "vlm_env"
$VENV_PATH = Join-Path $EVAL_DIR $VENV_NAME

Write-Host "================================================"
Write-Host "  DisasterM3 VLM Evaluation - Environment Setup"
Write-Host "================================================"
Write-Host ""

# ── Load HF_TOKEN from .env if present ──
$envFile = Join-Path $EVAL_DIR ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^([^#][^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
    Write-Host "[OK] Loaded .env"
}

# ── Create venv ──
Write-Host "Creating virtual environment at: $VENV_PATH"
python -m venv $VENV_PATH

Write-Host "Activating virtual environment..."
& "$VENV_PATH\Scripts\Activate.ps1"

Write-Host "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ── Install PyTorch with CUDA ──
Write-Host ""
Write-Host "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── Install remaining packages ──
Write-Host ""
Write-Host "Installing remaining packages..."
pip install -r (Join-Path $EVAL_DIR "requirements.txt")

# ── Download NLTK data for METEOR ──
Write-Host ""
Write-Host "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# ── Verify ──
Write-Host ""
Write-Host "Verifying installation..."
python -c @"
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
print(f'  HF_TOKEN:      {"set" if token else "NOT SET"}')
"@

Write-Host ""
Write-Host "================================================"
Write-Host "  Environment setup complete!"
Write-Host "  To activate manually:"
Write-Host "    $VENV_PATH\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  Quick test:"
Write-Host "    python run_test.py --model moondream2 --max_samples 5 --stratified"
Write-Host "================================================"
