# run_all_models.ps1 - Run all 5 VLMs on DisasterM3 sequentially
# Usage:
#   Full dataset: powershell -ExecutionPolicy Bypass -File run_all_models.ps1
#   50 samples:   powershell -ExecutionPolicy Bypass -File run_all_models.ps1 -MaxSamples 50
#
# Runs models from smallest to largest to minimize OOM risk.
# Each model is loaded, runs inference, then VRAM is cleared before the next.

param (
    [int]$MaxSamples = 0
)

$ErrorActionPreference = "Stop"
$EVAL_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate venv
& "$EVAL_DIR\vlm_env\Scripts\Activate.ps1"

# Models ordered by VRAM usage (smallest first)
$models = @(
    "moondream2",       # ~3 GB
    "kimi-vl-a3b",      # ~7 GB
    "phi-3.5-vision",   # ~9 GB
    "llava-1.5-7b",     # ~14 GB
    "qwen2.5-vl-7b"     # ~16 GB
)

$startTime = Get-Date

foreach ($model in $models) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "  Starting: $model"
    Write-Host "  Time: $(Get-Date -Format 'HH:mm:ss')"
    Write-Host "========================================"

    if ($MaxSamples -gt 0) {
        python "$EVAL_DIR\run_test.py" `
            --model $model `
            --max_samples $MaxSamples `
            --stratified `
            --output_prefix "eval"
    } else {
        python "$EVAL_DIR\run_test.py" `
            --model $model `
            --stratified `
            --output_prefix "eval"
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] $model failed with exit code $LASTEXITCODE"
        Write-Host "Continuing with next model..."
    } else {
        Write-Host "[OK] $model completed successfully."
    }
}

$elapsed = (Get-Date) - $startTime
Write-Host ""
Write-Host "========================================"
Write-Host "  All models complete!"
Write-Host "  Total time: $($elapsed.ToString('hh\:mm\:ss'))"
Write-Host "  Results in: $EVAL_DIR\results\"
Write-Host "========================================"
