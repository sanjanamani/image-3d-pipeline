# =======================================================================
# Stage-A setup for native Windows + NVIDIA GPU (PowerShell).
#
# Installs PyTorch (CUDA 12.1) + the layout-preview dependencies into a
# Python 3.10/3.11/3.12 virtual environment.  Does NOT install Trellis
# (its custom CUDA ops don't build on native Windows — use WSL2 or Colab
# for the full Trellis pipeline, Stage B).
#
# Usage (from the repo folder, in PowerShell):
#   .\setup.ps1
#   .venv\Scripts\Activate.ps1
#   python run.py path\to\photo.jpg --mode preview
# =======================================================================

$ErrorActionPreference = "Stop"

Write-Host "== Image-to-3D : Stage-A (preview) setup ==" -ForegroundColor Cyan

# --- 1. Find a SUPPORTED Python (3.10-3.12; NOT 3.13/3.14) ---------------
# PyTorch + prebuilt wheels do not yet cover Python 3.13/3.14, which causes
# pip to try compiling from source and fail.  We require 3.10-3.12.
$pyArgs = $null
foreach ($v in @("3.12", "3.11", "3.10")) {
    try {
        & py "-$v" -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $pyArgs = @("-$v"); break }
    } catch { }
}

if (-not $pyArgs) {
    Write-Host ""
    Write-Host "ERROR: No supported Python (3.10-3.12) found." -ForegroundColor Red
    Write-Host "Your default Python is too new for PyTorch wheels (you have:" -ForegroundColor Red
    (python --version) 2>&1 | Write-Host -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Python 3.12, then re-run this script:" -ForegroundColor Yellow
    Write-Host "  winget install -e --id Python.Python.3.12" -ForegroundColor Yellow
    Write-Host "  # close & reopen PowerShell, then:  .\setup.ps1"
    throw "Supported Python not found."
}
Write-Host "Using Python: $((& py $pyArgs --version) 2>&1)"

# --- 2. (Re)create the virtual environment ------------------------------
if (Test-Path ".venv") {
    Write-Host "Removing existing .venv (may have been built with a wrong Python) ..."
    Remove-Item -Recurse -Force ".venv"
}
Write-Host "Creating virtual environment .venv ..."
& py $pyArgs -m venv .venv
& ".venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip

# --- 3. PyTorch with CUDA 12.4 (RTX 4060 = Ada / CUDA 12.x) --------------
# Use cu124, which ships torch >= 2.6.  transformers refuses to load .bin
# checkpoints (e.g. OneFormer) on torch < 2.6 (CVE-2025-32434), and the
# cu121 index tops out at torch 2.5, so cu124 is required here.
Write-Host "Installing PyTorch (CUDA 12.4, torch >= 2.6) ..." -ForegroundColor Cyan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# --- 4. Preview dependencies --------------------------------------------
Write-Host "Installing preview dependencies ..." -ForegroundColor Cyan
pip install -r requirements-preview.txt

# --- 5. Verify CUDA is visible ------------------------------------------
Write-Host "Verifying CUDA ..." -ForegroundColor Cyan
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'))"

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Next:" -ForegroundColor Green
Write-Host "  .venv\Scripts\Activate.ps1"
Write-Host "  python run.py path\to\photo.jpg --mode preview"
Write-Host "Then open outputs\scene_preview.glb in the Windows '3D Viewer' app."
