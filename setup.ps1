# =======================================================================
# Stage-A setup for native Windows + NVIDIA GPU (PowerShell).
#
# Installs PyTorch (CUDA 12.1) + the layout-preview dependencies.
# Does NOT install Trellis (its custom CUDA ops don't build on native
# Windows — use WSL2 or Colab for the full Trellis pipeline, Stage B).
#
# Usage (from the repo folder, in PowerShell):
#   .\setup.ps1
#   .venv\Scripts\Activate.ps1
#   python run.py path\to\photo.jpg --mode preview
# =======================================================================

$ErrorActionPreference = "Stop"

Write-Host "== Image-to-3D : Stage-A (preview) setup ==" -ForegroundColor Cyan

# --- 1. Check Python -----------------------------------------------------
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { throw "Python not found on PATH. Install Python 3.10/3.11 first." }
Write-Host "Python: $((python --version) 2>&1)"

# --- 2. Virtual environment ---------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv ..."
    python -m venv .venv
}
& ".venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip

# --- 3. PyTorch with CUDA 12.1 (RTX 4060 = Ada / CUDA 12.x) --------------
Write-Host "Installing PyTorch (CUDA 12.1) ..." -ForegroundColor Cyan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

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
