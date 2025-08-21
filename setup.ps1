# Setup script for Windows (PowerShell)
# Usage (PowerShell, inside this folder):
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # once
#   .\setup.ps1

$ErrorActionPreference = "Stop"

$EnvName = ".venv"
$KernelName = "cv-2025"

function Resolve-Python {
    # Try 'py' launcher first
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @{ Cmd = "py"; Args = @("-3.12") }
    }
    # Fallback to 'python' on PATH
    $pyth = Get-Command python -ErrorAction SilentlyContinue
    if ($pyth) {
        return @{ Cmd = "python"; Args = @() }
    }
    throw "Python not found. Install Python 3.12 (winget install -e --id Python.Python.3.12) and re-run."
}

$pyInfo = Resolve-Python
$Cmd = $pyInfo.Cmd
$BaseArgs = $pyInfo.Args

# Get version
$versionText = & $Cmd $BaseArgs -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
Write-Host "Using Python:" $versionText
$parts = $versionText.Split('.')
if ([int]$parts[0] -lt 3 -or [int]$parts[1] -lt 10) {
    throw "Python $versionText detected. Please install Python 3.12 and try again."
}

# Create venv
& $Cmd $BaseArgs -m venv $EnvName

# Activate venv
$activatePath = Join-Path $EnvName "Scripts\Activate.ps1"
. $activatePath

# Upgrade pip/setuptools/wheel
python -m pip install --upgrade pip setuptools wheel

# Install requirements
pip install --prefer-binary -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name $KernelName --display-name "Python (CV-2025)"

# Import test
python -c "import sys; print('Python:', sys.version); \
import traceback; \
try:\n import cv2, numpy as np; print('OpenCV:', cv2.__version__); print('NumPy:', np.__version__); print('OK')\nexcept Exception as e:\n print('Import test failed:', e)"

Write-Host ""
Write-Host "✅ Done!"
Write-Host "To activate later:  .\.venv\Scripts\Activate.ps1"
Write-Host "Then launch Jupyter Lab:  jupyter lab"
