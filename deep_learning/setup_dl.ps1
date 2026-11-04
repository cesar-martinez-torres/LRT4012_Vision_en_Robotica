[CmdletBinding()]
param(
    [ValidateSet("Auto", "CUDA", "CPU")]
    [string]$Backend = "Auto",

    [string]$EnvironmentPath = "$env:LOCALAPPDATA\VisionRobotica\vision_robotica_dl",

    [string]$PythonExecutable = ""
)

$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$RequirementsPath = Join-Path $ScriptDirectory "requirements_dl.txt"
$CheckScriptPath = Join-Path $ScriptDirectory "check_dl_environment.py"
$TorchCudaIndex = "https://download.pytorch.org/whl/cu126"
$TorchCpuIndex = "https://download.pytorch.org/whl/cpu"


function Resolve-Python311 {
    param([string]$RequestedPython)

    $candidates = @()
    if ($RequestedPython) {
        $candidates += $RequestedPython
    }
    $candidates += @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    )

    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate)) {
            continue
        }
        $version = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($version -in @("3.11", "3.12")) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $version = & $pythonCommand.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($version -in @("3.11", "3.12")) {
            return $pythonCommand.Source
        }
    }

    throw "Python 3.11 or 3.12 was not found. Install one of these versions and run the installer again."
}


function Install-TorchBackend {
    param(
        [string]$PythonPath,
        [ValidateSet("CUDA", "CPU")]
        [string]$SelectedBackend
    )

    $index = if ($SelectedBackend -eq "CUDA") { $TorchCudaIndex } else { $TorchCpuIndex }
    Write-Host "Installing PyTorch backend: $SelectedBackend"
    & $PythonPath -m pip install --upgrade torch torchvision --index-url $index
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch installation failed for backend $SelectedBackend."
    }
}


$BasePython = Resolve-Python311 -RequestedPython $PythonExecutable
Write-Host "Base Python: $BasePython"
Write-Host "Environment: $EnvironmentPath"

$EnvironmentParent = Split-Path -Parent $EnvironmentPath
New-Item -ItemType Directory -Force -Path $EnvironmentParent | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $EnvironmentPath "Scripts\python.exe"))) {
    & $BasePython -m venv $EnvironmentPath
}

$EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"
& $EnvironmentPython -m pip install --upgrade pip setuptools wheel

$HasNvidia = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$SelectedBackend = $Backend
if ($Backend -eq "Auto") {
    $SelectedBackend = if ($HasNvidia) { "CUDA" } else { "CPU" }
}

try {
    Install-TorchBackend -PythonPath $EnvironmentPython -SelectedBackend $SelectedBackend
    $CudaReady = (& $EnvironmentPython -c "import torch; print('yes' if torch.cuda.is_available() else 'no')").Trim()
    if ($SelectedBackend -eq "CUDA" -and $CudaReady -ne "yes") {
        throw "PyTorch was installed, but CUDA is not available."
    }
}
catch {
    if ($Backend -ne "Auto" -or $SelectedBackend -eq "CPU") {
        throw
    }
    Write-Warning "$($_.Exception.Message) Falling back to the CPU build."
    & $EnvironmentPython -m pip uninstall -y torch torchvision
    Install-TorchBackend -PythonPath $EnvironmentPython -SelectedBackend "CPU"
    $SelectedBackend = "CPU"
}

& $EnvironmentPython -m pip install --upgrade -r $RequirementsPath
if ($LASTEXITCODE -ne 0) {
    throw "Installation of the course packages failed."
}

& $EnvironmentPython -m ipykernel install --user --name vision_robotica_dl --display-name "Python (Vision Robotica DL)"
& $EnvironmentPython $CheckScriptPath

Write-Host ""
Write-Host "Environment ready."
Write-Host "Backend: $SelectedBackend"
Write-Host "Activate with: & '$EnvironmentPath\Scripts\Activate.ps1'"
Write-Host "Jupyter kernel: Python (Vision Robotica DL)"
