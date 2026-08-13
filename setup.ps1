# Robot Vision - Windows environment setup
# Run from PowerShell with:
# powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Set-Location -LiteralPath $PSScriptRoot

$RequiredPython = "3.13"
$EnvName = ".venv"
$KernelName = "robot-vision-2026"
$KernelDisplayName = "Python (Robot Vision 2026)"
$VenvPath = Join-Path $PSScriptRoot $EnvName
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

function Test-PythonCandidate {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [string[]]$Arguments = @()
    )

    try {
        $version = & $Command @Arguments -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $version) {
            return $null
        }

        $majorMinor = ($version.Trim().Split('.')[0..1] -join '.')
        if ($majorMinor -ne $RequiredPython) {
            return $null
        }

        return @{
            Command = $Command
            Arguments = $Arguments
            Version = $version.Trim()
        }
    }
    catch {
        return $null
    }
}

function Resolve-Python {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $candidate = Test-PythonCandidate -Command $launcher.Source -Arguments @("-$RequiredPython")
        if ($candidate) {
            return $candidate
        }
    }

    foreach ($name in @("python3.13", "python")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            $candidate = Test-PythonCandidate -Command $command.Source
            if ($candidate) {
                return $candidate
            }
        }
    }

    throw @"
Python $RequiredPython was not found.
Install Python $RequiredPython and make it available through the Python launcher or PATH.
Suggested command:
  winget install -e --id Python.Python.3.13
Then close PowerShell, open it again, and rerun this script.
"@
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
    }
}

$pythonInfo = Resolve-Python
Write-Host "Using Python $($pythonInfo.Version): $($pythonInfo.Command)"

if (Test-Path -LiteralPath $VenvPython) {
    $venvVersion = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0 -or $venvVersion.Trim() -ne $RequiredPython) {
        throw @"
The existing $EnvName environment was not created with Python $RequiredPython.
Delete the $EnvName folder and run this script again.
"@
    }
    Write-Host "Reusing the existing $EnvName environment."
}
else {
    Write-Host "Creating $EnvName..."
    $venvArguments = @($pythonInfo.Arguments) + @("-m", "venv", $VenvPath)
    Invoke-Checked -Command $pythonInfo.Command -Arguments $venvArguments
}

Write-Host "Updating packaging tools..."
Invoke-Checked -Command $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

Write-Host "Installing course dependencies..."
$requirementsPath = Join-Path $PSScriptRoot "requirements.txt"
Invoke-Checked -Command $VenvPython -Arguments @("-m", "pip", "install", "--prefer-binary", "-r", $requirementsPath)

Write-Host "Registering the Jupyter kernel..."
Invoke-Checked -Command $VenvPython -Arguments @("-m", "ipykernel", "install", "--user", "--name", $KernelName, "--display-name", $KernelDisplayName)

Write-Host "Running the import test..."
Invoke-Checked -Command $VenvPython -Arguments @("-c", "import cv2, numpy, scipy, pandas, matplotlib, PIL, skimage; print('Python environment: OK'); print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__)")

Write-Host ""
Write-Host "Setup completed successfully."
Write-Host "Open this folder in Visual Studio Code and select the following notebook kernel:"
Write-Host "  $KernelDisplayName"
Write-Host ""
Write-Host "Optional - start JupyterLab directly with:"
Write-Host "  .\.venv\Scripts\python.exe -m jupyter lab"
