[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Output = "",
    [int]$Seed = 4012,
    [int]$SamplesPerTrack = 96,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Webots = "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
if (-not (Test-Path -LiteralPath $Webots)) { throw "Webots was not found: $Webots" }
if (-not $Output) { $Output = Join-Path $ProjectRoot "data\lane_segmentation_v1" }
$Output = [System.IO.Path]::GetFullPath($Output)
$PythonPath = (& $Python -c "import sys, cv2, numpy; print(sys.executable)").Trim()
if ($LASTEXITCODE -ne 0) { throw "Python requires numpy and opencv-python." }
$Runtime = Join-Path $ProjectRoot "controllers\lane_dataset_generator\runtime.ini"
if (-not (Test-Path -LiteralPath $Runtime)) {
    throw "Configure controllers/lane_dataset_generator/runtime.ini using runtime.ini.example and Python $PythonPath before capture."
}
$PrepareArgs = @((Join-Path $PSScriptRoot "prepare_tracks.py"), "--output", $Output,
    "--seed", $Seed.ToString(), "--samples-per-track", $SamplesPerTrack.ToString())
if ($Smoke) { $PrepareArgs += "--smoke" }
& $Python @PrepareArgs
if ($LASTEXITCODE -ne 0) { throw "Track preparation failed." }
$env:WEBOTS_LANE_DATASET = $Output
$env:WEBOTS_LANE_CAPTURE = "1"
$env:WEBOTS_LANE_TRACK = ""
try {
    & $Webots --batch --mode=fast --minimize --stdout --stderr (Join-Path $ProjectRoot "worlds\lane_dataset.wbt")
    if ($LASTEXITCODE -ne 0) { throw "Webots exited with code $LASTEXITCODE" }
}
finally {
    Remove-Item Env:WEBOTS_LANE_CAPTURE -ErrorAction SilentlyContinue
}
if (-not (Test-Path -LiteralPath (Join-Path $Output "capture_complete.json"))) { throw "Capture did not finish; inspect Webots output." }
& $Python (Join-Path $PSScriptRoot "audit_dataset.py") $Output
if ($LASTEXITCODE -ne 0) { throw "Dataset audit failed." }
Write-Host "Dataset ready: $Output"
