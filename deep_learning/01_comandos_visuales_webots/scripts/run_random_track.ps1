[CmdletBinding()]
param(
    [int]$Seed = -1,
    [ValidateRange(3, 8)]
    [int]$Rows = 4,
    [ValidateRange(3, 8)]
    [int]$Columns = 4,
    [double]$ConfidenceThreshold = 0.80,
    [switch]$Batch
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$World = Join-Path $ProjectRoot "worlds\turtlebot_random_tracks.wbt"
$Model = Join-Path $ProjectRoot "runs\mobilenet_v3_small\best_model.onnx"
$Webots = "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"

if (-not (Test-Path -LiteralPath $Webots)) {
    throw "Webots was not found at $Webots"
}
if (-not (Test-Path -LiteralPath $World)) {
    throw "World was not found at $World"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "ONNX model was not found. Run export_onnx.py after training."
}
if ($Seed -lt 0) {
    $Seed = Get-Random -Minimum 1 -Maximum 2147483647
}

$env:WEBOTS_MODEL_PATH = $Model
$env:WEBOTS_TRACK_SEED = $Seed.ToString()
$env:WEBOTS_TRACK_ROWS = $Rows.ToString()
$env:WEBOTS_TRACK_COLS = $Columns.ToString()
$env:WEBOTS_CONFIDENCE_THRESHOLD = $ConfidenceThreshold.ToString(
    [System.Globalization.CultureInfo]::InvariantCulture
)
$env:WEBOTS_STABLE_PREDICTIONS = "3"
$env:WEBOTS_FORWARD_DISTANCE = "0.42"
$env:WEBOTS_TURN_ANGLE = "90"
$env:WEBOTS_STOP_STEPS = "24"
$env:WEBOTS_GOAL_TOLERANCE = "0.04"
$env:WEBOTS_TURTLEBOT_BATCH = if ($Batch) { "1" } else { "0" }
$env:WEBOTS_TURTLEBOT_SAVE = if ($Batch) { "1" } else { "0" }

Write-Host "Random-track seed: $Seed"
Write-Host "Grid: ${Rows}x${Columns}"

$Arguments = @()
if ($Batch) {
    $Arguments += "--batch"
    $Arguments += "--mode=fast"
    $Arguments += "--stdout"
    $Arguments += "--stderr"
}
$Arguments += $World

Push-Location $ProjectRoot
try {
    & $Webots @Arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
