[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Model = "",
    [ValidateRange(1, 5)][int]$Track = 1,
    [switch]$Batch
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Webots = "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
if (-not (Test-Path -LiteralPath $Webots)) { throw "Webots was not found: $Webots" }
$PythonPath = (& $Python -c "import sys, numpy, cv2, onnxruntime; print(sys.executable)").Trim()
if ($LASTEXITCODE -ne 0) { throw "Python requires numpy, opencv-python and onnxruntime." }
if (-not $Model) { $Model = Join-Path $ProjectRoot "runs\segformer_b0\best_model.onnx" }
if (-not (Test-Path -LiteralPath $Model)) { throw "Export the trained model first: python export_onnx.py" }
if (-not (Test-Path -LiteralPath ([System.IO.Path]::ChangeExtension($Model, ".json")))) { throw "Missing ONNX preprocessing metadata JSON." }
foreach ($Controller in @("turtlebot_lane_follower", "fixed_lane_supervisor")) {
    $Runtime = Join-Path $ProjectRoot "controllers\$Controller\runtime.ini"
    $CommandPath = $PythonPath.Replace('\', '/')
    Set-Content -LiteralPath $Runtime -Value "[python]`nCOMMAND = $CommandPath" -Encoding ASCII
}
$RunDirectory = Join-Path $ProjectRoot ("runs\fixed_track\track_{0:D2}_{1}" -f $Track, (Get-Date -Format "yyyyMMdd_HHmmss_fff"))
New-Item -ItemType Directory -Path $RunDirectory -Force | Out-Null
$env:WEBOTS_LANE_MODEL = [System.IO.Path]::GetFullPath($Model)
$env:WEBOTS_LANE_RUN = $RunDirectory
$env:WEBOTS_LANE_BATCH = if ($Batch) { "1" } else { "0" }
$WorldName = if ($Track -eq 1) { "turtlebot_lane_fixed.wbt" } else { "turtlebot_lane_fixed_{0:D2}.wbt" -f $Track }
Write-Host "Fixed track: $Track. Results: $RunDirectory"
try {
    $WebotsArgs = @("--stdout", "--stderr")
    if ($Batch) { $WebotsArgs += @("--batch", "--mode=fast", "--minimize") }
    else { $WebotsArgs += "--mode=realtime" }
    & $Webots @WebotsArgs (Join-Path $ProjectRoot "worlds\$WorldName")
    if ($LASTEXITCODE -ne 0) { throw "Webots exited with code $LASTEXITCODE; inspect the console and $RunDirectory" }
}
finally {
    Remove-Item Env:WEBOTS_LANE_MODEL, Env:WEBOTS_LANE_RUN, Env:WEBOTS_LANE_BATCH -ErrorAction SilentlyContinue
}
