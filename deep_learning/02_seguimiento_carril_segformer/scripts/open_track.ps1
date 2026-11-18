[CmdletBinding()]
param([int]$Seed = -1, [string]$TrackId = "")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Webots = "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
if (-not (Test-Path -LiteralPath $Webots)) { throw "Webots was not found: $Webots" }
if ($Seed -lt 0) { $Seed = Get-Random -Minimum 1 -Maximum 2000000000 }
$env:WEBOTS_LANE_CAPTURE = "0"
$env:WEBOTS_LANE_SEED = $Seed.ToString()
$env:WEBOTS_LANE_DATASET = Join-Path $ProjectRoot "data\lane_segmentation_v1"
$env:WEBOTS_LANE_TRACK = $TrackId
if ($TrackId -and -not (Test-Path -LiteralPath (Join-Path $env:WEBOTS_LANE_DATASET "tracks\$TrackId.json"))) {
    throw "Unknown dataset track: $TrackId"
}
Write-Host "Track: $TrackId / preview seed: $Seed"
& $Webots --mode=realtime (Join-Path $ProjectRoot "worlds\lane_dataset.wbt")
