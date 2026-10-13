param(
    [switch]$IncludeDatasetCode
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$thirdParty = Join-Path $projectRoot "third_party"
New-Item -ItemType Directory -Force -Path $thirdParty | Out-Null

if (-not (Test-Path (Join-Path $thirdParty "EMS-YOLO/.git"))) {
    git clone https://github.com/BICLab/EMS-YOLO.git (Join-Path $thirdParty "EMS-YOLO")
}
if ($IncludeDatasetCode -and -not (Test-Path (Join-Path $thirdParty "Ev-UAV/.git"))) {
    git clone https://github.com/ChenYichen9527/Ev-UAV.git (Join-Path $thirdParty "Ev-UAV")
}

git -C (Join-Path $thirdParty "EMS-YOLO") rev-parse HEAD
if ($IncludeDatasetCode) {
    git -C (Join-Path $thirdParty "Ev-UAV") rev-parse HEAD
}

