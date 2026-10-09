[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$toolRoot = $PSScriptRoot
$kv260Root = Split-Path -Parent (Split-Path -Parent $toolRoot)
$workingPackage = Join-Path $kv260Root 'part3\board_working_2026-09-19\wangdao_board_package'
$bundle = Join-Path $toolRoot 'upload_bundle'
$probeSource = Join-Path $toolRoot 'generated\lane_probe'
$timeMajorProbeSource = Join-Path $toolRoot 'generated\time_major_probe'
$boardSource = Join-Path $toolRoot 'board'

$packageFiles = @(
    'wangdao_pl_flat.bit.bin',
    'wangdao_pl_flat.dtbo',
    'shell.json',
    'patch0_words.txt',
    'patch0_expected.txt',
    'patch1_words.txt',
    'patch1_expected.txt'
)
$boardFiles = @(
    'artifact_preflight.sh',
    'wangdao_axi_smoke.sh',
    'wangdao_safe_lib.sh',
    'wangdao_axi_capture.sh',
    'safe_reset_probe.sh',
    'safe_lane_probe.sh',
    'safe_time_major_probe.sh',
    'safe_one_sample.sh'
)

New-Item -ItemType Directory -Path $bundle -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $bundle 'lane_probe') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $bundle 'time_major_probe') -Force | Out-Null

foreach ($name in $packageFiles) {
    $source = Join-Path $workingPackage $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing verified board package file: $source"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $bundle $name) -Force
}
foreach ($name in $boardFiles) {
    $source = Join-Path $boardSource $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing board tool: $source"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $bundle $name) -Force
}
Copy-Item -LiteralPath (Join-Path $toolRoot 'README.md') -Destination (Join-Path $bundle 'README_PC.md') -Force
Copy-Item -Path (Join-Path $probeSource 'lane_probe_*.txt') -Destination (Join-Path $bundle 'lane_probe') -Force
Copy-Item -LiteralPath (Join-Path $probeSource 'lane_probe_manifest.json') -Destination (Join-Path $bundle 'lane_probe') -Force
Copy-Item -Path (Join-Path $timeMajorProbeSource 'time_major_probe_*.txt') -Destination (Join-Path $bundle 'time_major_probe') -Force
Copy-Item -LiteralPath (Join-Path $timeMajorProbeSource 'time_major_probe_manifest.json') -Destination (Join-Path $bundle 'time_major_probe') -Force

$records = Get-ChildItem -LiteralPath $bundle -Recurse -File |
    Where-Object Name -ne 'SHA256SUMS.json' |
    Sort-Object FullName |
    ForEach-Object {
        [ordered]@{
            path = $_.FullName.Substring($bundle.Length + 1).Replace('\', '/')
            bytes = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
$records | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $bundle 'SHA256SUMS.json') -Encoding utf8

Write-Host "UPLOAD_BUNDLE_READY $bundle"
Write-Host "files=$((Get-ChildItem -LiteralPath $bundle -Recurse -File).Count)"
