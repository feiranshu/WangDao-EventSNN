[CmdletBinding()]
param(
    [string]$Python = 'python',
    [string]$OutputRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ([string]::IsNullOrWhiteSpace($OutputRoot)) { $OutputRoot = Join-Path $repoRoot 'experiments\prepared_packages\train_dev10' }
if (![System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path (Get-Location).Path $OutputRoot }
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$batchDir = Join-Path $OutputRoot 's1_batch'
$fullDir = Join-Path $OutputRoot 'board_full'

& (Join-Path $PSScriptRoot 'check_deployment_prereqs.ps1') -Python $Python
& $Python (Join-Path $repoRoot 'host_tools\deployment_preparation\prepare_kv260_batch.py') --role train_dev --output-dir $batchDir
if ($LASTEXITCODE -ne 0) { throw 'prepare_kv260_batch.py failed' }
& $Python (Join-Path $repoRoot 'host_tools\deployment_preparation\prepare_ps_h2_batch.py') --manifest (Join-Path $batchDir 'manifest.json') --output-dir $fullDir
if ($LASTEXITCODE -ne 0) { throw 'prepare_ps_h2_batch.py failed' }

Copy-Item -LiteralPath (Join-Path $batchDir 'manifest.json') -Destination (Join-Path $fullDir 's1_manifest.json') -Force
Copy-Item -LiteralPath (Join-Path $batchDir 'sequences') -Destination $fullDir -Recurse -Force
foreach ($name in @('wangdao_axi_batch.c','wangdao_ps_h2.c','board_full_run.sh','run_measured_full.sh','board_full_run_logged.sh','verify_full.sh','package_board_results.sh')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot ('..\board_runtime\' + $name)) -Destination (Join-Path $fullDir $name) -Force
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot '..\measurement\measure_kv260_power.py') -Destination (Join-Path $fullDir 'measure_kv260_power.py') -Force
$implementationCandidates = @(
    (Join-Path $repoRoot 'verification_evidence\implementation\kv260_synthesis_implementation.json'),
    (Join-Path $repoRoot 'verification_evidence\inherited_first_gen_pl_baseline\implementation\kv260_synthesis_implementation.json')
)
$implementation = $implementationCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($null -eq $implementation) { throw 'matching KV260 implementation/resource evidence is missing' }
$implementationEvidence = Get-Content -LiteralPath $implementation -Raw | ConvertFrom-Json
$firmwareRoot = Join-Path $repoRoot 'kv260_deployment\firmware\artifacts'
$currentFirmware = @()
foreach ($name in @('wangdao_pl_flat.bit.bin','wangdao_pl_flat.dtbo','shell.json')) {
    $path = Join-Path $firmwareRoot $name
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "missing deployment artifact: $path" }
    $currentFirmware += [ordered]@{
        name = $name
        bytes = (Get-Item -LiteralPath $path).Length
        sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$implementationEvidence | Add-Member -NotePropertyName deployment_artifact_verification -NotePropertyValue ([ordered]@{
    status = 'matched_at_package_preparation'
    implementation_source = $implementation
    implementation_source_sha256 = (Get-FileHash -LiteralPath $implementation -Algorithm SHA256).Hash.ToLowerInvariant()
    current_firmware = $currentFirmware
}) -Force
$implementationEvidence | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath (Join-Path $fullDir 'hardware_resource_manifest.json') -Encoding utf8
$archive = Join-Path $OutputRoot 'train_dev10_board_full.tar.gz'
if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
& tar -czf $archive -C $OutputRoot 'board_full'
if ($LASTEXITCODE -ne 0) { throw 'tar failed' }
Write-Host "WANGDAO_TRAIN_DEV10_PACKAGE_READY=$archive"
Write-Host 'Upload the archive to an SD-backed directory. Keep batch_hold_run.sh open in SSH session 1, then run board_full/board_full_run_logged.sh in SSH session 2. The default logged run records timing, complexity, INA260 power/energy and returns the implementation resource snapshot.'
