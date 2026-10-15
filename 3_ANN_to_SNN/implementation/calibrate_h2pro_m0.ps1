param(
    [ValidateSet('all','calibrate','evaluate','status')][string]$Action = 'status',
    [string]$RunDirectory = 'results/runs/h2pro_m0_calibration_v2',
    [string]$Cache = 'results/runs/h2pro_distill_v1_features',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [int]$ChunkRows = 32768,
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $PSScriptRoot
try {
    & $Python -B software/scripts/h2pro_calibrate_m0.py $Action --cache $Cache --run-directory $RunDirectory --device $Device --chunk-rows $ChunkRows
    if ($LASTEXITCODE -ne 0) { throw "m0 calibration exited $LASTEXITCODE. Inspect the error; completed candidates are resumable." }
} finally {
    Pop-Location
}
