param(
    [ValidateSet('check', 'train', 'pause', 'status')][string]$Action = 'status',
    [ValidateRange(1, 2147483647)][int]$AdditionalEpochs = 20,
    [ValidateRange(1, 2147483647)][int]$CheckpointBatches = 10,
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe',
    [string]$Cache = 'results/runs/h2pro_distill_v1_features',
    [string]$Source = 'results/runs/h2pro_distill_r2t8_v1_s42',
    [string]$Diagnosis = 'results/runs/h2pro_fixed_diagnosis_v1',
    [string]$Suite = 'results/runs/h2pro_hardcase_v1_s42'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
Push-Location -LiteralPath $PSScriptRoot
try {
    $cliArgs = @('-B', 'software/scripts/train_h2pro_hardcase.py', $Action,
        '--cache', $Cache, '--source', $Source, '--diagnosis', $Diagnosis, '--suite', $Suite,
        '--additional-epochs', "$AdditionalEpochs", '--checkpoint-batches', "$CheckpointBatches")
    & $Python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw "Paired direction5 workflow exited $LASTEXITCODE. Inspect failure_*.json and resume last.pt." }
} finally {
    Pop-Location
}
