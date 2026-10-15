param(
    [ValidateSet('check', 'train', 'pause', 'status')][string]$Action = 'status',
    [ValidateRange(1, 2147483647)][int]$AdditionalEpochs = 50,
    [ValidateRange(1, 2147483647)][int]$CheckpointBatches = 10,
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe',
    [string]$Cache = 'results/runs/h2pro_distill_v1_features',
    [string]$Source = 'results/runs/h2pro_r1t16_train_v1_s42',
    [string]$Suite = 'results/runs/h2pro_r1_lr_v1_s42_ep159'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
Push-Location -LiteralPath $PSScriptRoot
try {
    $cliArgs = @('-B', 'software/scripts/train_h2pro_r1_lr.py', $Action,
        '--cache', $Cache, '--source', $Source, '--suite', $Suite,
        '--additional-epochs', "$AdditionalEpochs", '--checkpoint-batches', "$CheckpointBatches")
    & $Python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw "r1 paired LR workflow exited $LASTEXITCODE. Inspect failure_*.json; resume the same command." }
} finally {
    Pop-Location
}
