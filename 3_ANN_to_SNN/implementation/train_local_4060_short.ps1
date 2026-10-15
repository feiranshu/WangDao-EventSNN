param(
    [ValidateSet('check', 'train', 'pause', 'status')][string]$Action = 'status',
    [ValidateRange(1, 30)][int]$Epochs = 5,
    [ValidateSet('ann_linear', 'snn_linear')][string[]]$Arms = @('ann_linear', 'snn_linear'),
    [ValidateRange(1, 2147483647)][int]$CheckpointBatches = 100,
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe',
    [string]$Teacher = 'results/runs/h2pro_teacher_v1',
    [string]$Encoded = 'results/runs/h2pro_encoded_sparse_v1',
    [string]$Suite = 'results/runs/h2pro_fulltrain_v1_1_s42',
    [switch]$RetryFailed
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
Push-Location -LiteralPath $PSScriptRoot
try {
    $cliArgs = @('-B', 'software/scripts/train_h2pro_short.py', $Action,
                 '--teacher', $Teacher, '--encoded', $Encoded, '--suite', $Suite,
                 '--epochs', "$Epochs", '--checkpoint-batches', "$CheckpointBatches", '--arms') + $Arms
    if ($RetryFailed) { $cliArgs += '--retry-failed' }
    & $Python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw "Short-training workflow exited $LASTEXITCODE. Inspect the error above." }
} finally {
    Pop-Location
}
