param(
    [ValidateSet('prepare', 'check', 'train', 'pause', 'status')][string]$Action = 'status',
    [ValidateRange(1, 30)][int]$Epochs = 30,
    [ValidateRange(1, 1000)][int]$CheckpointBatches = 100,
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe',
    [string]$Teacher = 'results/runs/h2pro_teacher_v1',
    [string]$Encoded = 'results/runs/h2pro_encoded_sparse_v1',
    [string]$Suite = 'results/runs/h2pro_fulltrain_v1_s42',
    [switch]$RetryFailed
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
Push-Location -LiteralPath $PSScriptRoot
try {
    $cliArgs = @('-B', 'software/scripts/train_h2pro_long.py', $Action,
                 '--teacher', $Teacher, '--encoded', $Encoded, '--suite', $Suite,
                 '--epochs', "$Epochs", '--checkpoint-batches', "$CheckpointBatches")
    if ($RetryFailed) { $cliArgs += '--retry-failed' }
    if ($Action -eq 'check') {
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
        $cliArgs += @('--output', "results/runs/h2pro_longtrain_checks/check_${stamp}.json")
    }
    & $Python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw "Long-training workflow exited $LASTEXITCODE. Inspect failure.json/check output." }
} finally {
    Pop-Location
}
