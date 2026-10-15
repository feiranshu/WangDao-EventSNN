param(
    [ValidateSet('prepare', 'check', 'train', 'pause', 'status')][string]$Action = 'status',
    [ValidateRange(1, 2147483647)][int]$Epochs = 5,
    [ValidateRange(1, 2147483647)][int]$BatchSize = 8192,
    [ValidateRange(1, 2147483647)][int]$CheckpointBatches = 10,
    [double]$LearningRate = 0.0001,
    [double]$KDAlpha = 0.5,
    [double]$Temperature = 2.0,
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe',
    [string]$Cache = 'results/runs/h2pro_distill_v1_features',
    [string]$RunDirectory = 'results/runs/h2pro_distill_r2t8_v1_s42'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
Push-Location -LiteralPath $PSScriptRoot
try {
    $invariant = [System.Globalization.CultureInfo]::InvariantCulture
    $cliArgs = @('-B', 'software/scripts/train_h2pro_distill.py', $Action,
        '--cache', $Cache, '--run-directory', $RunDirectory, '--epochs', "$Epochs",
        '--batch-size', "$BatchSize", '--checkpoint-batches', "$CheckpointBatches",
        '--learning-rate', $LearningRate.ToString($invariant),
        '--kd-alpha', $KDAlpha.ToString($invariant), '--temperature', $Temperature.ToString($invariant))
    & $Python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw "r2/T8 workflow exited $LASTEXITCODE. Inspect the error and failure_*.json." }
} finally {
    Pop-Location
}
