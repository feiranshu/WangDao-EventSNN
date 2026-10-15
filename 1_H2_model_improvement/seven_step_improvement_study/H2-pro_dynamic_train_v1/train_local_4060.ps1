param(
    [ValidateSet('prepare','check','train','pause','status')][string]$Action = 'status',
    [ValidateRange(1,2147483647)][int]$Epochs = 5,
    [ValidateRange(1,2147483647)][int]$SourceBatchSize = 32768,
    [ValidateRange(1,2147483647)][int]$CheckpointBatches = 10,
    [ValidateRange(1,64)][int]$Workers = 4,
    [double]$LearningRate = 0.00001,
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe',
    [string]$Cache = 'results/cache/paired_features_v1',
    [string]$RunDirectory = 'results/runs/paired_ann_s42'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
Push-Location -LiteralPath $PSScriptRoot
try {
    $invariant = [System.Globalization.CultureInfo]::InvariantCulture
    $cliArgs = @('-B','run.py',$Action,'--cache',$Cache,'--run-directory',$RunDirectory,
        '--epochs',"$Epochs",'--source-batch-size',"$SourceBatchSize",'--checkpoint-batches',"$CheckpointBatches",
        '--learning-rate',$LearningRate.ToString($invariant),'--workers',"$Workers",'--device','cuda')
    & $Python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw "Paired ANN workflow exited $LASTEXITCODE. Inspect Python output and failure_*.json." }
} finally { Pop-Location }
