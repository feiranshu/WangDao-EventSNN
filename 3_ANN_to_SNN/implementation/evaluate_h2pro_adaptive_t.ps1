param(
    [ValidateSet('run','main','dev','status')][string]$Action = 'run',
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe',
    [string]$Device = 'cuda',
    [int]$BatchSize = 8192,
    [string]$RunDirectory = (Join-Path $PSScriptRoot 'results/runs/h2pro_adaptive_t_v1_s42_ep159')
)
$ErrorActionPreference = 'Stop'
& $Python -B (Join-Path $PSScriptRoot 'software/scripts/evaluate_h2pro_adaptive_t.py') --action $Action --run-directory $RunDirectory --device $Device --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw "Frozen adaptive-T evaluation exited $LASTEXITCODE. Inspect the Python error; completed sequence results are retained." }
