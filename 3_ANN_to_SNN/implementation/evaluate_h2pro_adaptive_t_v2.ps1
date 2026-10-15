param(
    [ValidateSet('run','main','dev','status')][string]$Action = 'run',
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe',
    [int]$BatchSize = 8192,
    [string]$RunDirectory = (Join-Path $PSScriptRoot 'results/runs/h2pro_adaptive_t_v2_s42_ep159')
)
$ErrorActionPreference = 'Stop'
& $Python -B (Join-Path $PSScriptRoot 'software/scripts/evaluate_h2pro_adaptive_t_v2.py') --action $Action --run-directory $RunDirectory --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw "Frozen step-specific exit evaluation failed (exit $LASTEXITCODE). Completed sequence outputs are retained." }
