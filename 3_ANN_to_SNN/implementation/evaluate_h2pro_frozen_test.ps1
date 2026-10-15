param(
    [ValidateSet('seal','run','status')][string]$Action = 'run',
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe',
    [int]$BatchSize = 8192,
    [string]$RunDirectory = (Join-Path $PSScriptRoot 'results/runs/h2pro_frozen_test_v1_ep159_20261016')
)
$ErrorActionPreference = 'Stop'
& $Python -B (Join-Path $PSScriptRoot 'software/scripts/evaluate_h2pro_frozen_test.py') --action $Action --run-directory $RunDirectory --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw "Frozen Test evaluation failed (exit $LASTEXITCODE). Completed sequence outputs are retained." }
