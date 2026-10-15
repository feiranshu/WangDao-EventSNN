param(
    [ValidateSet('prepare', 'check', 'train', 'evaluate')]
    [string]$Action = 'check',
    [ValidateSet('fixed', 'spike_route', 'membrane_route')]
    [string]$Mode = 'fixed',
    [ValidateRange(1, 30)][int]$Epochs = 5,
    [ValidateRange(1, 8)][int]$BatchSize = 2,
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe',
    [string]$Cache = 'results/runs/h2pro_teacher_v1',
    [string]$RunDirectory = '',
    [switch]$Resume,
    [switch]$Pilot
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Training interpreter missing: $Python"
}
if ($Resume -and $Action -ne 'train') {
    throw '-Resume is only supported with -Action train'
}
if (-not $RunDirectory) {
    $RunDirectory = "results/runs/h2pro_refiner_${Mode}_s42"
}
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
Push-Location -LiteralPath $PSScriptRoot
try {
    $cliArgs = @('-B', 'software/scripts/train_h2pro_refiner.py', $Action)
    if ($Action -eq 'prepare') {
        $cliArgs += @('--cache', $Cache)
    } elseif ($Action -eq 'check') {
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
        $cliArgs += @('--device', 'cuda', '--mode', $Mode, '--batch-size', "$BatchSize",
                      '--output', "results/runs/h2pro_checks/${Mode}_${stamp}.json")
        if (Test-Path -LiteralPath (Join-Path $Cache 'manifest.json')) {
            $cliArgs += @('--cache', $Cache)
        }
    } elseif ($Action -eq 'train') {
        $cliArgs += @('--device', 'cuda', '--cache', $Cache, '--run-directory', $RunDirectory,
                      '--mode', $Mode, '--batch-size', "$BatchSize", '--epochs', "$Epochs")
        if ($Resume) { $cliArgs += '--resume' }
        if ($Pilot) { $cliArgs += '--pilot' }
    } else {
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
        $cliArgs += @('--device', 'cuda', '--cache', $Cache, '--run-directory', $RunDirectory,
                      '--output', "$RunDirectory/evaluation_${stamp}.json")
    }
    & $Python @cliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Workflow failed (exit $LASTEXITCODE). Inspect the Python error above."
    }
} finally {
    Pop-Location
}
