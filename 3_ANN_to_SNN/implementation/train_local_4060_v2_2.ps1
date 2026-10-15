param(
    [ValidateSet('check', 'train', 'evaluate')][string]$Action = 'check',
    [ValidateRange(1, 30)][int]$Epochs = 1,
    [ValidateRange(1, 8)][int]$BatchSize = 2,
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe',
    [string]$Cache = 'results/runs/h2pro_teacher_v1',
    [string]$RunDirectory = 'results/runs/h2pro_refiner_fixed_v2_2_s42',
    [string]$Profile = '',
    [ValidateSet('user', 'user_authorized_agent')][string]$LaunchedBy = 'user',
    [switch]$Resume,
    [switch]$Pilot
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Training interpreter missing: $Python"
}
if ($Resume -and $Action -ne 'train') { throw '-Resume requires -Action train' }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath (Join-Path $Cache 'manifest.json') -PathType Leaf)) {
        throw 'Teacher cache missing. Prepare it with train_local_4060.ps1 -Action prepare first.'
    }
    $cliArgs = @('-B', 'software/scripts/train_h2pro_refiner_v2_2.py', $Action, '--device', 'cuda', '--cache', $Cache)
    if ($Action -ne 'evaluate') {
        $cliArgs += @('--batch-size', "$BatchSize")
        if ($Profile) { $cliArgs += @('--profile', $Profile) }
    }
    if ($Action -eq 'check') {
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
        $cliArgs += @('--output', "results/runs/h2pro_v2_2_checks/fixed_v2_2_${stamp}.json")
    } elseif ($Action -eq 'train') {
        $cliArgs += @('--run-directory', $RunDirectory, '--epochs', "$Epochs", '--launched-by', $LaunchedBy)
        if ($Resume) { $cliArgs += '--resume' }
        if ($Pilot) { $cliArgs += '--pilot' }
    } else {
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
        $cliArgs += @('--run-directory', $RunDirectory, '--output', "$RunDirectory/evaluation_${stamp}.json")
    }
    & $Python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw "Workflow failed (exit $LASTEXITCODE). Inspect the Python error above." }
} finally {
    Pop-Location
}
