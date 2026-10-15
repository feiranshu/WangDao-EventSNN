param(
    [ValidateSet('check','prepare','calibrate','evaluate','all','status')][string]$Action = 'check',
    [int]$ChunkRows = 8192,
    [string]$RunDirectory = 'results/runs/h2pro_task_calibration_v1_dev',
    [string]$CalibrationDirectory = 'results/runs/h2pro_task_calibration_v1_fit',
    [string]$Cache = 'results/runs/h2pro_calibration_v1_features_fine',
    [string]$PriorCalibration = 'results/runs/h2pro_calibration_v1_fit_fine',
    [string]$DevCache = 'results/runs/h2pro_direct_v1_features',
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $PSScriptRoot
try {
    $cliArgs = @('-B','software/scripts/h2pro_task_calibrate.py',$Action,'--cache',$Cache,
        '--prior-calibration',$PriorCalibration,'--calibration-directory',$CalibrationDirectory,
        '--run-directory',$RunDirectory,'--dev-cache',$DevCache,'--chunk-rows',"$ChunkRows")
    & $Python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw "Task calibration exited $LASTEXITCODE. Completed points/sequences can be resumed." }
} finally {
    Pop-Location
}
