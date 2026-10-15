param(
    [ValidateSet('check','prepare','calibrate','evaluate','all','status')][string]$Action = 'check',
    [int]$ChunkRows = 8192,
    [string]$RunDirectory = 'results/runs/h2pro_calibration_v1_dev',
    [string]$CalibrationDirectory = 'results/runs/h2pro_calibration_v1_fit_fine',
    [string]$Cache = 'results/runs/h2pro_calibration_v1_features_fine',
    [string]$DevCache = 'results/runs/h2pro_direct_v1_features',
    [string]$TeacherCache = 'results/runs/h2pro_teacher_v1',
    [string]$DatasetRoot = 'data/EV-UAV-dataset',
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $PSScriptRoot
try {
    $cliArgs = @('-B','software/scripts/h2pro_calibrate.py',$Action,'--cache',$Cache,
        '--calibration-directory',$CalibrationDirectory,'--run-directory',$RunDirectory,
        '--dev-cache',$DevCache,'--teacher-cache',$TeacherCache,'--dataset-root',$DatasetRoot,
        '--chunk-rows',"$ChunkRows")
    & $Python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw "Calibration workflow exited $LASTEXITCODE. Completed stage/sequence receipts are resumable." }
} finally {
    Pop-Location
}
