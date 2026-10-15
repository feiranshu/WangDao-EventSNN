param(
    [ValidateSet('check','prepare','evaluate','status')][string]$Action = 'check',
    [ValidateSet('count_reference','spike_add')][string]$Backend = 'count_reference',
    [int[]]$Steps = @(4,8,16,32,64,255),
    [int]$ChunkRows = 8192,
    [string]$RunDirectory = 'results/runs/h2pro_direct_v1_dev',
    [string]$Cache = 'results/runs/h2pro_direct_v1_features',
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
    $cliArgs = @('-B','software/scripts/h2pro_convert.py',$Action,'--cache',$Cache,
        '--run-directory',$RunDirectory,'--teacher-cache',$TeacherCache,'--dataset-root',$DatasetRoot,
        '--backend',$Backend,'--chunk-rows',"$ChunkRows",'--steps') + @($Steps | ForEach-Object { "$_" })
    & $Python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw "Conversion workflow exited $LASTEXITCODE; completed sequence results are resumable." }
} finally {
    Pop-Location
}
