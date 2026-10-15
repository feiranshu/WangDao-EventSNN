param(
    [ValidateSet('check','prepare','evaluate','all','status')][string]$Action = 'check',
    [string]$RunDirectory = 'results/runs/h2pro_proposal_transfer_v1_dev',
    [string]$TeacherCache = 'results/runs/h2pro_teacher_v1',
    [string]$Python = 'C:\Anaconda\envs\gazelle-train\python.exe'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $PSScriptRoot
try {
    & $Python -B 'software/scripts/h2pro_proposal_transfer.py' $Action --teacher-cache $TeacherCache --run-directory $RunDirectory
    if ($LASTEXITCODE -ne 0) { throw "Proposal transfer exited $LASTEXITCODE. Completed sequences can be resumed." }
} finally {
    Pop-Location
}
