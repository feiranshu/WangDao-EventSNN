param(
    [ValidateSet('status','download','adapt','verify','preview','model-check','all')]
    [string]$Action = 'status',
    [string]$RunDirectory = 'results/runs/fred_external_v1',
    [int[]]$Sequences = @(0, 1),
    [string]$DataPython = 'C:\Anaconda\python.exe',
    [string]$ModelPython = 'C:\Anaconda\envs\gazelle-train\python.exe'
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $downloads = Join-Path $RunDirectory 'downloads'
    $dataset = Join-Path $RunDirectory 'dataset'
    $runtime = Join-Path $RunDirectory 'runtime'
    function Invoke-FredPython([string]$Interpreter, [string[]]$Arguments) {
        & $Interpreter -B @Arguments
        if ($LASTEXITCODE -ne 0) { throw "FRED workflow failed (exit $LASTEXITCODE); existing inputs/results are retained." }
    }
    if ($Action -eq 'status') {
        $indexPath = Join-Path $dataset 'index.json'
        if (Test-Path -LiteralPath $indexPath) {
            $index = Get-Content -LiteralPath $indexPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $index | Select-Object dataset,sequence_count,file_count,events,evaluated_events,target_events_weak,total_bytes,requires_eval_mask | ConvertTo-Json
        } else { Write-Output 'Adapted dataset is not complete. Use -Action all to download and prepare the selected pilot sequences.' }
    }
    if ($Action -in @('download','all')) {
        Invoke-FredPython $DataPython (@('software/scripts/prepare_fred.py','download','--directory',$downloads,'--sequences') + @($Sequences | ForEach-Object { "$_" }))
    }
    if ($Action -in @('adapt','all')) {
        Invoke-FredPython $DataPython @('software/scripts/prepare_fred.py','adapt','--downloads',$downloads,'--output',$dataset,'--runtime',$runtime)
    }
    if ($Action -in @('verify','all')) {
        Invoke-FredPython $DataPython @('software/scripts/prepare_fred.py','verify','--dataset',$dataset,'--runtime',$runtime,'--report',(Join-Path $RunDirectory 'verification_full.json'))
    }
    if ($Action -in @('preview','all')) {
        Invoke-FredPython $DataPython @('software/scripts/prepare_fred.py','preview','--dataset',$dataset,'--downloads',$downloads,'--runtime',$runtime,'--output',(Join-Path $RunDirectory 'qa'))
    }
    if ($Action -in @('model-check','all')) {
        Invoke-FredPython $ModelPython @('software/scripts/check_fred_model.py','--dataset',$dataset,'--report',(Join-Path $RunDirectory 'model_smoke.json'))
    }
} finally { Pop-Location }
