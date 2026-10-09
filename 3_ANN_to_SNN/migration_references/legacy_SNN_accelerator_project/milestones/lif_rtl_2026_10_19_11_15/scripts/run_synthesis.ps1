param(
    [string]$VivadoRoot = $(if ($env:VIVADO_2025_ROOT) { $env:VIVADO_2025_ROOT } else { "E:\xlinxAMD\2025.1\Vivado" })
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$vivado = Join-Path $VivadoRoot "bin\vivado.bat"
$python = if ($env:SNN_PYTHON) { $env:SNN_PYTHON } else { "python" }
if (-not (Test-Path -LiteralPath $vivado)) { throw "Vivado not found: $vivado" }

$stage = Join-Path $env:TEMP ("snn_lif_synth_" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $stage | Out-Null
foreach ($directory in @("rtl", "constraints", "vivado")) {
    Copy-Item -LiteralPath (Join-Path $root $directory) -Destination $stage -Recurse
}
New-Item -ItemType Directory -Force -Path (Join-Path $stage "reports") | Out-Null
$env:LIF_RTL_ROOT = $stage

$runs = @(
    @{ Top = "int8_lif_baseline_top"; Name = "exact_int8" },
    @{ Top = "pot_lif_pe_top"; Name = "pot_approx" }
)

try {
    foreach ($run in $runs) {
        $stageReport = Join-Path $stage ("reports\" + $run.Name)
        New-Item -ItemType Directory -Force -Path $stageReport | Out-Null
        $log = Join-Path $stageReport "vivado.log"
        $journal = Join-Path $stageReport "vivado.jou"
        & $vivado -mode batch -source (Join-Path $stage "vivado\synth_common.tcl") `
            -log $log -journal $journal -tclargs $run.Top $run.Name
        $exitCode = $LASTEXITCODE
        $destination = Join-Path $root ("reports\" + $run.Name)
        New-Item -ItemType Directory -Force -Path $destination | Out-Null
        Copy-Item -Path (Join-Path $stageReport "*") -Destination $destination -Force -ErrorAction SilentlyContinue
        if ($exitCode -ne 0) { throw "$($run.Name) Vivado run failed; inspect $destination\vivado.log" }
    }
} finally {
    Write-Output "Vivado staging directory: $stage"
}

& $python (Join-Path $PSScriptRoot "summarize_reports.py") --root $root
if ($LASTEXITCODE -ne 0) { throw "Report summarization failed" }
Write-Output "Synthesis complete. Summary: $(Join-Path $root 'reports\summary.json')"
