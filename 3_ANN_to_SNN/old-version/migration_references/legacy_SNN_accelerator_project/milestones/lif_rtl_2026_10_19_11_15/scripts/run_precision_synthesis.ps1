param(
    [string]$VivadoRoot = $(if ($env:VIVADO_2025_ROOT) { $env:VIVADO_2025_ROOT } else { "E:\xlinxAMD\2025.1\Vivado" })
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$vivado = Join-Path $VivadoRoot "bin\vivado.bat"
$python = if ($env:SNN_PYTHON) { $env:SNN_PYTHON } else { "python" }
if (-not (Test-Path -LiteralPath $vivado)) { throw "Vivado not found: $vivado" }
if (-not (Test-Path -LiteralPath (Join-Path $root "ip\fp32_mul_exact\fp32_mul_exact.xci"))) {
    throw "Floating-point IP is missing. Run scripts/create_fp_ip.ps1 first."
}

$stage = Join-Path $env:TEMP ("snn_lif_precision_synth_" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $stage | Out-Null
foreach ($directory in @("rtl", "constraints", "vivado", "ip")) {
    Copy-Item -LiteralPath (Join-Path $root $directory) -Destination $stage -Recurse
}
New-Item -ItemType Directory -Force -Path (Join-Path $stage "reports") | Out-Null
$env:LIF_RTL_ROOT = $stage

$runs = @(
    @{ Top = "fp32_lif_exact_top"; Name = "fp32_exact" },
    @{ Top = "fp32_lif_approx_top"; Name = "fp32_approx" },
    @{ Top = "fp16_lif_exact_top"; Name = "fp16_exact" },
    @{ Top = "fp16_lif_approx_top"; Name = "fp16_approx" },
    @{ Top = "int8_lif_exact_top"; Name = "int8_exact" },
    @{ Top = "int8_lif_approx_top"; Name = "int8_approx" }
)

foreach ($run in $runs) {
    $stageReport = Join-Path $stage ("reports\" + $run.Name)
    New-Item -ItemType Directory -Force -Path $stageReport | Out-Null
    & $vivado -mode batch -source (Join-Path $stage "vivado\synth_precision_compare.tcl") `
        -log (Join-Path $stageReport "vivado.log") `
        -journal (Join-Path $stageReport "vivado.jou") `
        -tclargs $run.Top $run.Name
    $exitCode = $LASTEXITCODE
    $destination = Join-Path $root ("reports\precision_compare\" + $run.Name)
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    Copy-Item -Path (Join-Path $stageReport "*") -Destination $destination -Recurse -Force -ErrorAction SilentlyContinue
    if ($exitCode -ne 0) { throw "$($run.Name) failed; inspect $destination\vivado.log" }
}

& $python (Join-Path $PSScriptRoot "summarize_precision_results.py") --root $root
if ($LASTEXITCODE -ne 0) { throw "Precision report summarization failed" }
Write-Output "All six runs complete. Reports: $(Join-Path $root 'reports\precision_compare')"
Write-Output "Temporary Vivado directory retained at: $stage"
