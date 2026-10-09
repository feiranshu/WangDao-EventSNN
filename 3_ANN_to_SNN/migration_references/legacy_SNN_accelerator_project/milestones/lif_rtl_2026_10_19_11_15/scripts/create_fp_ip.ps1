param(
    [string]$VivadoRoot = $(if ($env:VIVADO_2025_ROOT) { $env:VIVADO_2025_ROOT } else { "E:\xlinxAMD\2025.1\Vivado" })
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$vivado = Join-Path $VivadoRoot "bin\vivado.bat"
if (-not (Test-Path -LiteralPath $vivado)) { throw "Vivado not found: $vivado" }

$stage = Join-Path $env:TEMP ("snn_lif_fp_ip_" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path (Join-Path $stage "vivado") | Out-Null
Copy-Item -LiteralPath (Join-Path $root "vivado\create_fp_ip.tcl") -Destination (Join-Path $stage "vivado\create_fp_ip.tcl")
$env:LIF_RTL_ROOT = $stage

& $vivado -mode batch -source (Join-Path $stage "vivado\create_fp_ip.tcl") `
    -log (Join-Path $stage "create_fp_ip.log") -journal (Join-Path $stage "create_fp_ip.jou")
if ($LASTEXITCODE -ne 0) { throw "Floating-point IP generation failed: $stage\create_fp_ip.log" }

$destination = Join-Path $root "ip"
New-Item -ItemType Directory -Force -Path $destination | Out-Null
Copy-Item -Path (Join-Path $stage "ip\*") -Destination $destination -Recurse -Force
Copy-Item -LiteralPath (Join-Path $stage "create_fp_ip.log") -Destination (Join-Path $destination "create_fp_ip.log") -Force
Write-Output "Floating-point IP generated at: $destination"
Write-Output "Temporary Vivado directory retained at: $stage"
