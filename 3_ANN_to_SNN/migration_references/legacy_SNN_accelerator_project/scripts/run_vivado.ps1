$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$vivado = Get-Command vivado -ErrorAction Stop
& $vivado.Source -mode batch -source (Join-Path $projectRoot "hardware/vivado/synth_core.tcl") -nojournal -nolog

