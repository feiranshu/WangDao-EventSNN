$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$iverilog = Get-Command iverilog -ErrorAction Stop
$vvp = Get-Command vvp -ErrorAction Stop
$build = Join-Path $env:TEMP "snn-accelerator-rtl-tests"
New-Item -ItemType Directory -Force -Path $build | Out-Null

$tests = @(
    @{ Name = "sat_add"; Sources = @("hardware/rtl/sat_add_dual_mode.sv", "hardware/tb/tb_sat_add_dual_mode.sv") },
    @{ Name = "lif"; Sources = @("hardware/rtl/sat_add_dual_mode.sv", "hardware/rtl/lif_neuron.sv", "hardware/tb/tb_lif_neuron.sv") },
    @{ Name = "parser"; Sources = @("hardware/rtl/aer_parser.sv", "hardware/tb/tb_aer_parser.sv") }
)

foreach ($test in $tests) {
    $output = Join-Path $build ($test.Name + ".vvp")
    $sources = $test.Sources | ForEach-Object { Join-Path $projectRoot $_ }
    & $iverilog.Source -g2012 -o $output @sources
    & $vvp.Source $output
}

