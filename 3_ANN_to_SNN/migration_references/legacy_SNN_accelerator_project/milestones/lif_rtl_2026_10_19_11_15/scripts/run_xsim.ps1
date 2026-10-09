param(
    [string]$VivadoRoot = $(if ($env:VIVADO_2025_ROOT) { $env:VIVADO_2025_ROOT } else { "E:\xlinxAMD\2025.1\Vivado" })
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$xvlog = Join-Path $VivadoRoot "bin\xvlog.bat"
$xelab = Join-Path $VivadoRoot "bin\xelab.bat"
$xsim = Join-Path $VivadoRoot "bin\xsim.bat"
foreach ($tool in @($xvlog, $xelab, $xsim)) {
    if (-not (Test-Path -LiteralPath $tool)) { throw "Vivado simulator tool not found: $tool" }
}

& (Join-Path $PSScriptRoot "run_python_tests.ps1")
if ($LASTEXITCODE -ne 0) { throw "Golden-model tests failed" }

$stage = Join-Path $env:TEMP ("snn_lif_xsim_" + [Guid]::NewGuid().ToString("N"))
$reportDir = Join-Path $root "reports\xsim"
New-Item -ItemType Directory -Force -Path $stage,$reportDir | Out-Null
Copy-Item -LiteralPath (Join-Path $root "rtl") -Destination $stage -Recurse
Copy-Item -LiteralPath (Join-Path $root "tb") -Destination $stage -Recurse

$rtlSources = @(
    "rtl/dual_mode_sat_add.sv",
    "rtl/lif_neuron_dual_mode.sv",
    "rtl/fp_mitchell_multiplier.sv",
    "rtl/fp_lif_neuron_xilinx.sv",
    "rtl/fp_lif_compare_tops.sv",
    "rtl/int8_lif_compare_tops.sv",
    "rtl/int8_mac_accumulator.sv",
    "rtl/pot_shift_accumulator.sv",
    "rtl/int8_lif_baseline_top.sv",
    "rtl/pot_lif_pe_top.sv"
)
$tests = @(
    "tb_dual_mode_sat_add",
    "tb_lif_neuron_dual_mode",
    "tb_fp_mitchell_multiplier",
    "tb_int8_lif_compare",
    "tb_int8_lif_baseline",
    "tb_pot_lif_pe"
)

try {
    foreach ($test in $tests) {
        $testDir = Join-Path $stage $test
        New-Item -ItemType Directory -Force -Path $testDir | Out-Null
        if ($test -eq "tb_dual_mode_sat_add" -or $test -eq "tb_fp_mitchell_multiplier") {
            Copy-Item -LiteralPath (Join-Path $stage "tb\vectors") -Destination $testDir -Recurse
        }
        Push-Location $testDir
        try {
            $sources = $rtlSources | ForEach-Object { Join-Path $stage $_ }
            $sources += Join-Path $stage ("tb/" + $test + ".sv")
            $savedPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            & $xvlog -sv @sources 2>&1 | Tee-Object -FilePath (Join-Path $reportDir ($test + "_xvlog.log"))
            $xvlogExitCode = $LASTEXITCODE
            $ErrorActionPreference = $savedPreference
            if ($xvlogExitCode -ne 0) { throw "${test}: xvlog failed" }
            $ErrorActionPreference = "Continue"
            & $xelab $test -s ($test + "_snapshot") -debug typical 2>&1 | Tee-Object -FilePath (Join-Path $reportDir ($test + "_xelab.log"))
            $xelabExitCode = $LASTEXITCODE
            $ErrorActionPreference = $savedPreference
            if ($xelabExitCode -ne 0) { throw "${test}: xelab failed" }
            $arguments = @(($test + "_snapshot"), "-runall")
            $ErrorActionPreference = "Continue"
            & $xsim @arguments 2>&1 | Tee-Object -FilePath (Join-Path $reportDir ($test + "_xsim.log"))
            $xsimExitCode = $LASTEXITCODE
            $ErrorActionPreference = $savedPreference
            if ($xsimExitCode -ne 0) { throw "${test}: xsim failed" }
            $simLog = Get-Content (Join-Path $reportDir ($test + "_xsim.log")) -Raw
            if ($simLog -notmatch "PASS $test") { throw "$test did not emit its PASS marker" }
        } finally {
            $ErrorActionPreference = "Stop"
            Pop-Location
        }
    }
} finally {
    Write-Output "XSIM staging directory: $stage"
}

Write-Output "All XSIM tests passed. Reports: $reportDir"
