param(
    [string]$VivadoRoot = "C:\Xilinx\Vivado\2019.1"
)

$ErrorActionPreference = "Stop"
$startedAt = (Get-Date).ToUniversalTime().ToString("o")
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$hardwareRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = Split-Path -Parent $hardwareRoot
$workDirectory = Join-Path $repositoryRoot "tmp\rtl_sim"
$resolvedRepository = [System.IO.Path]::GetFullPath($repositoryRoot)
$resolvedWork = [System.IO.Path]::GetFullPath($workDirectory)
if (-not $resolvedWork.StartsWith($resolvedRepository + [System.IO.Path]::DirectorySeparatorChar)) {
    throw "Simulation work directory must stay inside the repository"
}

$xvlog = Join-Path $VivadoRoot "bin\xvlog.bat"
$xelab = Join-Path $VivadoRoot "bin\xelab.bat"
$xsim = Join-Path $VivadoRoot "bin\xsim.bat"
foreach ($tool in @($xvlog, $xelab, $xsim)) {
    if (-not (Test-Path -LiteralPath $tool)) {
        throw "Vivado simulation tool was not found: $tool"
    }
}

if (Test-Path -LiteralPath $workDirectory) {
    Remove-Item -LiteralPath $workDirectory -Recurse -Force
}
New-Item -ItemType Directory -Path $workDirectory | Out-Null
Copy-Item -LiteralPath (Join-Path $hardwareRoot "vectors\fixed_add.txt") -Destination $workDirectory
Copy-Item -LiteralPath (Join-Path $hardwareRoot "vectors\fixed_add_tree.txt") -Destination $workDirectory
Copy-Item -LiteralPath (Join-Path $hardwareRoot "vectors\lif_neuron.txt") -Destination $workDirectory
Copy-Item -LiteralPath (Join-Path $hardwareRoot "vectors\linear_lif_neuron.txt") -Destination $workDirectory
Copy-Item -LiteralPath (Join-Path $hardwareRoot "vectors\layer1_weight_f4.txt") -Destination $workDirectory
Copy-Item -LiteralPath (Join-Path $hardwareRoot "vectors\layer1_weight_f5.txt") -Destination $workDirectory
Copy-Item -LiteralPath (Join-Path $hardwareRoot "vectors\layer1_weight_f6.txt") -Destination $workDirectory

$sources = @(
    (Join-Path $hardwareRoot "rtl\fixed_add.sv"),
    (Join-Path $hardwareRoot "rtl\fixed_add_tree.sv"),
    (Join-Path $hardwareRoot "rtl\signed_round_shift.sv"),
    (Join-Path $hardwareRoot "rtl\lif_neuron.sv"),
    (Join-Path $hardwareRoot "rtl\linear_lif_neuron.sv"),
    (Join-Path $hardwareRoot "tb\tb_fixed_add.sv"),
    (Join-Path $hardwareRoot "tb\tb_fixed_add_tree.sv"),
    (Join-Path $hardwareRoot "tb\tb_lif_neuron.sv"),
    (Join-Path $hardwareRoot "tb\tb_linear_lif_neuron.sv")
)
$tests = @(
    @{ Top = "tb_fixed_add"; Vector = "fixed_add.txt"; Manifest = "fixed_add" },
    @{ Top = "tb_fixed_add_tree"; Vector = "fixed_add_tree.txt"; Manifest = "fixed_add_tree" },
    @{ Top = "tb_lif_neuron"; Vector = "lif_neuron.txt"; Manifest = "lif_neuron" },
    @{ Top = "tb_linear_lif_neuron"; Vector = "linear_lif_neuron.txt"; Manifest = "linear_lif_neuron" }
)
$manifest = Get-Content -LiteralPath (Join-Path $hardwareRoot "vectors\manifest.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$completedTests = @()

Push-Location $workDirectory
try {
    & $xvlog --sv @sources
    if ($LASTEXITCODE -ne 0) {
        throw "SystemVerilog compilation failed"
    }
    foreach ($test in $tests) {
        $snapshot = "$($test.Top)_snapshot"
        & $xelab $test.Top -s $snapshot
        if ($LASTEXITCODE -ne 0) {
            throw "$($test.Top) elaboration failed"
        }
        & $xsim $snapshot -runall
        if ($LASTEXITCODE -ne 0) {
            throw "$($test.Top) simulation failed"
        }
        $completedTests += [PSCustomObject][ordered]@{
            testbench = $test.Top
            vector_file = "hardware/vectors/$($test.Vector)"
            records = [int]$manifest.files.($test.Manifest).records
            status = "passed"
        }
    }
} finally {
    Pop-Location
}

$stopwatch.Stop()
$totalRecords = [int](($completedTests | Measure-Object -Property records -Sum).Sum)
$result = [ordered]@{
    schema_version = 1
    status = "ok"
    experiment_id = "RTL-UNIT-SIM-X2019-v2"
    stage = "P4-rtl-unit-simulation"
    data_role = "synthetic_and_hw_calibration_golden_vectors"
    primary_metrics = [ordered]@{
        testbenches = $completedTests.Count
        records = $totalRecords
        passed_records = $totalRecords
        failed_records = 0
    }
    decision = "ready_for_first_layer_stacking"
    conclusion = "Board-independent arithmetic, LIF, and frozen first-layer neuron RTL match the Python golden vectors."
    simulation = [ordered]@{
        tool = "Vivado Simulator"
        version = Split-Path -Leaf $VivadoRoot
        scope = "behavioral_simulation_only"
        board_target = $null
        device_target = $null
        started_at_utc = $startedAt
        elapsed_seconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 6)
        tests = $completedTests
    }
    source = [ordered]@{
        vector_manifest = "hardware/vectors/manifest.json"
        vector_manifest_sha256 = (Get-FileHash -LiteralPath (Join-Path $hardwareRoot "vectors\manifest.json") -Algorithm SHA256).Hash.ToLowerInvariant()
        command = "hardware/run_simulation.ps1 -VivadoRoot $VivadoRoot"
    }
    test_used = $false
    validation_used = $false
}
$resultPath = Join-Path $repositoryRoot "results\rtl_unit_simulation.json"
$json = $result | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText(
    $resultPath,
    $json + [Environment]::NewLine,
    (New-Object System.Text.UTF8Encoding($false))
)
