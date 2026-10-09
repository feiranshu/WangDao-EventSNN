param(
    [string]$VivadoRoot = "C:\Xilinx\Vivado\2019.1",
    [string]$PythonPath = "C:\Anaconda\envs\gazelle-train\python.exe",
    [string]$VectorDirectory = "",
    [string]$OutputPath = "",
    [ValidateSet("units", "engine", "two_layers", "proposal")][string]$Suite = "units",
    [ValidateRange(1, 86400)][int]$TimeoutSeconds = 240,
    [string]$ExperimentId = "",
    [switch]$FaultChecks,
    [switch]$SkipZeroGroups,
    [switch]$MeasureCycles,
    [string]$FixedReference = ""
)
$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$runner = Join-Path $repositoryRoot "host_tools\rtl_verification\run_rtl_simulation.py"
$runnerArgs = @($runner, "--vivado-root", $VivadoRoot, "--suite", $Suite,
    "--timeout-seconds", $TimeoutSeconds)
if ($VectorDirectory) { $runnerArgs += @("--vector-directory", $VectorDirectory) }
if ($OutputPath) { $runnerArgs += @("--output", $OutputPath) }
if ($ExperimentId) { $runnerArgs += @("--experiment-id", $ExperimentId) }
if ($FaultChecks) { $runnerArgs += "--fault-checks" }
if ($SkipZeroGroups) { $runnerArgs += "--skip-zero-groups" }
if ($MeasureCycles) { $runnerArgs += "--measure-cycles" }
if ($FixedReference) { $runnerArgs += @("--fixed-reference", $FixedReference) }
& $PythonPath @runnerArgs
if ($LASTEXITCODE -ne 0) { throw "RTL verification failed; inspect the result JSON and logs." }
