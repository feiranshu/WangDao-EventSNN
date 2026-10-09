$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = if ($env:SNN_PYTHON) { $env:SNN_PYTHON } else { "python" }
$reportDir = Join-Path $root "reports\python"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

Push-Location (Join-Path $root "model")
try {
    & $python run_tests.py | Tee-Object -FilePath (Join-Path $reportDir "unittest.log")
    $testExitCode = $LASTEXITCODE
    if ($testExitCode -ne 0) { throw "Python bit-true tests failed" }
    & $python generate_vectors.py --output (Join-Path $root "tb\vectors\add_vectors.mem") --count 2048
    if ($LASTEXITCODE -ne 0) { throw "Golden-vector generation failed" }
    & $python generate_fp_vectors.py --output-dir (Join-Path $root "tb\vectors") --count 2048
    if ($LASTEXITCODE -ne 0) { throw "Floating-point vector generation failed" }
    & $python compare_precisions.py --output (Join-Path $root "reports\precision_compare\functional_metrics.json")
    if ($LASTEXITCODE -ne 0) { throw "Precision comparison failed" }
} finally {
    $ErrorActionPreference = "Stop"
    Pop-Location
}
