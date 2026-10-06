$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = if ($env:SNN_PYTHON) { $env:SNN_PYTHON } else { "python" }
$env:PYTHONPATH = Join-Path $projectRoot "software"
& $python -m unittest discover -s (Join-Path $projectRoot "software/tests") -v
& $python (Join-Path $projectRoot "scripts/export_example.py") --output (Join-Path $projectRoot "results/generated/example")

