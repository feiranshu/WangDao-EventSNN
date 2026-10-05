param(
    [string]$PackageName = "H2_step3_server_20261004"
)

$ErrorActionPreference = "Stop"
$step3 = Split-Path -Parent $MyInvocation.MyCommand.Path
python (Join-Path $step3 "build_server_package.py") --package-name $PackageName
if ($LASTEXITCODE -ne 0) {
    throw "Step-3 server package build or verification failed"
}
