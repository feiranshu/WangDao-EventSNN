$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$repoRootTcl = $repoRoot -replace "\\", "/"
$vivado = "D:\AMDDesignTools\2025.2.1\Vivado\bin\vivado.bat"
$script = Join-Path $PSScriptRoot "generate_kv260_artifacts.tcl"
$manifest = Join-Path $env:APPDATA "Xilinx\Vivado\tclapp\manifest.tcl"
$backup = "$manifest.wangdao-backup"

if (-not (Test-Path -LiteralPath $vivado)) {
    throw "Vivado was not found at $vivado"
}
if (-not (Test-Path -LiteralPath $script)) {
    throw "Vivado Tcl script was not found at $script"
}
if (Test-Path -LiteralPath $backup) {
    throw "Refusing to overwrite an existing manifest backup: $backup"
}

$manifestMoved = $false
try {
    if (Test-Path -LiteralPath $manifest) {
        Move-Item -LiteralPath $manifest -Destination $backup
        $manifestMoved = $true
    }

    $env:WANGDAO_REPO_ROOT = $repoRootTcl
    & $vivado -mode batch -nolog -nojournal -source $script
    if ($LASTEXITCODE -ne 0) {
        throw "Vivado batch generation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:WANGDAO_REPO_ROOT -ErrorAction SilentlyContinue
    if ($manifestMoved) {
        if (Test-Path -LiteralPath $manifest) {
            Remove-Item -LiteralPath $manifest -Force
        }
        Move-Item -LiteralPath $backup -Destination $manifest
    }
}
