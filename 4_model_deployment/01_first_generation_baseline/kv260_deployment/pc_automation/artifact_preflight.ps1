[CmdletBinding()]
param([string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\firmware\artifacts'))

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$manifestPath = Join-Path $ArtifactDirectory 'board_validated_manifest.json'
if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Board-validated artifact manifest not found: $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($property in $manifest.files.PSObject.Properties) {
    $path = Join-Path $ArtifactDirectory $property.Name
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing artifact: $path" }
    $actualBytes = (Get-Item -LiteralPath $path).Length
    $actualHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualBytes -ne [int64]$property.Value.bytes) { throw "Size mismatch: $path ($actualBytes)" }
    if ($actualHash -ne [string]$property.Value.sha256) { throw "SHA256 mismatch: $path ($actualHash)" }
    Write-Host ("ARTIFACT_OK {0} bytes={1} sha256={2}" -f $property.Name, $actualBytes, $actualHash)
}
$shell = Get-Content -LiteralPath (Join-Path $ArtifactDirectory 'shell.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($shell.shell_type -ne 'PL_FLAT' -or [string]$shell.num_slots -ne '1') {
    throw 'shell.json must contain shell_type=PL_FLAT and num_slots=1'
}
Write-Host 'WANGDAO_ARTIFACT_PREFLIGHT_PASS'
