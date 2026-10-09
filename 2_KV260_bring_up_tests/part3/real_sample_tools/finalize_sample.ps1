[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^(?:0123|0132|0213|0231|0312|0321|1023|1032|1203|1230|1302|1320|2013|2031|2103|2130|2301|2310|3012|3021|3102|3120|3201|3210)$')]
    [string]$TimeToLane
)

$ErrorActionPreference = 'Stop'
$python = 'D:\software\anaconda\envs\snn_env\python.exe'
$exporter = Join-Path $PSScriptRoot 'export_real_patch.py'
$sampleDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) 'real_samples\sample_000'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}

& $python $exporter --time-to-lane $TimeToLane --output $sampleDirectory
if ($LASTEXITCODE -ne 0) {
    throw "export_real_patch.py failed with exit code $LASTEXITCODE"
}

$inputPath = Join-Path $sampleDirectory 'sample_000_input_words.txt'
$expectedPath = Join-Path $sampleDirectory 'sample_000_expected_logits.txt'
$metaPath = Join-Path $sampleDirectory 'sample_000_meta.json'
$inputCount = (Get-Content -LiteralPath $inputPath).Count
$expectedCount = (Get-Content -LiteralPath $expectedPath).Count
$meta = Get-Content -Raw -LiteralPath $metaPath | ConvertFrom-Json

if ($inputCount -ne 128) { throw "Expected 128 input words, found $inputCount" }
if ($expectedCount -ne 16) { throw "Expected 16 logits, found $expectedCount" }
if (-not $meta.board_ready) { throw 'Metadata says board_ready=false' }
if ($meta.time_to_lane -ne $TimeToLane) {
    throw "Metadata order $($meta.time_to_lane) does not match $TimeToLane"
}

Write-Host "FINAL_SAMPLE_READY time_to_lane=$TimeToLane inputs=$inputCount logits=$expectedCount"
Get-FileHash -LiteralPath $inputPath, $expectedPath, $metaPath -Algorithm SHA256 |
    Select-Object Path, Hash

