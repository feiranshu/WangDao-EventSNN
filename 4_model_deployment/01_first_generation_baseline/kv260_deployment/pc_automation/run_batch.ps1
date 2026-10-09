[CmdletBinding()]
param(
    [string]$SshTarget = "kria",
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [string]$CaptureDirectory,
    [string]$AppName = "wangdao_pl_flat",
    [string]$BaseAddress = "0xA0000000",
    [switch]$SkipActiveCheck,
    [switch]$InteractiveSudo,
    [switch]$KeepRemoteArtifacts,
    [ValidateRange(1, 2147483647)]
    [int]$MaxSequences = 0
)

# ASCII-only: this script is intended to run under Windows PowerShell 5.1.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-CallerPath {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [switch]$MustExist
    )
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        $candidate = $PathValue
    } else {
        $candidate = Join-Path (Get-Location).Path $PathValue
    }
    $full = [System.IO.Path]::GetFullPath($candidate)
    if ($MustExist) {
        return (Resolve-Path -LiteralPath $full -ErrorAction Stop).Path
    }
    return $full
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$manifestPath = Resolve-CallerPath -PathValue $Manifest -MustExist
if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Manifest not found: $manifestPath"
}
$manifestRoot = Split-Path -Parent $manifestPath
if ([string]::IsNullOrWhiteSpace($CaptureDirectory)) {
    $CaptureDirectory = Join-Path $manifestRoot 'captures'
}
$CaptureDirectory = Resolve-CallerPath -PathValue $CaptureDirectory

if ($BaseAddress -notmatch '^0[xX][0-9A-Fa-f]+$') {
    throw "BaseAddress must be a hexadecimal address such as 0xA0000000"
}
$manifestValue = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifestValue.status -ne 'ok' -or [int]$manifestValue.schema_version -ne 1) {
    throw "Manifest is not a successful KV260 batch manifest"
}
$records = @($manifestValue.sequences)
if ($records.Count -eq 0) {
    throw "Manifest contains no sequence records"
}
if ($MaxSequences -gt 0 -and $MaxSequences -lt $records.Count) {
    $records = $records[0..($MaxSequences - 1)]
}

Get-Command ssh -ErrorAction Stop | Out-Null
Get-Command scp -ErrorAction Stop | Out-Null
$driverSource = Join-Path $PSScriptRoot '..\board_runtime\wangdao_axi_batch.c'
if (!(Test-Path -LiteralPath $driverSource -PathType Leaf)) {
    throw "Driver source not found: $driverSource"
}
New-Item -ItemType Directory -Path $CaptureDirectory -Force | Out-Null

$sshOptions = @('-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4')
$scpOptions = @('-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4')
if ($InteractiveSudo) {
    $sshOptions += '-tt'
    $sudo = 'sudo'
} else {
    $sshOptions += @('-o', 'BatchMode=yes')
    $scpOptions += @('-o', 'BatchMode=yes')
    $sudo = 'sudo -n'
}

function Quote-Remote {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value.Contains("'")) {
        throw "Unexpected single quote in generated remote path or pattern"
    }
    return "'" + $Value + "'"
}

function Invoke-Remote {
    param([Parameter(Mandatory = $true)][string]$Command)
    & ssh @sshOptions $SshTarget $Command
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "Remote command failed ($code): $Command"
    }
}

function Copy-ToRemote {
    param(
        [Parameter(Mandatory = $true)][string]$LocalPath,
        [Parameter(Mandatory = $true)][string]$RemotePath
    )
    & scp @scpOptions $LocalPath (('{0}:{1}' -f $SshTarget, $RemotePath))
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "scp failed ($code): $LocalPath"
    }
}

function Copy-FromRemote {
    param(
        [Parameter(Mandatory = $true)][string]$RemotePath,
        [Parameter(Mandatory = $true)][string]$LocalPath
    )
    & scp @scpOptions (('{0}:{1}' -f $SshTarget, $RemotePath)) $LocalPath
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "scp download failed ($code): $RemotePath"
    }
}

$remoteRoot = '/tmp/wangdao_kv260_batch_' + [guid]::NewGuid().ToString('N')
$remoteSource = "$remoteRoot/wangdao_axi_batch.c"
$remoteDriver = "$remoteRoot/wangdao_axi_batch"
$remoteSudo = $sudo

try {
    Write-Host "Checking $SshTarget ..."
    $check = "set -e; uname -a; command -v xmutil; command -v ssh || true"
    Invoke-Remote $check
    if (!$SkipActiveCheck) {
        $activePattern = '^[[:space:]]*' + $AppName + '[[:space:]].*->[[:digit:]]+'
        Invoke-Remote ("set -e; if ! {0} xmutil listapps | grep -E {1} >/dev/null; then echo ACTIVE_APPLICATION_NOT_FOUND >&2; exit 31; fi" -f $remoteSudo, (Quote-Remote $activePattern))
    }

    Invoke-Remote ("set -e; mkdir -p {0}" -f (Quote-Remote $remoteRoot))
    Copy-ToRemote $driverSource $remoteSource
    Write-Host 'Compiling the board-side AXI batch driver ...'
    $compile = "set -e; if command -v cc >/dev/null 2>&1; then cc -O2 -std=c11 -Wall -Wextra -o {0} {1}; elif command -v gcc >/dev/null 2>&1; then gcc -O2 -std=c11 -Wall -Wextra -o {0} {1}; else echo C_COMPILER_NOT_FOUND >&2; exit 34; fi" -f (Quote-Remote $remoteDriver), (Quote-Remote $remoteSource)
    Invoke-Remote $compile
    if ($InteractiveSudo) {
        Invoke-Remote 'set -e; sudo -v'
    }

    $sequenceNumber = 0
    foreach ($record in $records) {
        $sequenceNumber++
        $inputRelative = [string]$record.input_file
        $inputPath = [System.IO.Path]::GetFullPath((Join-Path $manifestRoot $inputRelative))
        if (!(Test-Path -LiteralPath $inputPath -PathType Leaf)) {
            throw "Input file not found: $inputPath"
        }
        $stem = [System.IO.Path]::GetFileNameWithoutExtension([System.IO.Path]::GetFileNameWithoutExtension($inputPath))
        $remoteInput = "$remoteRoot/$stem.inputs.bin"
        $remoteCapture = "$remoteRoot/$stem.capture.bin"
        $localCapture = Join-Path $CaptureDirectory "$stem.capture.bin"
        $patches = [int]$record.patches
        $expectedBytes = $patches * 16 * 4

        Write-Host ("[{0}/{1}] Uploading {2} ({3} patches) ..." -f $sequenceNumber, $records.Count, $record.name, $patches)
        Copy-ToRemote $inputPath $remoteInput
        $run = "set -e; {0} {1} {2} {3} {4}; test `$(stat -c %s {4}) -eq {5}" -f $remoteSudo, (Quote-Remote $remoteDriver), (Quote-Remote $BaseAddress), (Quote-Remote $remoteInput), (Quote-Remote $remoteCapture), $expectedBytes
        Invoke-Remote $run
        Copy-FromRemote $remoteCapture $localCapture
        if (!(Test-Path -LiteralPath $localCapture -PathType Leaf)) {
            throw "Capture was not downloaded: $localCapture"
        }
        if ((Get-Item -LiteralPath $localCapture).Length -ne $expectedBytes) {
            throw "Capture size mismatch for $($record.name)"
        }
        Invoke-Remote ("set -e; rm -f {0} {1}" -f (Quote-Remote $remoteInput), (Quote-Remote $remoteCapture))
    }
    Write-Host 'WANGDAO_BATCH_CAPTURE_PASS'
}
finally {
    if (!$KeepRemoteArtifacts -and $remoteRoot) {
        try {
            & ssh @sshOptions $SshTarget ("rm -rf {0}" -f (Quote-Remote $remoteRoot)) | Out-Null
        }
        catch {
            Write-Warning "Could not remove remote temporary directory $remoteRoot"
        }
    }
}
