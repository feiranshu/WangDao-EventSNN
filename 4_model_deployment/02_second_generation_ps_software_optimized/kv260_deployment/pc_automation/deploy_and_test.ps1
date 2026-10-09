[CmdletBinding()]
param(
    [string]$SshTarget = "kria",
    [string]$AppName = "wangdao_pl_flat",
    [string]$PackageDir,
    [string]$InputFile,
    [string]$ExpectedFile,
    [ValidateRange(0, 2147483647)]
    [int]$PatchIndex = 0,
    [switch]$SkipDeploy,
    [switch]$UnloadCurrent,
    [switch]$InteractiveSudo,
    [switch]$KeepRemoteArtifacts
)

# This script intentionally uses only ASCII so it can run under Windows
# PowerShell 5.1 without an encoding-dependent parser error.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-CallerPath {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        $candidate = $PathValue
    } else {
        $candidate = Join-Path (Get-Location).Path $PathValue
    }
    return [System.IO.Path]::GetFullPath($candidate)
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ([string]::IsNullOrWhiteSpace($PackageDir)) {
    $PackageDir = Join-Path $repoRoot 'kv260_deployment\firmware\artifacts'
}
if ([string]::IsNullOrWhiteSpace($InputFile)) {
    $InputFile = Join-Path $repoRoot 'pl_design\test_vectors\proposal_inputs.txt'
}
if ([string]::IsNullOrWhiteSpace($ExpectedFile)) {
    $ExpectedFile = Join-Path $repoRoot 'pl_design\test_vectors\proposal_expected.txt'
}
$PackageDir = Resolve-CallerPath -PathValue $PackageDir
$InputFile = Resolve-CallerPath -PathValue $InputFile
$ExpectedFile = Resolve-CallerPath -PathValue $ExpectedFile

$packageBit = Join-Path $PackageDir 'wangdao_pl_flat.bit.bin'
$packageDtbo = Join-Path $PackageDir 'wangdao_pl_flat.dtbo'
$packageShell = Join-Path $PackageDir 'shell.json'
$remoteSmoke = Join-Path $PSScriptRoot '..\board_runtime\wangdao_axi_smoke.sh'

foreach ($path in @($packageBit, $packageDtbo, $packageShell, $InputFile, $ExpectedFile, $remoteSmoke)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file not found: $path"
    }
}
$preflightScript = Join-Path $PSScriptRoot 'artifact_preflight.ps1'
if (!(Test-Path -LiteralPath $preflightScript -PathType Leaf)) {
    throw "Artifact preflight script not found: $preflightScript"
}
& $preflightScript -ArtifactDirectory $PackageDir
if ($AppName -notmatch '^[A-Za-z0-9_.-]+$') {
    throw "AppName contains unsafe characters: $AppName"
}

Get-Command ssh -ErrorAction Stop | Out-Null
Get-Command scp -ErrorAction Stop | Out-Null

$inputText = Get-Content -LiteralPath $InputFile -Raw -Encoding UTF8
$inputTokens = @($inputText -split '\s+' | Where-Object { $_ -ne '' })
$tokensPerPatch = 512
$wordsPerPatch = 128
$patchStart = $PatchIndex * $tokensPerPatch
if (($inputTokens.Count -lt ($patchStart + $tokensPerPatch))) {
    throw "Input file does not contain patch $PatchIndex"
}

$wordList = New-Object 'System.Collections.Generic.List[string]'
for ($wordIndex = 0; $wordIndex -lt $wordsPerPatch; $wordIndex++) {
    $tokenStart = $patchStart + ($wordIndex * 4)
    $bytes = New-Object 'System.Collections.Generic.List[uint32]'
    for ($byteIndex = 0; $byteIndex -lt 4; $byteIndex++) {
        $token = $inputTokens[$tokenStart + $byteIndex]
        if ($token -notmatch '^[0-9A-Fa-f]+$') {
            throw "Invalid input token '$token' at patch $PatchIndex"
        }
        $byteValue = [Convert]::ToUInt32($token, 16)
        if ($byteValue -gt 255) {
            throw "Input token '$token' is wider than one byte"
        }
        [void]$bytes.Add($byteValue)
    }
    $wordValue = [uint32]($bytes[0] -bor ($bytes[1] -shl 8) -bor ($bytes[2] -shl 16) -bor ($bytes[3] -shl 24))
    [void]$wordList.Add(('0x{0:X8}' -f $wordValue))
}

$expectedLines = @(Get-Content -LiteralPath $ExpectedFile -Encoding UTF8 | Where-Object { $_.Trim() -ne '' })
$expectedStart = $PatchIndex * 16
if ($expectedLines.Count -lt ($expectedStart + 16)) {
    throw "Expected file does not contain patch $PatchIndex"
}
$expectedList = New-Object 'System.Collections.Generic.List[string]'
for ($index = 0; $index -lt 16; $index++) {
    $fields = @($expectedLines[$expectedStart + $index] -split '\s+')
    if ($fields.Count -lt 1 -or $fields[0] -notmatch '^[0-9A-Fa-f]+$') {
        throw "Invalid expected output line: $($expectedLines[$expectedStart + $index])"
    }
    $expectedValue = [Convert]::ToUInt32($fields[0], 16)
    [void]$expectedList.Add(('0x{0:X8}' -f $expectedValue))
}

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('wangdao_kv260_' + [guid]::NewGuid().ToString('N'))
$remoteRoot = '/tmp/wangdao_kv260_' + [guid]::NewGuid().ToString('N')
$localWords = Join-Path $tempRoot 'patch_words.txt'
$localExpected = Join-Path $tempRoot 'expected_words.txt'
$sshOptions = @('-o', 'ConnectTimeout=8')
$scpOptions = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8')
if ($InteractiveSudo) {
    $sshOptions += '-tt'
    $sudoCommand = 'sudo'
} else {
    $sshOptions += @('-o', 'BatchMode=yes')
    $sudoCommand = 'sudo -n'
}
$remotePackage = '/lib/firmware/xilinx/' + $AppName

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

New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
Set-Content -LiteralPath $localWords -Value ($wordList -join [Environment]::NewLine) -Encoding ASCII
Set-Content -LiteralPath $localExpected -Value ($expectedList -join [Environment]::NewLine) -Encoding ASCII

try {
    Write-Host "Checking $SshTarget ..."
    $devmemCheck = 'command -v devmem >/dev/null 2>&1 || (command -v busybox >/dev/null 2>&1 && busybox --list | grep -qx devmem)'
    if ($InteractiveSudo) {
        Invoke-Remote ("set -e; uname -a; command -v xmutil; {0}" -f $devmemCheck)
    } else {
        Invoke-Remote ("set -e; uname -a; command -v xmutil; {0}; {1} true; {1} xmutil listapps" -f $devmemCheck, $sudoCommand)
    }

    Invoke-Remote ("mkdir -p '{0}'" -f $remoteRoot)
    Copy-ToRemote $remoteSmoke ("{0}/wangdao_axi_smoke.sh" -f $remoteRoot)
    Copy-ToRemote $localWords ("{0}/patch_words.txt" -f $remoteRoot)
    Copy-ToRemote $localExpected ("{0}/expected_words.txt" -f $remoteRoot)

    if (!$SkipDeploy) {
        Write-Host "Uploading PL_FLAT package ..."
        Copy-ToRemote $packageBit ("{0}/wangdao_pl_flat.bit.bin" -f $remoteRoot)
        # The rebuilt overlay names the package firmware "wangdao_pl_flat.bit.bin".
        # Keep the legacy alias as a fallback for older cached overlays or package
        # metadata that still refers to "wangdao.bit.bin".
        Copy-ToRemote $packageBit ("{0}/wangdao.bit.bin" -f $remoteRoot)
        Copy-ToRemote $packageDtbo ("{0}/wangdao_pl_flat.dtbo" -f $remoteRoot)
        Copy-ToRemote $packageShell ("{0}/shell.json" -f $remoteRoot)
    }

    $privilegedCommands = New-Object 'System.Collections.Generic.List[string]'
    [void]$privilegedCommands.Add('set -e')
    if ($InteractiveSudo) {
        [void]$privilegedCommands.Add('sudo -v')
    }
    if (!$SkipDeploy) {
        if (!$UnloadCurrent) {
            # Do not attempt a runtime switch while another flat application is
            # active.  Require an explicit -UnloadCurrent acknowledgement.
            [void]$privilegedCommands.Add(("if {0} xmutil listapps | grep -E '^[[:space:]]*[A-Za-z0-9_.-]+[[:space:]].*->[[:digit:]]+' >/dev/null; then echo ACTIVE_APP_REQUIRES_UNLOAD >&2; exit 33; fi" -f $sudoCommand))
        }
        $installCommand = "{0} install -d -m 0755 '$remotePackage'; {0} install -m 0644 '$remoteRoot/wangdao_pl_flat.bit.bin' '$remotePackage/wangdao_pl_flat.bit.bin'; {0} install -m 0644 '$remoteRoot/wangdao.bit.bin' '$remotePackage/wangdao.bit.bin'; {0} install -m 0644 '$remoteRoot/wangdao_pl_flat.dtbo' '$remotePackage/wangdao_pl_flat.dtbo'; {0} install -m 0644 '$remoteRoot/shell.json' '$remotePackage/shell.json'" -f $sudoCommand
        [void]$privilegedCommands.Add($installCommand)
        [void]$privilegedCommands.Add('sleep 2')
        [void]$privilegedCommands.Add(("echo AVAILABLE_APPLICATIONS; {0} xmutil listapps" -f $sudoCommand))
        if ($UnloadCurrent) {
            [void]$privilegedCommands.Add(("echo UNLOAD_CURRENT; {0} xmutil unloadapp" -f $sudoCommand))
        }
        [void]$privilegedCommands.Add(("echo LOAD_APPLICATION; {0} xmutil loadapp '{1}'" -f $sudoCommand, $AppName))
        [void]$privilegedCommands.Add(("echo APPLICATIONS_AFTER_LOAD; {0} xmutil listapps" -f $sudoCommand))
        # xmutil has printed a load error while still returning success on some
        # target images.  Require the package to appear with an active slot before
        # touching AXI registers, otherwise stop with a clear failure.
        [void]$privilegedCommands.Add(("if ! {0} xmutil listapps | grep -E '^[[:space:]]*{1}[[:space:]].*->[[:digit:]]+' >/dev/null; then echo LOAD_VERIFY_FAILED >&2; exit 32; fi" -f $sudoCommand, $AppName))
    }
    Write-Host ("Running AXI-Lite smoke test for patch {0} ..." -f $PatchIndex)
    [void]$privilegedCommands.Add(("{0} sh '{1}/wangdao_axi_smoke.sh' 0xA0000000 '{1}/patch_words.txt' '{1}/expected_words.txt'" -f $sudoCommand, $remoteRoot))
    Invoke-Remote ($privilegedCommands -join '; ')
    Write-Host 'WANGDAO_BOARD_SMOKE_PASS'
}
finally {
    if (!$KeepRemoteArtifacts -and $remoteRoot) {
        try {
            & ssh @sshOptions $SshTarget ("rm -rf '{0}'" -f $remoteRoot) | Out-Null
        }
        catch {
            Write-Warning "Could not remove remote temporary directory $remoteRoot"
        }
    }
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
