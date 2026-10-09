param(
    [string]$PackageName = "H2_step2_server_20261003"
)

$ErrorActionPreference = "Stop"
$second = Split-Path -Parent $MyInvocation.MyCommand.Path
$implementation = [IO.Path]::GetFullPath((Join-Path $second "..\..\implementation"))
$upload = Join-Path $second "upload"
$zipPath = Join-Path $upload ($PackageName + ".zip")
$shaPath = Join-Path $upload ($PackageName + ".zip.sha256")
$manifestCopy = Join-Path $upload ($PackageName + ".manifest.json")

New-Item -ItemType Directory -Path $upload -Force | Out-Null
foreach ($target in @($zipPath, $shaPath, $manifestCopy)) {
    if (Test-Path -LiteralPath $target) {
        throw "Refusing to overwrite existing package artifact: $target"
    }
}

$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$workRoot = Join-Path $tempRoot ("wangdao_step2_package_" + [guid]::NewGuid().ToString("N"))
$bundle = Join-Path $workRoot $PackageName
$verifyRoot = Join-Path $tempRoot ("wangdao_step2_verify_" + [guid]::NewGuid().ToString("N"))

function Assert-TemporaryPath([string]$Path) {
    $resolved = [IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe temporary path: $resolved"
    }
    if ($resolved -eq $tempRoot) {
        throw "Refusing to operate on the whole temporary root"
    }
}

function Copy-TreeWithoutCache([string]$Source, [string]$Destination) {
    $sourceRoot = [IO.Path]::GetFullPath($Source)
    Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | Where-Object {
        $_.Extension -ne ".pyc" -and $_.FullName -notmatch "[\\/]__pycache__[\\/]"
    } | ForEach-Object {
        $relative = $_.FullName.Substring($sourceRoot.Length).TrimStart("\", "/")
        $target = Join-Path $Destination $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $target
    }
}

try {
    Assert-TemporaryPath $workRoot
    Assert-TemporaryPath $verifyRoot
    New-Item -ItemType Directory -Path $bundle -Force | Out-Null

    $rootFiles = @(
        "README_SERVER.md",
        "run_step2.py",
        "step2_common.py",
        "step2_training.py",
        "step2_audit.py",
        "collect_step2_results.py",
        "step2_protocol.json",
        "requirements.txt",
        "run_server.sh"
    )
    foreach ($name in $rootFiles) {
        Copy-Item -LiteralPath (Join-Path $second $name) -Destination (Join-Path $bundle $name)
    }
    Copy-Item -LiteralPath (Join-Path $second "README_SERVER.md") -Destination (Join-Path $bundle "README.md")
    Copy-TreeWithoutCache (Join-Path $second "tests") (Join-Path $bundle "tests")
    $runtimeFiles = @(
        "__init__.py",
        "data.py",
        "h2_refiner.py",
        "h2_research.py",
        "phase1_common.py",
        "phase1_metrics.py",
        "runtime.py",
        "h2\__init__.py",
        "h2\constants.py",
        "h2\features.py",
        "h2\numpy_model.py",
        "h2\representation.py",
        "h2\torch_model.py"
    )
    foreach ($relative in $runtimeFiles) {
        $source = Join-Path $implementation ("software\wangdao\" + $relative)
        $target = Join-Path $bundle ("software\wangdao\" + $relative)
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $target
    }

    New-Item -ItemType Directory -Path (Join-Path $bundle "software\weights") -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $implementation "software\weights\H2_D1_seed42.npz") `
        -Destination (Join-Path $bundle "software\weights\H2_D1_seed42.npz")
    New-Item -ItemType Directory -Path (Join-Path $bundle "data\EV-UAV-dataset") -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $implementation "data\index.json") -Destination (Join-Path $bundle "data\index.json")
    @"
Place or symlink the EV-UAV-dataset here on the server.
Do not copy the dataset into the upload archive.
See ../README_SERVER.md for the exact command and required layout.
"@ | Set-Content -LiteralPath (Join-Path $bundle "data\EV-UAV-dataset\PLACE_DATASET_HERE.txt") -Encoding UTF8

    $entries = Get-ChildItem -LiteralPath $bundle -Recurse -File | Sort-Object FullName | ForEach-Object {
        [ordered]@{
            path = $_.FullName.Substring($bundle.Length).TrimStart("\", "/").Replace("\", "/")
            bytes = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $manifest = [ordered]@{
        package = $PackageName
        created = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
        dataset_included = $false
        validation_or_test_results_included = $false
        files = @($entries)
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $bundle "PACKAGE_MANIFEST.json") -Encoding UTF8
    Copy-Item -LiteralPath (Join-Path $bundle "PACKAGE_MANIFEST.json") -Destination $manifestCopy

    python (Join-Path $second "build_posix_zip.py") $bundle $zipPath
    if ($LASTEXITCODE -ne 0) {
        throw "POSIX ZIP creation or permission verification failed"
    }
    $zipHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    "$zipHash  $($PackageName).zip" | Set-Content -LiteralPath $shaPath -Encoding ASCII

    New-Item -ItemType Directory -Path $verifyRoot -Force | Out-Null
    Expand-Archive -LiteralPath $zipPath -DestinationPath $verifyRoot
    $verifiedBundle = Join-Path $verifyRoot $PackageName
    $verifiedManifest = Get-Content -LiteralPath (Join-Path $verifiedBundle "PACKAGE_MANIFEST.json") -Raw | ConvertFrom-Json
    foreach ($entry in $verifiedManifest.files) {
        $target = Join-Path $verifiedBundle $entry.path
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
            throw "Extracted package is missing $($entry.path)"
        }
        $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $entry.sha256) {
            throw "Extracted hash mismatch: $($entry.path)"
        }
    }
    [ordered]@{
        zip = $zipPath
        sha256 = $zipHash
        manifest = $manifestCopy
        verified_files = @($verifiedManifest.files).Count
        extraction_verified = $true
    } | ConvertTo-Json -Depth 4
}
finally {
    foreach ($target in @($workRoot, $verifyRoot)) {
        Assert-TemporaryPath $target
        if (Test-Path -LiteralPath $target) {
            Remove-Item -LiteralPath $target -Recurse -Force
        }
    }
}
