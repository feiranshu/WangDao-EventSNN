[CmdletBinding()]
param([string]$Python = 'python')

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion.Major -lt 5) { throw 'PowerShell 5.1 or newer is required' }
foreach ($command in @('ssh', 'scp', 'tar')) { Get-Command $command -ErrorAction Stop | Out-Null }
& $Python -c "import sys; assert sys.version_info >= (3, 9), sys.version; import numpy, torch; print(sys.version); print('numpy', numpy.__version__, 'torch', torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.9+, NumPy and PyTorch are required' }
& (Join-Path $PSScriptRoot 'artifact_preflight.ps1')
Write-Host 'WANGDAO_PC_PREREQUISITES_PASS'
