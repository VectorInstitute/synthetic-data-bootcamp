param([int]$Port = 8766)
$ErrorActionPreference = 'Stop'
$methodRoot = Split-Path -Parent $PSScriptRoot
$synthetic10kRoot = Split-Path -Parent $methodRoot
$implementationsRoot = Split-Path -Parent $synthetic10kRoot
$repositoryRoot = Split-Path -Parent $implementationsRoot
$candidates = @(
    (Join-Path $repositoryRoot '.venv\Scripts\python.exe'),
    (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
)
$pythonOnPath = Get-Command python -ErrorAction SilentlyContinue
if ($pythonOnPath) { $candidates += $pythonOnPath.Source }
$dashboardPython = $null
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) {
        & $candidate -c 'import numpy' 2>$null
        if ($LASTEXITCODE -eq 0) { $dashboardPython = $candidate; break }
    }
}
if (-not $dashboardPython) {
    throw 'Python with numpy was not found. Install method_3/requirements.txt in your project environment.'
}
Push-Location -LiteralPath $synthetic10kRoot
try {
    Write-Host "Dashboard: http://127.0.0.1:$Port (Ctrl+C to stop)"
    & $dashboardPython -m method_3.server --port $Port
} finally {
    Pop-Location
}
