param(
    [switch]$RunTests
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Python environment not found at $PythonExe. Create/install .venv first."
}

Write-Host "[1/2] Running pipeline..." -ForegroundColor Cyan
& $PythonExe "src\asos_pipeline.py"

if ($RunTests) {
    Write-Host "[2/2] Running unit tests..." -ForegroundColor Cyan
    & $PythonExe -m unittest "tests\test_asos_pipeline.py" -v
}

Write-Host "Done." -ForegroundColor Green
Write-Host "Outputs:" -ForegroundColor Green
Write-Host "- data\processed" -ForegroundColor Green
Write-Host "- reports\figures" -ForegroundColor Green
Write-Host "- reports\dashboard_summary.md" -ForegroundColor Green
