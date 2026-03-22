Set-Location C:\CHAOS_PRODUCTION
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

Write-Host "=== CHAOS HEALTHCHECK ===" -ForegroundColor Cyan
Write-Host "[1] Python"
python --version

Write-Host "[2] pip"
python -m pip --version

Write-Host "[3] Core imports"
python -c "import ib_insync, numpy; print('IMPORTS OK')"

Write-Host "[4] Git"
git branch --show-current
git status --short

Write-Host "[5] Files"
if (Test-Path .\requirements.txt) { Write-Host "requirements.txt OK" } else { Write-Host "requirements.txt MISSING" }
if (Test-Path .\test_ibkr_connection.py) { Write-Host "test_ibkr_connection.py OK" } else { Write-Host "test_ibkr_connection.py MISSING" }

Write-Host "=== HEALTHCHECK COMPLETE ===" -ForegroundColor Green
