Set-Location C:\CHAOS_PRODUCTION
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
Write-Host "CHAOS environment activated"
python --version
python -m pip --version
