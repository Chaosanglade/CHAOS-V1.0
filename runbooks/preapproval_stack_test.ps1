Set-Location C:\CHAOS_PRODUCTION
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

Write-Host "=== PREAPPROVAL STACK TEST START ===" -ForegroundColor Cyan

powershell -ExecutionPolicy Bypass -File C:\CHAOS_PRODUCTION\runbooks\healthcheck_chaos.ps1

Write-Host "`n[ROUTER TEST]"
python C:\CHAOS_PRODUCTION\execution\ibkr_router.py

Write-Host "`n[IBKR CONNECTION TEST]"
python C:\CHAOS_PRODUCTION\test_ibkr_connection.py

Write-Host "`n=== PREAPPROVAL STACK TEST END ===" -ForegroundColor Green
