@echo off
cd /d "%~dp0"
python scripts/setup_demo.py
if errorlevel 1 (
  pause
  exit /b 1
)
set "SAFETY_MAIL_MODE=preview"
set "SAFETY_MAIL_PUBLIC_URL=http://127.0.0.1:8765"
python -m uvicorn app.server:app --host 127.0.0.1 --port 8765
pause
