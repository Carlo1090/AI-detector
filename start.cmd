@echo off
rem Double-click to start DetectAI and open it in your browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The Python environment is missing. See README.md, "Run it".
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m api --open
pause
