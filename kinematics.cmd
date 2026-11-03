@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create .venv and install requirements.txt first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "scripts\interactive\kinematics_viewer.py" %*
if errorlevel 1 pause
