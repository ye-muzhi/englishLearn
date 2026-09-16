@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_app_windows.ps1"
if errorlevel 1 (
  echo.
  echo EnglishLearn could not start. See the message above.
  pause
)
