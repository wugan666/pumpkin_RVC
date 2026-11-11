@echo off
rem Keep this file pure ASCII; messages live in tools\msg\*.txt.
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
call tools\ensure_env.bat
if errorlevel 1 (
    pause
    exit /b 1
)
set "PATH=%SCRIPT_DIR%;%SCRIPT_DIR%\runtime\Scripts;%PATH%"
set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"
type tools\msg\first_start.txt
runtime\Scripts\python.exe realtime_gui.py
pause
