@echo off
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
echo 首次启动程序可能需要耐心等待20秒
runtime\Scripts\python.exe realtime_gui.py
pause
