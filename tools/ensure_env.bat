@echo off
rem Prepares the Python environment in runtime\ on first use, then verifies it.
rem Called by the launchers from the project root; errorlevel 1 means failure.
if exist "runtime\Scripts\python.exe" goto have_python

set "BOOT="
py -3.12 -c "import sys" >nul 2>&1 && set "BOOT=py -3.12"
if not defined BOOT python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1 && set "BOOT=python"
if defined BOOT goto make_env

echo.
echo 没有找到 Python 3.12。
echo 可以让本程序用 winget 自动安装，或者手动从 https://www.python.org/downloads/ 安装 Python 3.12 (64 位)。
choice /c YN /m "现在自动安装 Python 3.12 吗"
if errorlevel 2 exit /b 1
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo 自动安装失败，请手动安装 Python 3.12 后重新运行。
    exit /b 1
)
echo.
echo Python 已安装。请关闭本窗口，然后重新双击启动。
exit /b 1

:make_env
echo 正在创建运行环境 runtime ...
%BOOT% -m venv runtime
if errorlevel 1 (
    echo 创建运行环境失败。
    exit /b 1
)

:have_python
"runtime\Scripts\python.exe" tools\setup_env.py
exit /b %errorlevel%
