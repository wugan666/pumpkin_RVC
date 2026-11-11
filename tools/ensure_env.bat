@echo off
rem Prepares the Python environment in runtime\ on first use, then verifies it.
rem Called by the launchers from the project root; errorlevel 1 means failure.
rem Keep this file pure ASCII: cmd misreads batch files that contain multi-byte
rem text. Messages live in tools\msg\*.txt and are shown with "type".
if exist "runtime\Scripts\python.exe" goto have_python

rem Where the python.org / winget per-user and all-users installs put Python 3.12.
set "PY312_USER=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
set "PY312_ALL=%ProgramFiles%\Python312\python.exe"

call :find_python
if defined BOOT goto make_env

type tools\msg\no_python.txt
choice /c YN
if errorlevel 2 exit /b 1
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
rem A fresh install is not on this window's PATH yet, so look in the install folder.
call :find_python
if defined BOOT goto make_env
if exist "%PY312_USER%" goto installed_reopen
type tools\msg\python_install_failed.txt
exit /b 1

:installed_reopen
type tools\msg\python_installed.txt
exit /b 1

:make_env
type tools\msg\making_env.txt
%BOOT% -m venv runtime
if errorlevel 1 (
    type tools\msg\venv_failed.txt
    exit /b 1
)

:have_python
"runtime\Scripts\python.exe" tools\setup_env.py
exit /b %errorlevel%

:find_python
set "BOOT="
py -3.12 -c "import sys" >nul 2>&1 && set "BOOT=py -3.12"
if not defined BOOT python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1 && set "BOOT=python"
if not defined BOOT if exist "%PY312_USER%" "%PY312_USER%" -c "import sys" >nul 2>&1 && set BOOT="%PY312_USER%"
if not defined BOOT if exist "%PY312_ALL%" "%PY312_ALL%" -c "import sys" >nul 2>&1 && set BOOT="%PY312_ALL%"
exit /b 0
