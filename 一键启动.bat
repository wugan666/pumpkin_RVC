@echo off
rem Keep every .bat file pure ASCII: cmd misreads batch files that contain
rem multi-byte text. Messages live in tools\msg\*.txt and are shown with "type".
chcp 65001 >nul
cd /d "%~dp0"
type tools\msg\menu.txt
choice /c 12 /n /t 15 /d 1 /m ">"
if errorlevel 2 (
    call go-webui.bat
) else (
    call go-realtime_gui.bat
)
