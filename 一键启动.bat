@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   RVC 变声器
echo ============================================
echo   [1] 实时变声（打游戏 / 语音聊天）
echo   [2] 克隆音色 / 网页界面
echo.
echo   15 秒内不选择将自动进入 [1]
echo ============================================
choice /c 12 /n /t 15 /d 1 /m "请输入 1 或 2: "
if errorlevel 2 (
    call go-webui.bat
) else (
    call go-realtime_gui.bat
)
