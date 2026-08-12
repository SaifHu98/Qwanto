@echo off
title Qwanto Server
echo.
echo ===================================
echo   Qwanto - Inference Runtime
echo ===================================
echo.

REM Kill any existing process on port 8000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING" 2^>nul') do (
    echo Killing PID %%a on port 8000...
    taskkill /F /PID %%a >nul 2>&1
)

REM Check if engine exists
if not exist "%~dp0c\glm.exe" (
    echo [ERROR] Engine not found: c\glm.exe
    echo Build it first with: cd c ^& make glm
    echo.
    pause
    exit /b 1
)

echo Starting Qwanto on http://127.0.0.1:8000/
echo Press Ctrl+C to stop
echo.

cd /d "%~dp0c"
python coli web --model "D:\models\glm52_i4" --ram 20 --port 8000 --auto-tier

echo.
echo Server stopped.
pause
