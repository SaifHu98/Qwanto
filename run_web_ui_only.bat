@echo off
REM ==========================================
REM Qwanto Web UI Only - No Model Required
REM ==========================================
REM Serves the dashboard UI only. API calls will fail until you connect a backend.

cd /d "%~dp0"

echo.
echo   ██████╗ ███████╗████████╗███████╗ ██████╗ █████╗ ███╗   ██╗
echo   ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝██╔══██╗████╗  ██║
echo   ██████╔╝█████╗     ██║   █████╗  ██║     ███████║██╔██╗ ██║
echo   ██╔══██╗██╔══╝     ██║   ██╔══╝  ██║     ██╔══██║██║╚██╗██║
echo   ██║  ██║███████╗   ██║   ███████╗╚██████╗██║  ██║██║ ╚████║
echo   ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝
echo.
echo   Qwanto Web Dashboard (UI Only - No Model)
echo.

if not exist "web\dist\index.html" (
    echo [INFO] Building web UI...
    cd web
    npm install
    npm run build
    cd ..
    echo.
)

if not exist "web\dist\index.html" (
    echo [ERROR] Build failed - web\dist\index.html not found
    pause
    exit /b 1
)

echo [INFO] Starting web UI server on http://127.0.0.1:8080/
echo [INFO] Dashboard will load but show "Not connected" until you attach a backend.
echo [INFO] Press Ctrl+C to stop.
echo.

cd web\dist
python -m http.server 8080 --bind 127.0.0.1