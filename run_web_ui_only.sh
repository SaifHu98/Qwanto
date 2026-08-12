#!/bin/bash
# Qwanto Web UI Only - No Model Required
# Serves the dashboard UI only. API calls will fail until you connect a backend.

cd "$(dirname "$0")"

echo ""
echo "  ██████╗ ███████╗████████╗███████╗ ██████╗ █████╗ ███╗   ██╗"
echo "  ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝██╔══██╗████╗  ██║"
echo "  ██████╔╝█████╗     ██║   █████╗  ██║     ███████║██╔██╗ ██║"
echo "  ██╔══██╗██╔══╝     ██║   ██╔══╝  ██║     ██╔══██║██║╚██╗██║"
echo "  ██║  ██║███████╗   ██║   ███████╗╚██████╗██║  ██║██║ ╚████║"
echo "  ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝"
echo ""
echo "  Qwanto Web Dashboard (UI Only - No Model)"
echo ""

if [ ! -f "web/dist/index.html" ]; then
    echo "[INFO] Building web UI..."
    cd web
    npm install
    npm run build
    cd ..
    echo ""
fi

if [ ! -f "web/dist/index.html" ]; then
    echo "[ERROR] Build failed - web/dist/index.html not found"
    exit 1
fi

echo "[INFO] Starting web UI server on http://127.0.0.1:8080/"
echo "[INFO] Dashboard will load but show 'Not connected' until you attach a backend."
echo "[INFO] Press Ctrl+C to stop."
echo ""

cd web/dist
python3 -m http.server 8080 --bind 127.0.0.1