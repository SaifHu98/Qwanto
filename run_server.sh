#!/bin/bash
# Qwanto/Colibri Quick Launcher (Linux/macOS/WSL)
# Edit MODEL_PATH below or export MODEL_PATH=/your/model/path

MODEL_PATH="${MODEL_PATH:-/mnt/nvme/glm52_i4}"
RAM_GB="${RAM_GB:-20}"
PORT="${PORT:-8000}"
AUTO_TIER="${AUTO_TIER:-1}"

echo ""
echo "  ██████╗ ███████╗████████╗███████╗ ██████╗ █████╗ ███╗   ██╗"
echo "  ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝██╔══██╗████╗  ██║"
echo "  ██████╔╝█████╗     ██║   █████╗  ██║     ███████║██╔██╗ ██║"
echo "  ██╔══██╗██╔══╝     ██║   ██╔══╝  ██║     ██╔══██║██║╚██╗██║"
echo "  ██║  ██║███████╗   ██║   ███████╗╚██████╗██║  ██║██║ ╚████║"
echo "  ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝"
echo ""
echo "  Qwanto / Colibri - Unified Inference Runtime"
echo "  CPU • GPU • RAM • NVMe - Measured, Not Assumed"
echo ""
echo "  Model:  $MODEL_PATH"
echo "  RAM:    ${RAM_GB} GB"
echo "  Port:   $PORT"
echo "  URL:    http://127.0.0.1:$PORT/"
echo ""

if [ ! -f "$MODEL_PATH/tokenizer.json" ]; then
    echo "[ERROR] Model not found at $MODEL_PATH"
    echo ""
    echo "Edit this file and set MODEL_PATH, or export MODEL_PATH=/your/path"
    echo "Download: https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp"
    echo "Convert:  cd c && ./qwanto convert --model \$MODEL_PATH --ebits 4 --io-bits 8"
    echo ""
    exit 1
fi

cd "$(dirname "$0")/c"

if [ ! -f "./glm" ]; then
    echo "[INFO] Building engine..."
    ./setup.sh
    echo ""
fi

echo "[INFO] Starting server with web dashboard..."
echo "[INFO] Press Ctrl+C to stop"
echo ""

if [ "$AUTO_TIER" = "1" ]; then
    ./coli web --model "$MODEL_PATH" --ram "$RAM_GB" --port "$PORT" --auto-tier
else
    ./coli web --model "$MODEL_PATH" --ram "$RAM_GB" --port "$PORT"
fi