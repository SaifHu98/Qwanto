import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "c" / "openai_server.py"


def _read_line(stream, output):
    output.put(stream.readline())


def test_gateway_sidecar_publishes_dynamic_ready_handshake(tmp_path):
    ready_file = tmp_path / "gateway.ready.json"
    env = os.environ.copy()
    env.update({
        "QWANTO_DISABLE_SETTINGS": "1",
        "QWANTO_MODEL_ROOT": str(tmp_path / "models"),
        "QWANTO_DESKTOP_SIDECAR": "1",
    })
    process = subprocess.Popen(
        [
            sys.executable,
            str(SERVER),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ready-file",
            str(ready_file),
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ready_queue = queue.Queue()
    reader = threading.Thread(target=_read_line, args=(process.stdout, ready_queue), daemon=True)
    reader.start()

    try:
        line = ready_queue.get(timeout=10)
        assert line.startswith("QWANTO_GATEWAY_READY "), line
        ready = json.loads(line.split(" ", 1)[1])
        assert ready["host"] == "127.0.0.1"
        assert ready["port"] > 0
        assert ready["url"].endswith(f":{ready['port']}")

        deadline = time.time() + 3
        while time.time() < deadline and not ready_file.exists():
            time.sleep(0.05)
        assert ready_file.exists()
        assert json.loads(ready_file.read_text(encoding="utf-8"))["port"] == ready["port"]

        with urllib.request.urlopen(f"{ready['url']}/health", timeout=3) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert health["gateway"] == "qwanto"
        assert health["status"] == "model_required"
        assert health["desktop_sidecar"] is True
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
