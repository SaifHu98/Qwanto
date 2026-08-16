import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "c" / "openai_server.py"


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get_json(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        assert response.status == 200
        return json.loads(response.read().decode("utf-8"))


def test_local_gateway_exposes_stable_control_plane_endpoints(tmp_path):
    port = _free_port()
    env = os.environ.copy()
    env["QWANTO_DISABLE_SETTINGS"] = "1"
    process = subprocess.Popen(
        [
            sys.executable,
            str(SERVER),
            "--model",
            str(tmp_path / "missing.qwn"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    base = f"http://127.0.0.1:{port}"
    try:
        health = None
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                health = _get_json(f"{base}/health")
                break
            except (urllib.error.URLError, ConnectionError):
                if process.poll() is not None:
                    break
                time.sleep(0.1)

        if health is None:
            stdout, stderr = process.communicate(timeout=2)
            raise AssertionError(f"gateway did not start (exit={process.returncode})\nstdout={stdout}\nstderr={stderr}")

        models = _get_json(f"{base}/v1/models")
        config = _get_json(f"{base}/v1/qwanto/config")
        telemetry = _get_json(f"{base}/v1/qwanto/telemetry")

        assert health["gateway"] == "qwanto"
        assert health["api_version"] == "1"
        assert health["endpoints"]["models"] == "/v1/models"
        assert models["schema_version"] == "1"
        assert isinstance(models["data"], list)
        assert config["schema_version"] == "1"
        assert "backend" in config
        assert telemetry["schema_version"] == "1"
        assert "request_count" in telemetry
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)


def test_web_search_requires_the_desktop_approval_channel(tmp_path):
    port = _free_port()
    env = os.environ.copy()
    env["QWANTO_DISABLE_SETTINGS"] = "1"
    process = subprocess.Popen(
        [sys.executable, str(SERVER), "--model", str(tmp_path / "missing.qwn"), "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                _get_json(f"http://127.0.0.1:{port}/health")
                break
            except (urllib.error.URLError, ConnectionError):
                if process.poll() is not None:
                    break
                time.sleep(0.1)
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/qwanto/search",
            data=json.dumps({"query": "should not leave the desktop boundary"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=3)
        except urllib.error.HTTPError as error:
            assert error.code == 403
        else:
            raise AssertionError("web search must require the desktop approval channel")
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
