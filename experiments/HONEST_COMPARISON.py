"""Rebuild, convert, and measure the native QWN path without fake metrics."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "experiments" / "results" / "honest_comparison_latest.json"
QWN_RESULT = re.compile(r"qwnrun result: status=(\w+) tokens=(\d+)")


def run_command(argv, cwd=ROOT, timeout=900):
    try:
        completed = subprocess.run(
            [str(value) for value in argv], cwd=str(cwd), capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {"returncode": -1, "stdout": exc.stdout or "", "stderr": "timeout"}
    except OSError as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


def build_native():
    script = ROOT / "c" / "build_native.bat"
    if os.name != "nt":
        return {"status": "skipped", "reason": "native build script is Windows-only"}
    result = run_command(["cmd.exe", "/d", "/c", str(script)])
    return {"status": "ok" if result["returncode"] == 0 else "error",
            "returncode": result["returncode"], "stderr": result["stderr"][-2000:]}


def build_cuda():
    script = ROOT / "c" / "build_cuda.bat"
    if os.name != "nt":
        return {"status": "skipped", "reason": "CUDA build script is Windows-only"}
    result = run_command(["cmd.exe", "/d", "/c", str(script)])
    # Missing nvcc is an intentional CPU fallback and the script returns 0.
    return {"status": "ok" if result["returncode"] == 0 else "error",
            "returncode": result["returncode"], "stderr": result["stderr"][-2000:]}


def select_qwnrun():
    candidates = [ROOT / "c" / "qwnrun.exe", ROOT / "c" / "qwnrun",
                  ROOT / "c" / "qwnrun_omp.exe", ROOT / "c" / "qwnrun_clang.exe"]
    return next((path for path in candidates if path.exists()), None)


def convert_model(source: Path, output: Path):
    command = [sys.executable, str(ROOT / "c" / "coli"), "pack",
               str(source), str(output), "--quant", "q4_0"]
    return run_command(command, timeout=1800)


def gpu_snapshot():
    if shutil.which("nvidia-smi") is None:
        return {"available": False}
    result = run_command(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], timeout=10)
    rows = []
    for line in result["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            rows.append({"index": parts[0], "name": parts[1],
                         "utilization_percent": parts[2],
                         "memory_used_mib": parts[3], "memory_total_mib": parts[4]})
    return {"available": result["returncode"] == 0, "gpus": rows}


def load_llama_baselines():
    baselines = {}
    for filename in ("llama_15B.json", "llama_4B.json"):
        path = ROOT / "experiments" / "results" / filename
        if path.exists():
            try:
                baselines[filename] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                baselines[filename] = {"status": "error", "error": str(exc)}
    return baselines


def run_qwnrun(binary: Path, model: Path, max_tokens=4, ctx=2048):
    started = time.perf_counter()
    result = run_command([binary, model, "Hello", max_tokens, ctx], timeout=900)
    wall = time.perf_counter() - started
    combined = result["stdout"] + "\n" + result["stderr"]
    match = QWN_RESULT.search(combined)
    status = match.group(1) if match else "error"
    tokens = int(match.group(2)) if match else 0
    bad_output = any(marker in result["stdout"].lower()
                     for marker in ("nan", "inf", "�", "generation failed", "open error"))
    if result["returncode"] != 0 or status != "ok" or bad_output or tokens <= 0:
        tokens = 0
        status = "error"
    return {
        "status": status,
        "returncode": result["returncode"],
        "wall_seconds": wall,
        "tokens": tokens,
        "tok_per_sec": tokens / wall if tokens and wall > 0 else None,
        "stdout_preview": result["stdout"][:500],
        "stderr_preview": result["stderr"][:2000],
        "gpu_after": gpu_snapshot(),
    }


def main():
    build = {"native": build_native(), "cuda": build_cuda()}
    requested_binary = os.environ.get("QWANTO_BENCH_BINARY")
    binary = Path(requested_binary) if requested_binary else None
    if binary is None and build["native"].get("status") == "ok":
        binary = select_qwnrun()
    report = {"build": build, "binary": str(binary) if binary else None,
              "models": [], "llama_server_baselines": load_llama_baselines(),
              "pytest": None}
    if binary:
        sources = [ROOT / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
                   ROOT / "models" / "DeepSeek-V4-Pro-Qwen3.5-4B-MTP-BF16.gguf"]
        with tempfile.TemporaryDirectory(prefix="qwanto-honest-") as directory:
            for source in sources:
                item = {"source": str(source), "exists": source.exists()}
                if source.exists():
                    output = Path(directory) / (source.stem + ".qwn")
                    conversion = convert_model(source, output)
                    item["conversion"] = {"returncode": conversion["returncode"],
                                          "stderr": conversion["stderr"][-2000:]}
                    if conversion["returncode"] == 0 and output.exists():
                        item["qwn_bytes"] = output.stat().st_size
                        item["runs"] = [run_qwnrun(binary, output) for _ in range(3)]
                report["models"].append(item)
    else:
        report["error"] = "qwnrun binary was not built or found"

    tests = run_command([sys.executable, "-m", "pytest", "c/tests/", "-q"], timeout=1800)
    report["pytest"] = {"returncode": tests["returncode"],
                        "stdout": tests["stdout"][-4000:],
                        "stderr": tests["stderr"][-2000:]}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["pytest"]["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
