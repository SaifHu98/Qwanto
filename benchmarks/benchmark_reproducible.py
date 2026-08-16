#!/usr/bin/env python3
"""
Qwanto Evidence-Producing Benchmark Harness
Executes the native local Qwanto runtime, records real monotonic timings,
and produces a verifiable, machine-readable evidence artifact.
Zero hardcoded performance or hardware values.
"""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

def compute_file_sha256(path: Path) -> str:
    if not path.exists():
        return "file_not_found"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def detect_host_hardware() -> dict:
    cpu_model = platform.processor() or "Unknown CPU"
    cpu_threads = os.cpu_count() or 1

    # Detect physical RAM
    total_ram_gb = None
    if sys.platform == "win32":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_ram_gb = round(stat.ullTotalPhys / (1024 ** 3), 2)
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        total_ram_gb = round(kb / (1024 ** 2), 2)
                        break
        except Exception:
            pass

    # Detect GPU hardware dynamically via nvidia-smi / local tools
    gpus = []
    gpu_detection_status = "queried"
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if res.returncode == 0:
                for line in res.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append({
                            "name": parts[0],
                            "vram_mb": float(parts[1]),
                            "driver_version": parts[2],
                            "vendor": "NVIDIA"
                        })
        except Exception as e:
            gpu_detection_status = f"nvidia-smi query error: {e}"
    
    if not gpus and sys.platform == "win32":
        try:
            res = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if res.returncode == 0:
                names = [l.strip() for l in res.stdout.splitlines() if l.strip() and l.strip() != "Name"]
                for n in names:
                    gpus.append({
                        "name": n,
                        "vram_mb": None,
                        "driver_version": None,
                        "vendor": "AMD" if "AMD" in n or "Radeon" in n else ("Intel" if "Intel" in n else "Unknown")
                    })
        except Exception:
            pass

    if not gpus:
        gpu_detection_status = "No dedicated GPU querying tool available"

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu_model": cpu_model,
        "cpu_threads": cpu_threads,
        "ram_total_gb": total_ram_gb,
        "gpus_detected": gpus if gpus else None,
        "gpu_query_status": gpu_detection_status
    }

def resolve_qwnrun_executable(custom_path: str = None) -> Path | None:
    candidates = [
        Path(custom_path) if custom_path else None,
        PROJECT_ROOT / "c" / "qwnrun_msvc.exe",
        PROJECT_ROOT / "c" / "qwnrun.exe",
        PROJECT_ROOT / "c" / "qwnrun",
        Path("qwnrun.exe"),
        Path("qwnrun"),
    ]
    for c in candidates:
        if c and c.exists():
            return c.resolve()
    which_exe = shutil.which("qwnrun")
    return Path(which_exe).resolve() if which_exe else None

def execute_real_benchmark(
    model_path: str,
    prompt: str,
    max_tokens: int = 64,
    custom_executable: str = None
) -> dict:
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    host_hw = detect_host_hardware()

    model_file = Path(model_path).resolve()
    executable = resolve_qwnrun_executable(custom_executable)

    unavailable_metrics = {}

    if not executable or not executable.exists():
        return {
            "schema_version": "2.0.0",
            "benchmark_id": f"qwn-bench-err-{int(time.time())}",
            "timestamp_utc": timestamp_utc,
            "evidence_classification": "UNAVAILABLE",
            "error_reason": f"qwnrun executable not found (checked project and PATH)",
            "host_environment": host_hw,
            "measured_evidence": None,
            "unavailable_metrics": {"all": "Runtime executable unavailable"}
        }

    if not model_file.exists():
        return {
            "schema_version": "2.0.0",
            "benchmark_id": f"qwn-bench-err-{int(time.time())}",
            "timestamp_utc": timestamp_utc,
            "evidence_classification": "UNAVAILABLE",
            "error_reason": f"Model container file not found: {model_file}",
            "host_environment": host_hw,
            "measured_evidence": None,
            "unavailable_metrics": {"all": "Model container file unavailable"}
        }

    model_size = model_file.stat().st_size
    model_sha256 = compute_file_sha256(model_file)

    # Launch actual qwnrun process in one-shot / benchmark mode
    cmd = [
        str(executable),
        str(model_file),
        prompt,
        str(max_tokens),
        "4096"
    ]

    start_monotonic = time.perf_counter()
    first_token_monotonic = None
    generated_tokens = 0
    raw_stdout = []
    raw_stderr = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Read stdout in real-time to capture first payload time
        for line in proc.stdout:
            raw_stdout.append(line)
            if first_token_monotonic is None and line.strip() and not line.startswith("Prompt tokens:"):
                first_token_monotonic = time.perf_counter()
            if "Generated Tokens :" in line or "status=ok tokens=" in line:
                # Parse generated token count from runtime output
                for part in line.split():
                    if part.isdigit():
                        generated_tokens = int(part)
                        break

        proc.wait(timeout=60)
        end_monotonic = time.perf_counter()
        stderr_output = proc.stderr.read()
        raw_stderr.append(stderr_output)

        wall_seconds = end_monotonic - start_monotonic
        ttft_ms = (first_token_monotonic - start_monotonic) * 1000.0 if first_token_monotonic else None

        # Parse tokens from stderr diagnostics if not found in stdout
        if generated_tokens == 0:
            for line in stderr_output.splitlines():
                if "tokens=" in line:
                    for part in line.split():
                        if part.startswith("tokens="):
                            try:
                                generated_tokens = int(part.split("=")[1])
                            except ValueError:
                                pass

        tok_per_sec = (generated_tokens / wall_seconds) if (wall_seconds > 0 and generated_tokens > 0) else None

        stdout_full = "".join(raw_stdout)
        stderr_full = "".join(raw_stderr)

        return {
            "schema_version": "2.0.0",
            "benchmark_id": f"qwn-bench-{int(time.time())}",
            "timestamp_utc": timestamp_utc,
            "evidence_classification": "MEASURED",
            "host_environment": host_hw,
            "runtime_metadata": {
                "executable_path": str(executable),
                "executable_sha256": compute_file_sha256(executable)
            },
            "model_metadata": {
                "path": str(model_file),
                "file_size_bytes": model_size,
                "sha256": model_sha256
            },
            "benchmark_parameters": {
                "prompt_length_chars": len(prompt),
                "max_tokens_requested": max_tokens,
                "command_argv": cmd
            },
            "measured_evidence": {
                "generated_tokens": generated_tokens,
                "wall_seconds": round(wall_seconds, 4),
                "tok_per_sec": round(tok_per_sec, 2) if tok_per_sec is not None else None,
                "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
                "stdout_sha256": hashlib.sha256(stdout_full.encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr_full.encode("utf-8")).hexdigest()
            },
            "unavailable_metrics": {
                "vram_allocated_gb": "NVML process polling inactive",
                "nvme_bandwidth_mb_s": "Direct block device counter inactive"
            }
        }
    except Exception as e:
        return {
            "schema_version": "2.0.0",
            "benchmark_id": f"qwn-bench-err-{int(time.time())}",
            "timestamp_utc": timestamp_utc,
            "evidence_classification": "UNAVAILABLE",
            "error_reason": f"Process execution failed: {e}",
            "host_environment": host_hw,
            "measured_evidence": None,
            "unavailable_metrics": {"execution": str(e)}
        }

def main():
    parser = argparse.ArgumentParser(description="Qwanto Evidence-Producing Benchmark Harness")
    parser.add_argument("--model", default="experiments/results/4B_hyper_vsq2.qwn", help="Path to .qwn model file")
    parser.add_argument("--prompt", default="Explain zero-copy NVMe memory tiering in Qwanto.", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--executable", default=None, help="Custom path to qwnrun binary")
    parser.add_argument("--output", default="benchmark_evidence.json", help="Output evidence JSON path")
    args = parser.parse_args()

    print("=================================================================", file=sys.stderr)
    print(">> QWANTO EVIDENCE-PRODUCING BENCHMARK HARNESS (EMPIRICAL MEASUREMENT)", file=sys.stderr)
    print("=================================================================", file=sys.stderr)

    report = execute_real_benchmark(
        model_path=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        custom_executable=args.executable
    )

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    classification = report.get("evidence_classification")
    print(f"\n[STATUS] Classification: {classification}")
    if classification == "MEASURED":
        meas = report["measured_evidence"]
        print(f"[MEASURED] Tokens Generated: {meas['generated_tokens']}")
        print(f"[MEASURED] Wall Clock Time:  {meas['wall_seconds']}s")
        print(f"[MEASURED] Throughput:       {meas['tok_per_sec']} tok/s")
        print(f"[MEASURED] TTFT:             {meas['ttft_ms']} ms")
    else:
        print(f"[BLOCKED] Reason: {report.get('error_reason')}")

    print(f"[OUTPUT] Artifact written to {out_path.resolve()}")

if __name__ == "__main__":
    main()
