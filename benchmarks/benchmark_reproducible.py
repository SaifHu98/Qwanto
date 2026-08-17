#!/usr/bin/env python3
"""Run a real local qwnrun benchmark and emit auditable evidence.

The harness never supplies a fallback throughput, token count, hardware value,
or memory value. A run is ``MEASURED`` only when qwnrun exits successfully,
reports a valid positive token count, and the measured wall time is positive.
All other outcomes remain explicitly classified and contain no invented
performance metrics.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .runtime_config_snapshot import make_runtime_config_snapshot, update_runtime_config_snapshot
except ImportError:
    from runtime_config_snapshot import make_runtime_config_snapshot, update_runtime_config_snapshot

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SCHEMA_VERSION = "4.0.0"
CLASSIFICATIONS = {"MEASURED", "UNAVAILABLE", "INVALID", "TEST_FIXTURE", "EXPERIMENTAL", "PROJECTED"}

_STATUS_RE = re.compile(r"status=(?P<status>ok|error)\b(?:\s+tokens=(?P<tokens>-?\d+))?")
_GENERATED_RE = re.compile(r"Generated\s+Tokens\s*:\s*(?P<tokens>-?\d+)", re.IGNORECASE)
_TTFT_RE = re.compile(r"ttft_ms=(?P<ttft>[+-]?(?:\d+(?:\.\d*)?|\.\d+))")
_RUNTIME_FIELD_RE = re.compile(
    r"\b(?P<key>backend|backend_actual|kernel|kernel_requested|model_dtype|gpu_device|"
    r"gpu_matmul_count|cpu_fallback_count|cuda_upload_bytes|cuda_resident_bytes|cuda_dll_sha256|"
    r"active_threads|dispatch_reason|thinking_mode|decode_function|config_backend|context_size|"
    r"max_tokens|seed|kv_cache_mode|quantization|temperature|top_p|first_real_forward_ms|"
    r"file_open_ms|mmap_ms|metadata_parse_ms|tokenizer_init_ms|kv_cache_alloc_ms|"
    r"advisory_preload_ms|first_tensor_touch_ms|total_end_to_end_ms|prefill_ms|decode_wall_ms|"
    r"prefill_tok_per_sec|decode_tok_per_sec|generation_wall_ms|"
    r"sampling_ms)=(?P<value>[A-Za-z0-9_.;=+-]+)", re.IGNORECASE)
_KERNEL_SELECTED_RE = re.compile(r"kernel\s+selected:\s*(?P<kernel>[A-Za-z0-9_.-]+)", re.IGNORECASE)


def compute_file_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "file_not_found"
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(65536):
                digest.update(chunk)
    except OSError:
        return "file_unreadable"
    return digest.hexdigest()


def project_revision(host_hw: dict | None = None) -> dict[str, str | bool]:
    """Capture the source revision used to produce the evidence artifact."""
    if (host_hw or {}).get("os") == "TEST_FIXTURE":
        return {"qwn_version": "TEST_FIXTURE", "git_commit": "TEST_FIXTURE", "git_worktree_dirty": False}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, timeout=5, check=False,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, timeout=5, check=False,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "Unavailable", False
    version = "Unavailable"
    cargo_manifest = PROJECT_ROOT / "desktop" / "src-tauri" / "Cargo.toml"
    try:
        match = re.search(r"^version\s*=\s*\"([^\"]+)\"", cargo_manifest.read_text(encoding="utf-8"), re.MULTILINE)
        if match:
            version = match.group(1)
    except OSError:
        pass
    return {"qwn_version": version, "git_commit": commit or "Unavailable", "git_worktree_dirty": dirty}


def model_manifest_metadata(model_file: Path) -> dict[str, str]:
    """Resolve architecture and native dtype from the checked-in model manifest."""
    result = {"architecture": "Unavailable", "qwn_dtype": "Unavailable", "model_id": "Unavailable"}
    manifest_path = PROJECT_ROOT / "docs" / "model-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return result
    model_hash = compute_file_sha256(model_file)
    try:
        candidates = manifest.get("models", [])
        for candidate in candidates:
            target = candidate.get("target_file")
            target_hash = candidate.get("target_sha256")
            target_path = (PROJECT_ROOT / target).resolve() if target else None
            if (target_hash and target_hash == model_hash) or (target_path and target_path == model_file):
                result.update({
                    "architecture": candidate.get("architecture", "Unavailable"),
                    "qwn_dtype": candidate.get("quantization", "Unavailable"),
                    "model_id": candidate.get("model_id", "Unavailable"),
                })
                break
    except (AttributeError, TypeError):
        return result
    return result


def parse_build_info(build_info: str | None) -> dict[str, str | int | bool]:
    """Normalize qwnrun's key=value build record without treating detection as execution."""
    values: dict[str, str | int | bool] = {}
    for key, raw in re.findall(r"\b([a-z][a-z0-9_]*)=([A-Za-z0-9_.-]+)", build_info or ""):
        if raw.lower() in {"true", "false"}:
            values[key] = raw.lower() == "true"
        elif raw.isdigit():
            values[key] = int(raw)
        else:
            values[key] = raw
    return values


def detect_host_hardware() -> dict:
    cpu_model = platform.processor() or None
    cpu_threads = os.cpu_count() or 1
    total_ram_gb = None

    if sys.platform == "win32":
        try:
            class MemoryStatus(ctypes.Structure):
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

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total_ram_gb = round(status.ullTotalPhys / (1024 ** 3), 2)
        except Exception:
            total_ram_gb = None
    elif sys.platform.startswith("linux"):
        try:
            with Path("/proc/meminfo").open(encoding="utf-8") as stream:
                for line in stream:
                    if line.startswith("MemTotal:"):
                        total_ram_gb = round(int(line.split()[1]) / (1024 ** 2), 2)
                        break
        except (OSError, ValueError):
            total_ram_gb = None

    gpus = []
    gpu_detection_status = "not queried"
    if shutil.which("nvidia-smi"):
        gpu_detection_status = "queried"
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 3:
                        try:
                            vram_mb = float(parts[1])
                        except ValueError:
                            vram_mb = None
                        gpus.append({
                            "name": parts[0],
                            "vram_mb": vram_mb,
                            "driver_version": parts[2],
                            "vendor": "NVIDIA",
                        })
        except (OSError, subprocess.SubprocessError):
            gpu_detection_status = "nvidia-smi query failed"

    if not gpus and sys.platform == "win32" and shutil.which("wmic"):
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                names = [line.strip() for line in result.stdout.splitlines() if line.strip() and line.strip() != "Name"]
                for name in names:
                    vendor = "AMD" if "AMD" in name or "Radeon" in name else "Intel" if "Intel" in name else "Unknown"
                    gpus.append({"name": name, "vram_mb": None, "driver_version": None, "vendor": vendor})
                gpu_detection_status = "queried"
        except (OSError, subprocess.SubprocessError):
            gpu_detection_status = "wmic query failed"

    if not gpus and gpu_detection_status == "not queried":
        gpu_detection_status = "no supported GPU query tool available"

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu_model": cpu_model,
        "cpu_threads": cpu_threads,
        "ram_total_gb": total_ram_gb,
        "gpus_detected": gpus or None,
        "gpu_query_status": gpu_detection_status,
    }


def resolve_qwnrun_executable(custom_path: str | None = None) -> Path | None:
    if custom_path:
        candidate = Path(custom_path).expanduser()
        return candidate.resolve() if candidate.is_file() else None

    candidates = [
        PROJECT_ROOT / "c" / "qwnrun_msvc.exe",
        PROJECT_ROOT / "c" / "qwnrun.exe",
        PROJECT_ROOT / "c" / "qwnrun",
        Path("qwnrun.exe"),
        Path("qwnrun"),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    found = shutil.which("qwnrun")
    return Path(found).resolve() if found else None


def resolve_cuda_library(executable: Path | None) -> Path | None:
    """Resolve the adjacent CUDA sidecar using the runtime's deterministic path."""
    configured = os.environ.get("QWANTO_CUDA_DLL")
    if configured:
        candidate = Path(configured).expanduser()
        return candidate.resolve() if candidate.is_file() else None
    if executable:
        names = ("qwn_cuda.dll", "qwn_cuda.so", "qwn_cuda.dylib")
        for name in names:
            candidate = executable.parent / name
            if candidate.is_file():
                return candidate.resolve()
    return None


def parse_runtime_output(stdout: str, stderr: str) -> tuple[int | None, str | None]:
    """Parse qwnrun's one-shot result line without accepting guessed values."""
    matches = list(_STATUS_RE.finditer(stderr)) + list(_STATUS_RE.finditer(stdout))
    generated = list(_GENERATED_RE.finditer(stdout)) + list(_GENERATED_RE.finditer(stderr))

    if not matches:
        return None, "runtime output did not contain a status record"
    if len(matches) > 1 and {match.group("status") for match in matches} != {"ok"}:
        return None, "runtime emitted conflicting status records"
    if any(match.group("status") == "error" for match in matches):
        return None, "runtime reported generation failure"

    token_values = [int(match.group("tokens")) for match in matches if match.group("tokens") is not None]
    token_values.extend(int(match.group("tokens")) for match in generated)
    if not token_values:
        return None, "runtime output did not contain a token-count record"
    if len(set(token_values)) != 1:
        return None, "runtime emitted conflicting token counts"
    return token_values[0], None


def parse_ttft_ms(stdout: str, stderr: str) -> tuple[float | None, str | None]:
    """Read TTFT only when qwnrun exposes it in its measured result record."""
    values = [float(match.group("ttft")) for match in _TTFT_RE.finditer(stderr)]
    values.extend(float(match.group("ttft")) for match in _TTFT_RE.finditer(stdout))
    if not values:
        return None, None
    if len(set(values)) != 1:
        return None, "runtime emitted conflicting TTFT values"
    if values[0] < 0:
        return None, "runtime reported a negative TTFT"
    return values[0], None


def parse_runtime_metrics(stdout: str, stderr: str) -> dict:
    """Read only counters emitted by qwnrun; absent counters stay unavailable."""
    values = {}
    for match in _RUNTIME_FIELD_RE.finditer(stderr + "\n" + stdout):
        key, raw = match.group("key"), match.group("value")
        if key in {"backend", "kernel", "cuda_dll_sha256"}:
            values[key] = raw.lower() if key != "cuda_dll_sha256" else raw
        else:
            try:
                values[key] = int(raw)
            except ValueError:
                try:
                    values[key] = float(raw)
                except ValueError:
                    values[key] = raw
    selected_kernel = list(_KERNEL_SELECTED_RE.finditer(stderr + "\n" + stdout))
    if selected_kernel:
        selected = selected_kernel[-1].group("kernel").lower()
        values["kernel"] = "vnni" if selected == "avx-vnni" else selected
    return values


def _report_base(timestamp_utc: str, host_hw: dict, model_file: Path, executable: Path | None, cmd: list[str]) -> dict:
    cuda_library = resolve_cuda_library(executable)
    revision = project_revision(host_hw)
    model_info = model_manifest_metadata(model_file)
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": f"qwn-bench-{time.time_ns()}",
        "timestamp_utc": timestamp_utc,
        "evidence_classification": "UNAVAILABLE",
        "error_reason": None,
        "host_environment": host_hw,
        "runtime_metadata": {
            **revision,
            "executable_path": str(executable) if executable else None,
            "executable_sha256": compute_file_sha256(executable) if executable else "file_not_found",
            "cuda_library_path": str(cuda_library) if cuda_library else None,
            "cuda_dll_sha256": compute_file_sha256(cuda_library) if cuda_library else "Unavailable",
            "compiler": "Unavailable",
            "optimization_flags": "Unavailable",
            "openmp_enabled": "Unavailable",
            "active_thread_count": "Unavailable",
            "selected_cpu_isa_kernel": "Unavailable",
            "gpu_kernel_coverage": "Unavailable",
        },
        "model_metadata": {
            **model_info,
            "path": str(model_file),
            "file_size_bytes": model_file.stat().st_size if model_file.is_file() else None,
            "sha256": compute_file_sha256(model_file),
        },
        "benchmark_parameters": {
            "prompt": None,
            "prompt_length_chars": None,
            "max_tokens_requested": None,
            "command_argv": cmd,
        },
        "runtime_config_snapshot": make_runtime_config_snapshot(
            backend="cpu", context_size=4096, max_tokens=1, seed=0,
            prompt="", threads=None, warmup_tokens=0,
        ),
        "execution_evidence": {
            "returncode": None,
            "timed_out": False,
            "stdout_sha256": None,
            "stderr_sha256": None,
        },
        "measured_evidence": None,
        "unavailable_metrics": {
            "vram_allocated_gb": "NVML process polling inactive",
            "nvme_bandwidth_mb_s": "Direct block device counter inactive",
            "prefill_throughput_tok_s": "qwnrun did not report a prefill counter",
            "gpu_utilization": "Direct GPU utilization polling inactive",
        },
    }


def _finish_report(report: dict, classification: str, reason: str | None) -> dict:
    if classification not in CLASSIFICATIONS:
        raise ValueError(f"unknown evidence classification: {classification}")
    report["evidence_classification"] = classification
    report["error_reason"] = reason
    if classification != "MEASURED":
        report["unavailable_metrics"]["execution"] = reason or "No valid measured execution evidence"
    return report


def execute_real_benchmark(
    model_path: str,
    prompt: str,
    max_tokens: int = 64,
    custom_executable: str | None = None,
    timeout_seconds: float = 60.0,
    backend: str = "cpu",
    threads: int | None = None,
    context_size: int = 4096,
    seed: int = 0,
    warmup_tokens: int = 8,
) -> dict:
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    host_hw = detect_host_hardware()
    model_file = Path(model_path).expanduser().resolve()
    executable = resolve_qwnrun_executable(custom_executable)
    if backend not in {"cpu", "cuda", "auto"}:
        raise ValueError("backend must be cpu, cuda, or auto")
    cmd = [str(executable) if executable else "qwnrun", str(model_file), prompt,
           str(max_tokens), str(context_size), "--backend", backend,
           "--ctx-size", str(context_size), "--max-tokens", str(max_tokens),
           "--seed", str(seed), "--thinking", "none"]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    report = _report_base(timestamp_utc, host_hw, model_file, executable, cmd)
    report["benchmark_parameters"].update({
        "prompt": prompt,
        "prompt_length_chars": len(prompt),
        "max_tokens_requested": max_tokens,
        "backend_requested": backend,
        "context_size": context_size,
        "seed": seed,
        "warmup_tokens": warmup_tokens,
        "cpu_threads_requested": threads,
    })
    report["runtime_config_snapshot"] = make_runtime_config_snapshot(
        backend=backend, context_size=context_size, max_tokens=max_tokens,
        seed=seed, prompt=prompt, threads=threads, warmup_tokens=warmup_tokens,
    )

    if max_tokens <= 0 or not prompt:
        return _finish_report(report, "INVALID", "prompt must be non-empty and max_tokens must be positive")
    if executable is None:
        return _finish_report(report, "UNAVAILABLE", "qwnrun executable not found in the project or PATH")
    if not model_file.is_file():
        return _finish_report(report, "UNAVAILABLE", f"model container file not found: {model_file}")

    def run_command(command: list[str], timeout: float):
        started = time.perf_counter()
        try:
            environment = os.environ.copy()
            environment["QWANTO_TEMP"] = "0"
            environment["QWANTO_TOP_P"] = "1"
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, encoding="utf-8", errors="replace",
                                       env=environment)
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                stdout, stderr = process.communicate()
            return process, stdout or "", stderr or "", time.perf_counter() - started, timed_out
        except OSError as error:
            return None, "", str(error), 0.0, False

    build_process, build_stdout, build_stderr, _, _ = run_command([str(executable), "--build-info", "--json"], 10.0)
    report["runtime_metadata"]["build_info"] = (build_stdout + "\n" + build_stderr).strip() or None
    report["runtime_metadata"]["build_info_returncode"] = build_process.returncode if build_process else None
    build_fields = parse_build_info(report["runtime_metadata"]["build_info"])
    try:
        build_json = json.loads(build_stdout)
        if isinstance(build_json, dict):
            build_fields.update({key: value for key, value in build_json.items() if not isinstance(value, dict)})
            features = build_json.get("cpu_features", {})
            if isinstance(features, dict):
                build_fields.update({f"cpu_{key}": value for key, value in features.items()})
    except (json.JSONDecodeError, TypeError):
        pass
    report["runtime_metadata"].update({
        "compiler": build_fields.get("compiler", "Unavailable"),
        "optimization_flags": build_fields.get("optimization_flags", "Unavailable"),
        "openmp_enabled": build_fields.get("openmp_enabled", "Unavailable"),
        "active_thread_count": build_fields.get("active_threads", "Unavailable"),
        "selected_cpu_isa_kernel": build_fields.get(
            "selected_isa_kernel", build_fields.get("isa_backend", "Unavailable")),
        "gpu_kernel_coverage": build_fields.get("gpu_kernel_coverage", "Unavailable"),
    })

    if warmup_tokens > 0:
        warmup_cmd = list(cmd)
        max_index = warmup_cmd.index("--max-tokens") + 1
        warmup_cmd[max_index] = str(warmup_tokens)
        warmup_cmd[3] = str(warmup_tokens)
        warmup_process, _, warmup_stderr, _, warmup_timed_out = run_command(warmup_cmd, timeout_seconds)
        report["benchmark_parameters"]["warmup_returncode"] = warmup_process.returncode if warmup_process else None
        report["benchmark_parameters"]["warmup_timed_out"] = warmup_timed_out
        if warmup_process is None:
            return _finish_report(report, "UNAVAILABLE", "warmup qwnrun process could not be started")
        if warmup_timed_out:
            return _finish_report(report, "UNAVAILABLE", "warmup qwnrun execution timed out")
        if warmup_process.returncode != 0:
            return _finish_report(report, "INVALID", f"warmup qwnrun exited with status {warmup_process.returncode}")

    start_monotonic = time.perf_counter()
    timed_out = False
    process, stdout, stderr, wall_seconds, timed_out = run_command(cmd, timeout_seconds)
    if process is None:
        return _finish_report(report, "UNAVAILABLE", f"qwnrun process could not be started: {stderr}")

    stdout = stdout or ""
    stderr = stderr or ""
    report["execution_evidence"].update({
        "returncode": process.returncode,
        "timed_out": timed_out,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
    })

    if timed_out:
        return _finish_report(report, "UNAVAILABLE", f"qwnrun timed out after {timeout_seconds:g} seconds")
    if process.returncode != 0:
        return _finish_report(report, "INVALID", f"qwnrun exited with status {process.returncode}")

    runtime_metrics = parse_runtime_metrics(stdout, stderr)
    update_runtime_config_snapshot(report["runtime_config_snapshot"], runtime_metrics)
    report["measured_evidence"] = {
        "backend_requested": backend,
        "backend_actual": runtime_metrics.get("backend", "Unavailable"),
        "selected_kernel": runtime_metrics.get("kernel", report["runtime_metadata"]["selected_cpu_isa_kernel"]),
        "gpu_matmul_count": runtime_metrics.get("gpu_matmul_count", "Unavailable"),
        "cpu_fallback_count": runtime_metrics.get("cpu_fallback_count", "Unavailable"),
        "cuda_upload_bytes": runtime_metrics.get("cuda_upload_bytes", "Unavailable"),
        "cuda_resident_bytes": runtime_metrics.get("cuda_resident_bytes", "Unavailable"),
        "gpu_device": runtime_metrics.get("gpu_device", "Unavailable"),
        "cuda_dll_sha256": runtime_metrics.get("cuda_dll_sha256", report["runtime_metadata"]["cuda_dll_sha256"]),
        "runtime_config_snapshot": report["runtime_config_snapshot"],
    }
    if backend == "cuda" and (runtime_metrics.get("backend") != "cuda" or
                               runtime_metrics.get("gpu_matmul_count", 0) <= 0 or
                               runtime_metrics.get("cpu_fallback_count", 0) != 0):
        return _finish_report(report, "INVALID", "CUDA benchmark did not prove GPU matmul-only execution")

    generated_tokens, parse_error = parse_runtime_output(stdout, stderr)
    if parse_error:
        return _finish_report(report, "INVALID", parse_error)
    ttft_ms, ttft_error = parse_ttft_ms(stdout, stderr)
    if ttft_error:
        return _finish_report(report, "INVALID", ttft_error)
    if generated_tokens is None or generated_tokens <= 0:
        return _finish_report(report, "INVALID", "qwnrun reported zero or negative generated tokens")
    if wall_seconds <= 0:
        return _finish_report(report, "INVALID", "monotonic wall time was not positive")

    report["measured_evidence"].update({
        "generated_tokens": generated_tokens,
        "wall_seconds": round(wall_seconds, 6),
        "prefill_tok_per_sec": runtime_metrics.get("prefill_tok_per_sec", "Unavailable"),
        "decode_tok_per_sec": runtime_metrics.get(
            "decode_tok_per_sec", round(generated_tokens / wall_seconds, 6)),
        "tok_per_sec": round(generated_tokens / wall_seconds, 6),
        "ttft_ms": ttft_ms,
        "prompt_prefill_tok_s": "Unavailable",
        "gpu_utilization": "Unavailable",
        "vram_measured_bytes": runtime_metrics.get("cuda_resident_bytes", "Unavailable"),
        "file_open_ms": runtime_metrics.get("file_open_ms", "Unavailable"),
        "mmap_ms": runtime_metrics.get("mmap_ms", "Unavailable"),
        "metadata_parse_ms": runtime_metrics.get("metadata_parse_ms", "Unavailable"),
        "tokenizer_init_ms": runtime_metrics.get("tokenizer_init_ms", "Unavailable"),
        "kv_cache_alloc_ms": runtime_metrics.get("kv_cache_alloc_ms", "Unavailable"),
        "advisory_preload_ms": runtime_metrics.get("advisory_preload_ms", "Unavailable"),
        "first_tensor_touch_ms": runtime_metrics.get("first_tensor_touch_ms", "Unavailable"),
        "first_real_forward_ms": runtime_metrics.get("first_real_forward_ms", "Unavailable"),
        "prefill_ms": runtime_metrics.get("prefill_ms", "Unavailable"),
        "decode_wall_ms": runtime_metrics.get("decode_wall_ms", "Unavailable"),
        "sampling_ms": runtime_metrics.get("sampling_ms", "Unavailable"),
        "total_end_to_end_ms": runtime_metrics.get("total_end_to_end_ms", round(wall_seconds * 1000.0, 3)),
        "generation_wall_ms": runtime_metrics.get("generation_wall_ms", "Unavailable"),
    })
    return _finish_report(report, "MEASURED", None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="experiments/results/4B_hyper_vsq2.qwn", help="Path to a local .qwn model")
    parser.add_argument("--prompt", default="Explain zero-copy NVMe memory tiering in Qwanto.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--executable", default=None, help="Custom local qwnrun path")
    parser.add_argument("--output", default="benchmark_evidence.json", help="Evidence JSON output path")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    args = parser.parse_args()

    report = execute_real_benchmark(
        model_path=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        custom_executable=args.executable,
        timeout_seconds=args.timeout,
        backend=args.backend,
        threads=args.threads,
        context_size=args.context_size,
        seed=args.seed,
        warmup_tokens=args.warmup_tokens,
    )
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[STATUS] Classification: {report['evidence_classification']}")
    if report["measured_evidence"]:
        evidence = report["measured_evidence"]
        print(f"[MEASURED] Tokens Generated: {evidence['generated_tokens']}")
        print(f"[MEASURED] Wall Clock Time:  {evidence['wall_seconds']}s")
        print(f"[MEASURED] Throughput:       {evidence['tok_per_sec']} tok/s")
    else:
        print(f"[STATUS] Reason: {report['error_reason']}")
    print(f"[OUTPUT] Artifact written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
