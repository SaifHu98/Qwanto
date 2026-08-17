#!/usr/bin/env python3
"""Measure qwnrun startup, persistent prefill, and persistent warm decode.

Each invocation emits one machine-readable evidence object.  The persistent
modes talk directly to the qwnrun line protocol and never substitute a
one-shot process measurement for warm runtime throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SCHEMA_VERSION = "5.0.0"
MODES = {"cold-start", "prefill", "warm-decode"}
CLASSIFICATIONS = {"MEASURED", "UNAVAILABLE", "INVALID"}
KEY_VALUE = re.compile(r"(?P<key>[a-z][a-z0-9_]*)=(?P<value>[^\s]+)", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return "file_not_found"
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return "file_unreadable"
    return digest.hexdigest()


def parse_key_values(text: str) -> dict[str, str | int | float | bool]:
    values: dict[str, str | int | float | bool] = {}
    for match in KEY_VALUE.finditer(text or ""):
        key, raw = match.group("key"), match.group("value")
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            values[key] = lowered == "true"
        else:
            try:
                values[key] = int(raw)
            except ValueError:
                try:
                    values[key] = float(raw)
                except ValueError:
                    values[key] = raw
    return values


def _number(values: dict, key: str) -> float | None:
    value = values.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def persistent_pid_proven(pids: list[int], process_pid: int, request_count: int) -> bool:
    return request_count >= 2 and len(pids) == request_count and all(pid == process_pid for pid in pids)


def cuda_execution_proven(values: dict) -> bool:
    return (values.get("backend_actual") == "cuda" and
            values.get("gpu_matmul_count", 0) > 0 and
            values.get("cpu_fallback_count", 0) == 0)


def parse_ready_stat(line: bytes) -> dict:
    fields = line.decode("utf-8", "replace").strip().split()
    if len(fields) < 7 or fields[0] != "STAT":
        raise ValueError(f"invalid qwnrun readiness STAT: {' '.join(fields)}")
    return {
        "completion_tokens": int(fields[1]),
        "tokens_per_second": float(fields[2]),
        "prompt_tokens": int(fields[5]),
        "length_limited": bool(int(fields[6])),
        **parse_key_values(" ".join(fields[7:])),
    }


def parse_done(line: bytes) -> dict:
    fields = line.decode("utf-8", "replace").strip().split()
    if len(fields) < 9 or fields[0] != "DONE" or fields[2] != "STAT":
        raise ValueError(f"invalid qwnrun DONE record: {' '.join(fields)}")
    return {
        "runtime_stat_line": line.decode("utf-8", "replace").strip(),
        "request_id": fields[1],
        "generated_tokens": int(fields[3]),
        "decode_tok_per_sec": float(fields[4]),
        "prompt_tokens": int(fields[7]),
        "length_limited": bool(int(fields[8])),
        **parse_key_values(" ".join(fields[9:])),
    }


def _read_with_timeout(stream, size: int | None, timeout: float):
    result: queue.Queue[bytes | BaseException] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            result.put(stream.readline() if size is None else stream.read(size))
        except BaseException as error:  # propagate stream failures to caller
            result.put(error)

    threading.Thread(target=read, daemon=True).start()
    value = result.get(timeout=timeout)
    if isinstance(value, BaseException):
        raise value
    return value


class PersistentQwnrun:
    def __init__(self, executable: Path, model: Path, backend: str, context_size: int,
                 max_tokens: int, threads: int | None, seed: int, timeout: float):
        command = [str(executable), str(model), "--serve", "--backend", backend,
                   "--ctx-size", str(context_size), "--max-tokens", str(max_tokens),
                   "--seed", str(seed)]
        if threads is not None:
            command += ["--threads", str(threads)]
        environment = os.environ.copy()
        environment["SERVE"] = "1"
        self.timeout = timeout
        self.started = time.perf_counter()
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=environment, bufsize=0,
        )
        self.process_create_ms = (time.perf_counter() - self.started) * 1000.0
        self.pid = self.process.pid
        try:
            ready = _read_with_timeout(self.process.stdout, None, timeout)
            if b"\x01\x01READY\x01\x01" not in ready:
                raise ValueError(f"qwnrun did not emit READY: {ready!r}")
            stat = _read_with_timeout(self.process.stdout, None, timeout)
            self.ready_stat = parse_ready_stat(stat)
            self.ready_ms = (time.perf_counter() - self.started) * 1000.0
        except Exception as error:
            _, stderr, returncode = self.close()
            detail = stderr.strip() or "no qwnrun stderr"
            raise ValueError(
                f"{error}; qwnrun_returncode={returncode}; qwnrun_stderr={detail}"
            ) from error

    def request(self, request_id: str, prompt: str, max_tokens: int,
                temperature: float = 0.0, top_p: float = 1.0) -> dict:
        payload = prompt.encode("utf-8")
        header = (f"SUBMIT {request_id} 0 {len(payload)} {max_tokens} "
                  f"{temperature:.8g} {top_p:.8g}\n").encode("ascii")
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("qwnrun protocol pipes are unavailable")
        self.process.stdin.write(header + payload + b"\n")
        self.process.stdin.flush()
        while True:
            line = _read_with_timeout(self.process.stdout, None, self.timeout)
            if not line:
                raise RuntimeError("qwnrun exited before completing the request")
            fields = line.decode("utf-8", "replace").strip().split()
            if not fields:
                continue
            if fields[0] == "DATA" and len(fields) == 3:
                size = int(fields[2])
                data = _read_with_timeout(self.process.stdout, size, self.timeout)
                terminator = _read_with_timeout(self.process.stdout, 1, self.timeout)
                if len(data) != size or terminator != b"\n":
                    raise RuntimeError("invalid qwnrun DATA frame")
            elif fields[0] == "DONE":
                return parse_done(line)
            elif fields[0] == "ERROR":
                raise RuntimeError("qwnrun request failed: " + " ".join(fields[2:]))

    def close(self) -> tuple[str, str, int | None]:
        if self.process.poll() is None:
            if self.process.stdin is not None:
                self.process.stdin.close()
                self.process.stdin = None
            try:
                stdout, stderr = self.process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.kill()
                stdout, stderr = self.process.communicate()
        else:
            stdout, stderr = self.process.communicate()
        return (
            stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else stdout or "",
            stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr or "",
            self.process.returncode,
        )


def resolve_executable(custom: str | None) -> Path | None:
    candidates = [
        Path(custom).expanduser() if custom else None,
        PROJECT_ROOT / "c" / "qwnrun_msvc.exe",
        PROJECT_ROOT / "c" / "qwnrun.exe",
        PROJECT_ROOT / "c" / "qwnrun",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    return None


def revision() -> dict[str, str | bool]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=False,
                                timeout=5).stdout.strip() or "Unavailable"
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True, check=False,
                                    timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "Unavailable", False
    version = "Unavailable"
    manifest = PROJECT_ROOT / "desktop" / "src-tauri" / "Cargo.toml"
    try:
        match = re.search(r'^version\s*=\s*"([^"]+)"', manifest.read_text(encoding="utf-8"), re.MULTILINE)
        if match:
            version = match.group(1)
    except OSError:
        pass
    return {"qwn_version": version, "git_commit": commit, "git_worktree_dirty": dirty}


def build_info(executable: Path, backend: str, threads: int | None) -> tuple[str, dict, int | None]:
    command = [str(executable), "--build-info", "--backend", backend]
    if threads is not None:
        command += ["--threads", str(threads)]
    try:
        result = subprocess.run(command, cwd=executable.parent, capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                check=False, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        return str(error), {}, None
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    return text.strip(), parse_key_values(text), result.returncode


def model_manifest_metadata(model: Path) -> dict[str, str]:
    """Resolve model identity from the checked-in manifest, when available."""
    result = {"model_id": "Unavailable", "architecture": "Unavailable",
              "qwn_dtype": "Unavailable"}
    manifest_path = PROJECT_ROOT / "docs" / "model-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return result
    model_hash = sha256_file(model)
    for candidate in manifest.get("models", []):
        target = candidate.get("target_file")
        target_path = (PROJECT_ROOT / target).resolve() if target else None
        if ((candidate.get("target_sha256") == model_hash) or
                (target_path is not None and target_path == model)):
            result.update({
                "model_id": candidate.get("model_id", "Unavailable"),
                "architecture": candidate.get("architecture", "Unavailable"),
                "qwn_dtype": candidate.get("quantization", "Unavailable"),
            })
            break
    return result


def base_report(mode: str, model: Path, executable: Path | None, prompt: str,
                context_size: int, max_tokens: int, backend: str,
                threads: int | None, seed: int, warmup_tokens: int) -> dict:
    model_hash = sha256_file(model)
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": f"qwn-phase-{time.time_ns()}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "evidence_classification": "UNAVAILABLE",
        "error_reason": None,
        "host_environment": {
            "os": f"{platform.system()} {platform.release()}",
            "cpu_model": platform.processor() or None,
            "cpu_threads": os.cpu_count() or 1,
        },
        "runtime_metadata": {
            **revision(),
            "executable_path": str(executable) if executable else None,
            "executable_sha256": sha256_file(executable) if executable else "file_not_found",
            "binary_sha256": sha256_file(executable) if executable else "file_not_found",
            "compiler": "Unavailable",
            "compiler_version": "Unavailable",
            "optimization_flags": "Unavailable",
            "openmp_compiled": "Unavailable",
            "openmp_runtime_loaded": "Unavailable",
            "requested_cpu_threads": threads if threads is not None else "auto",
            "active_cpu_threads": "Unavailable",
            "selected_cpu_isa_kernel": "Unavailable",
            "cpu_feature_detection": "Unavailable",
            "model_dtype": "Unavailable",
            "backend_requested": backend,
            "backend_actual": "Unavailable",
            "gpu_matmul_count": "Unavailable",
            "cpu_fallback_count": "Unavailable",
            "model_hash": model_hash,
        },
        "model_metadata": {"path": str(model), "sha256": model_hash,
                           "architecture": "Unavailable", "qwn_dtype": "Unavailable"},
        "benchmark_parameters": {
            "prompt": prompt,
            "context_size": context_size,
            "seed": seed,
            "max_tokens_requested": max_tokens,
            "warmup_tokens": warmup_tokens,
            "cpu_threads_requested": threads,
        },
        "execution_evidence": {
            "process_create_ms": None,
            "runtime_ready_ms": None,
            "process_pid": None,
            "same_pid_sequential_requests": False,
            "request_ids": [],
            "returncode": None,
            "stdout_sha256": None,
            "stderr_sha256": None,
        },
        "measurements": {},
        "unavailable_metrics": {},
    }


def finish(report: dict, classification: str, reason: str | None = None) -> dict:
    if classification not in CLASSIFICATIONS:
        raise ValueError(classification)
    report["evidence_classification"] = classification
    report["error_reason"] = reason
    if reason:
        report["unavailable_metrics"]["execution"] = reason
    return report


def run_phase_benchmark(model_path: str, mode: str, prompt: str, max_tokens: int,
                        custom_executable: str | None = None, backend: str = "cpu",
                        threads: int | None = None, context_size: int = 4096,
                        seed: int = 0, warmup_tokens: int = 4,
                        timeout: float = 120.0) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    model = Path(model_path).expanduser().resolve()
    executable = resolve_executable(custom_executable)
    report = base_report(mode, model, executable, prompt, context_size, max_tokens,
                         backend, threads, seed, warmup_tokens)
    report["model_metadata"].update(model_manifest_metadata(model))
    if not model.is_file():
        return finish(report, "UNAVAILABLE", f"QWN model file not found: {model}")
    if executable is None:
        return finish(report, "UNAVAILABLE", "qwnrun executable not found")
    build_text, build_fields, build_rc = build_info(executable, backend, threads)
    report["runtime_metadata"]["build_info"] = build_text
    report["runtime_metadata"]["build_info_returncode"] = build_rc
    for source, target in (
        ("compiler", "compiler"), ("compiler_version", "compiler_version"),
        ("optimization_flags", "optimization_flags"),
        ("binary_sha256", "binary_sha256"),
        ("openmp_enabled", "openmp_compiled"),
        ("openmp_runtime_loaded", "openmp_runtime_loaded"),
        ("requested_threads", "requested_cpu_threads"),
        ("hot_path_active_threads", "active_cpu_threads"),
        ("hot_path_isa_kernel", "selected_cpu_isa_kernel"),
    ):
        if source in build_fields:
            report["runtime_metadata"][target] = build_fields[source]
    features = [name for name in ("avx2", "f16c", "fma", "vnni")
                if build_fields.get(f"cpu_{name}") is True]
    if features:
        report["runtime_metadata"]["cpu_feature_detection"] = ",".join(features)

    runtime: PersistentQwnrun | None = None
    stdout = stderr = ""
    try:
        try:
            runtime = PersistentQwnrun(executable, model, backend, context_size,
                                       max_tokens, threads, seed, timeout)
        except (OSError, RuntimeError, ValueError, queue.Empty) as error:
            return finish(report, "UNAVAILABLE", f"persistent qwnrun startup failed: {error}")
        report["execution_evidence"].update({
            "process_create_ms": round(runtime.process_create_ms, 3),
            "process_pid": runtime.pid,
            "runtime_ready_ms": round(runtime.ready_ms, 3),
        })
        ready = runtime.ready_stat
        report["measurements"].update({
            "cold_start_ms": round(runtime.ready_ms, 3),
            "model_load_ms": _number(ready, "model_load_ms"),
            "runtime_ready_ms": _number(ready, "runtime_ready_ms"),
        })
        if mode == "cold-start":
            if _number(ready, "model_load_ms") is None or _number(ready, "runtime_ready_ms") is None:
                return finish(report, "UNAVAILABLE",
                               "qwnrun readiness STAT did not expose model_load_ms and runtime_ready_ms")
            return finish(report, "MEASURED")

        runtime.request("warmup", prompt, warmup_tokens)
        measured: list[dict] = []
        request_count = 1 if mode == "prefill" else 2
        for index in range(request_count):
            measured.append(runtime.request(f"{mode}-{index + 1}", prompt, max_tokens))
        pids = [int(item["pid"]) for item in measured if isinstance(item.get("pid"), (int, float))]
        same_pid = persistent_pid_proven(pids, runtime.pid, request_count) if mode == "warm-decode" else (
            bool(pids) and all(pid == runtime.pid for pid in pids))
        report["execution_evidence"].update({
            "same_pid_sequential_requests": same_pid and len(measured) >= request_count,
            "request_ids": [item["request_id"] for item in measured],
        })
        if not same_pid:
            return finish(report, "INVALID", "measured requests did not prove PID reuse")
        selected = measured[-1]
        report["measurements"].update({
            "prompt_tokens": selected.get("prompt_tokens"),
            "prefill_ms": selected.get("prefill_ms"),
            "prefill_tok_per_sec": selected.get("prefill_tok_per_sec"),
            "generated_tokens": selected.get("generated_tokens"),
            "first_token_ms": selected.get("first_token_ms"),
            "decode_wall_ms": selected.get("decode_wall_ms"),
            "decode_tok_per_sec": selected.get("decode_tok_per_sec"),
            "request_id": selected.get("request_id"),
            "runtime_stat_line": selected.get("runtime_stat_line"),
            "process_pid": runtime.pid,
            "sequential_request_count": len(measured),
        })
        if mode == "prefill":
            if not (_number(selected, "prefill_ms") and _number(selected, "prefill_tok_per_sec")):
                return finish(report, "UNAVAILABLE", "qwnrun did not report a positive prefill measurement")
        else:
            if not (_number(selected, "decode_wall_ms") and _number(selected, "decode_tok_per_sec") and
                    selected.get("generated_tokens", 0) > 0):
                return finish(report, "UNAVAILABLE", "qwnrun did not report a positive warm decode measurement")
        for key in ("backend_actual", "kernel", "gpu_matmul_count", "cpu_fallback_count", "active_threads"):
            if key in selected:
                target = "selected_cpu_isa_kernel" if key == "kernel" else (
                    "active_cpu_threads" if key == "active_threads" else key)
                report["runtime_metadata"][target] = selected[key]
        if backend == "cuda" and not cuda_execution_proven(selected):
            return finish(report, "INVALID",
                          "CUDA benchmark did not prove actual GPU matmul-only execution")
        return finish(report, "MEASURED")
    except (RuntimeError, ValueError, queue.Empty) as error:
        return finish(report, "UNAVAILABLE", f"persistent qwnrun measurement failed: {error}")
    finally:
        if runtime is not None:
            stdout, stderr, returncode = runtime.close()
            report["execution_evidence"].update({
                "returncode": returncode,
                "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
            })
            runtime_fields = parse_key_values(stderr)
            for source, target in (("model_dtype", "model_dtype"),
                                   ("backend_actual", "backend_actual"),
                                   ("gpu_matmul_count", "gpu_matmul_count"),
                                   ("cpu_fallback_count", "cpu_fallback_count"),
                                   ("active_threads", "active_cpu_threads"),
                                   ("hot_path_isa_kernel", "selected_cpu_isa_kernel")):
                if source in runtime_fields:
                    report["runtime_metadata"][target] = runtime_fields[source]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--model", default="experiments/results/4B_hyper_vsq2.qwn")
    parser.add_argument("--prompt", default="Explain zero-copy NVMe memory tiering in Qwanto.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--executable", default=None)
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = run_phase_benchmark(
        args.model, args.mode, args.prompt, args.max_tokens, args.executable,
        args.backend, args.threads, args.context_size, args.seed,
        args.warmup_tokens, args.timeout,
    )
    output = Path(args.output or f"benchmark_{args.mode.replace('-', '_')}.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "mode": report["mode"],
        "classification": report["evidence_classification"],
        "reason": report["error_reason"],
        "output": str(output.resolve()),
        "measurements": report["measurements"],
        "pid_reuse": report["execution_evidence"]["same_pid_sequential_requests"],
    }, indent=2))


if __name__ == "__main__":
    main()
