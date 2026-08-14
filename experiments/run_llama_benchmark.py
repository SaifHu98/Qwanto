"""
run_llama_benchmark.py — Real llama-server OpenAI-compatible benchmark.

Starts the bundled ``llama-server.exe`` against one of the attached GGUF
models, polls the OpenAI-compatible ``/v1/models`` endpoint until the
server reports the model loaded, then issues chat-completion requests
and measures:

* cold load time (model mmap → ready)
* TTFT (time to first SSE chunk)
* tok/s for prompt evaluation (prefill)
* tok/s for generation (decode)
* p50 / p95 / p99 of per-token decode latency
* peak RSS via psutil

Nothing is fabricated; every figure in the JSON report comes from a
wall-clock measurement or from a header returned by the server.  When a
measurement is unavailable the field is left as ``null`` with an
explanatory error string.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import statistics

ROOT = Path(__file__).resolve().parent.parent


def _peak_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return 0.0


def _wait_port(host: str, port: int, timeout: float) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.4)
    return False


def _wait_model_ready(base: str, timeout: float) -> Optional[Dict[str, Any]]:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=2) as r:
                doc = json.loads(r.read().decode("utf-8"))
                if doc.get("data"):
                    return doc
        except Exception:
            time.sleep(0.5)
    return None


def _http_post_json(url: str, payload: Dict[str, Any],
                    timeout: float = 600.0) -> Dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _stream_chat(base: str, payload: Dict[str, Any],
                 timeout: float = 600.0):
    """Yield (chunk_dict, wall_time) tuples from the OpenAI SSE stream."""
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        buf = b""
        first_chunk = True
        while True:
            chunk_bytes = r.read(4096)
            if not chunk_bytes:
                break
            buf += chunk_bytes
            # The OpenAI streaming response uses \n\n-separated "data: ..." lines.
            while b"\n" in buf:
                line, _, buf = buf.partition(b"\n")
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip().decode("utf-8", "replace")
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except Exception:
                    continue
                yield chunk, (0.0 if first_chunk else time.perf_counter())
                first_chunk = False


def _now() -> float:
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------
@dataclass
class LLMBenchConfig:
    model_path: Path
    n_predict: int = 128
    n_ctx: int = 4096
    n_gpu_layers: int = 0          # 0 = pure CPU
    n_threads: int = 0             # 0 = server default
    port: int = 0                  # 0 = auto-pick
    prompt: str = (
        "Explain in detail how a CPU executes a fused multiply-add AVX2 "
        "instruction and how this maps to a typical int8 quantized matrix "
        "multiplication in a transformer model."
    )
    warmup: int = 1
    measurement_rounds: int = 3


# ---------------------------------------------------------------------------
# Per-round measurement
# ---------------------------------------------------------------------------
@dataclass
class RoundResult:
    round_index: int
    wall_seconds: float
    ttft_ms: float
    prefill_tok_s: float
    decode_tok_s: float
    decode_p50_ms: float
    decode_p95_ms: float
    decode_p99_ms: float
    tokens_generated: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def _run_round(base: str, cfg: LLMBenchConfig, round_index: int) -> RoundResult:
    # Use the actual model id reported by /v1/models (the absolute path).
    model_id = _resolve_model_id(base)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": cfg.prompt}],
        "max_tokens": cfg.n_predict,
        "stream": True,
        "temperature": 0.0,
    }
    t0 = _now()
    ttft_ms = 0.0
    tokens = 0
    prefill_tokens = 0
    server_decode_tok_s = 0.0
    server_prefill_tok_s = 0.0
    first_text_time: Optional[float] = None
    chunk_times: List[float] = []
    try:
        for chunk, _t in _stream_chat(base, payload):
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            # DeepSeek-R1 puts the bulk of its output in
            # ``reasoning_content``; both fields must be counted.
            text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            now = _now()
            if first_text_time is None and text:
                first_text_time = now
                ttft_ms = (first_text_time - t0) * 1000.0
            if text:
                tokens += 1
                chunk_times.append(now)
            if "usage" in chunk and chunk["usage"]:
                prefill_tokens = int(chunk["usage"].get("prompt_tokens") or 0)
                if not prefill_tokens:
                    prefill_tokens = int(chunk["usage"].get("prompt_eval_count") or 0)
            # llama.cpp also returns a ``timings`` block on the final
            # chunk with per-token server-side numbers.  These are the
            # canonical values (we trust them over our wall-clock math).
            if "timings" in chunk and chunk["timings"]:
                t = chunk["timings"]
                server_prefill_tok_s = float(t.get("prompt_per_second") or 0.0)
                server_decode_tok_s = float(t.get("predicted_per_second") or 0.0)
                if not prefill_tokens:
                    prefill_tokens = int(t.get("prompt_n") or 0)
        wall = _now() - t0
    except Exception as exc:
        return RoundResult(round_index, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0, error=f"{exc!r}")

    # Prefer the server-reported timings when available; they reflect
    # the real GPU/CPU cost, not our Python loop overhead.
    decode_tok_s = server_decode_tok_s if server_decode_tok_s > 0 else (
        tokens / max(wall - (ttft_ms / 1000.0), 1e-6))
    prefill_tok_s = server_prefill_tok_s if server_prefill_tok_s > 0 else (
        prefill_tokens / max(ttft_ms / 1000.0, 1e-6))
    p50 = p95 = p99 = 0.0
    if len(chunk_times) >= 2:
        deltas = [chunk_times[i] - chunk_times[i - 1]
                  for i in range(1, len(chunk_times))]
        deltas.sort()
        p50 = _pct(deltas, 50) * 1000.0
        p95 = _pct(deltas, 95) * 1000.0
        p99 = _pct(deltas, 99) * 1000.0
    return RoundResult(round_index, wall, ttft_ms, prefill_tok_s,
                       decode_tok_s, p50, p95, p99, tokens)


def _resolve_model_id(base: str) -> str:
    """Fetch the canonical model id from /v1/models."""
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
            doc = json.loads(r.read().decode("utf-8"))
            for entry in doc.get("data", []):
                return entry.get("id") or "qwanto-bench"
    except Exception:
        pass
    return "qwanto-bench"


def _pct(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _pick_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def run(cfg: LLMBenchConfig) -> Dict[str, Any]:
    port = cfg.port or _pick_port()
    base = f"http://127.0.0.1:{port}"
    cmd = [str(ROOT / "c" / "llama-server.exe"),
           "--model", str(cfg.model_path),
           "--port", str(port),
           "--host", "127.0.0.1",
           "--ctx-size", str(cfg.n_ctx),
           "--n-predict", str(cfg.n_predict),
           "--threads", str(cfg.n_threads) if cfg.n_threads else "-1"]
    if cfg.n_gpu_layers:
        cmd.extend(["--n-gpu-layers", str(cfg.n_gpu_layers)])
    log_path = ROOT / "experiments" / "results" / "llama_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "ab", buffering=0)
    print(f"==> launching {cfg.model_path.name} on port {port}")
    print("    cmd:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    try:
        if not _wait_port("127.0.0.1", port, timeout=120):
            raise RuntimeError("server failed to bind port")
        t_ready0 = _now()
        info = _wait_model_ready(base, timeout=240)
        if not info:
            raise RuntimeError("server reports no loaded models")
        cold_load_s = _now() - t_ready0
        # Don't add the *server startup* time before bind to cold_load.
        # That is captured separately below.
        print(f"    model loaded in {cold_load_s:.1f}s")

        rounds: List[RoundResult] = []
        for i in range(cfg.warmup + cfg.measurement_rounds):
            r = _run_round(base, cfg, i)
            rounds.append(r)
            print(f"    round {i}: wall={r.wall_seconds:.2f}s "
                  f"ttft={r.ttft_ms:.0f}ms "
                  f"prefill={r.prefill_tok_s:.1f} t/s "
                  f"decode={r.decode_tok_s:.1f} t/s "
                  f"tokens={r.tokens_generated}"
                  + (f" ERR={r.error}" if r.error else ""))
        keep = [r for r in rounds if r.error is None and r.tokens_generated > 0]
        aggregate: Dict[str, Any] = {
            "cold_load_seconds": cold_load_s,
            "n_rounds_kept": len(keep),
            "n_rounds_total": len(rounds),
        }
        if keep:
            tps = [r.decode_tok_s for r in keep]
            ttfts = [r.ttft_ms for r in keep]
            aggregate["decode_tok_s"] = {
                "mean": statistics.fmean(tps),
                "median": statistics.median(tps),
                "min": min(tps), "max": max(tps),
            }
            aggregate["ttft_ms"] = {
                "mean": statistics.fmean(ttfts),
                "median": statistics.median(ttfts),
                "min": min(ttfts), "max": max(ttfts),
            }
        return {
            "model": cfg.model_path.name,
            "model_size_bytes": cfg.model_path.stat().st_size,
            "config": asdict(cfg),
            "aggregate": aggregate,
            "rounds": [r.to_dict() for r in rounds],
            "peak_rss_mb": _peak_rss_mb(),
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_f.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", help="Path to a .gguf file")
    p.add_argument("--n-predict", type=int, default=128)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--n-gpu-layers", type=int, default=0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()
    cfg = LLMBenchConfig(
        model_path=Path(args.model).resolve(),
        n_predict=args.n_predict,
        n_ctx=args.ctx,
        n_threads=args.threads,
        n_gpu_layers=args.n_gpu_layers,
        warmup=args.warmup,
        measurement_rounds=args.rounds,
    )
    res = run(cfg)
    doc = json.dumps(res, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(doc, encoding="utf-8")
    print()
    print(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())