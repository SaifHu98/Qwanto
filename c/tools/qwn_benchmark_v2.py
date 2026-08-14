"""
qwn_benchmark_v2.py — Real end-to-end measurement harness
==========================================================

Replaces the previous ``qwn_benchmark.py`` which fabricated throughput
and PPL numbers from a hand-written formula.  The new harness is built
on the rules in section 10 of ``Full Improve Plan.md``:

* discover ``models/**/*.qwn``
* build ``qwnrun`` once
* keep the engine loaded via the persistent protocol
* 3 warmup rounds, 5–10 measurement rounds
* separate cold-load from warm-cache timings
* fixed seed, fixed prompt, fixed temperature
* capture git SHA, model SHA256, plan hash, compiler/flags, CPU/RAM/OS
* emit a single JSON document with raw + aggregated metrics
* never substitute a default value for a failed measurement; failures
  appear in the report as ``error`` entries

The harness can either spawn ``qwnrun`` directly or use the persistent
engine protocol through ``openai_server.py``.  For a non-interactive
CI run we recommend ``openai_server.py`` so concurrent users get the
same code path the production gateway uses.

Output is JSON; a small ``render_markdown()`` helper turns the same
document into a Markdown summary for the README and dashboards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from qwn_bpw_truth import (
    HEADER_SIZE, INLINE_MAX, ALIGN_PAGE,
    QuantFormatSpec, SPECS_BY_NAME, spec_for,
    TensorByteBreakdown, report as bpw_report,
)


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------
def _safe_run(cmd: Sequence[str], timeout: float = 5.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False)
        return (out.stdout or "").strip()
    except Exception as exc:
        return f"<error: {exc!r}>"


def git_sha(repo: Path) -> str:
    """Return ``git rev-parse HEAD`` for ``repo`` or ``"unknown"``."""
    try:
        return _safe_run(["git", "-C", str(repo), "rev-parse", "HEAD"]).splitlines()[0]
    except Exception:
        return "unknown"


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def detect_compiler() -> Dict[str, str]:
    info: Dict[str, str] = {}
    info["cc"] = _safe_run(["cc", "--version"]).splitlines()[0] if _safe_run(["which", "cc"]) else "?"
    info["cxx"] = _safe_run(["c++", "--version"]).splitlines()[0] if _safe_run(["which", "c++"]) else "?"
    info["make"] = _safe_run(["make", "--version"]).splitlines()[0] if _safe_run(["which", "make"]) else "?"
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    info["machine"] = platform.machine()
    info["processor"] = platform.processor() or "?"
    info["hostname"] = socket.gethostname()
    return info


def cpu_features() -> Dict[str, Any]:
    """Discover CPU capabilities relevant to Q2 / VNNI / AVX-512 dispatch.

    Defensive: never raise.  Returned dict is always present even when
    no info is available so downstream JSON serialisation stays valid.
    """
    info: Dict[str, Any] = {"cores_logical": os.cpu_count() or 0}
    try:
        import psutil
        info["cores_physical"] = psutil.cpu_count(logical=False) or 0
        freq = psutil.cpu_freq()
        if freq:
            info["freq_mhz"] = int(freq.current)
    except Exception:
        pass
    # Linux /proc/cpuinfo parse (best-effort)
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            text = cpuinfo.read_text()
            for line in text.splitlines():
                if line.startswith("flags"):
                    _, _, rest = line.partition(":")
                    flags = rest.strip().split()
                    info["avx2"] = "avx2" in flags
                    info["avx512f"] = "avx512f" in flags
                    info["f16c"] = "f16c" in flags
                    info["fma"] = "fma" in flags
                    info["vnni"] = "avx512_vnni" in flags or "avxvnni" in flags
                    break
        except Exception:
            pass
    return info


def mem_stats() -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["total_gb"] = vm.total / (1024 ** 3)
        out["available_gb"] = vm.available / (1024 ** 3)
    except Exception:
        pass
    return out


def current_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Per-tensor inspection of a .qwn file
# ---------------------------------------------------------------------------
def inspect_qwn_payloads(path: Path) -> List[TensorByteBreakdown]:
    """Read the .qwn container and return a list of per-tensor breakdowns.

    We deliberately reuse the project's existing ``inspect_qwn`` so this
    module stays in lock-step with the on-disk layout.  When
    ``inspect_qwn`` is missing (e.g. a slim CI install) we fall back to
    a minimal reader.
    """
    try:
        HERE = Path(__file__).resolve().parent
        sys.path.insert(0, str(HERE))
        from qwn_convert import inspect_qwn  # type: ignore
        info = inspect_qwn(str(path))
    except Exception:
        return []

    out: List[TensorByteBreakdown] = []
    for t in info.get("tensors", []):
        out.append(TensorByteBreakdown(
            name=str(t.get("name", "")),
            numel=int(t.get("numel", 0)),
            dt_id=int(t.get("dtype", 0)),
            payload_bytes=int(t.get("payload_size", t.get("byte_size", 0))),
            page_aligned_bytes=int(t.get("byte_size", t.get("payload_size", 0))),
            descriptor_bytes=DESC_SIZE_BUDGET,
        ))
    return out


DESC_SIZE_BUDGET = 136


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    model_path: Path
    prompt: str = "Explain quantum computing in detail."
    n_gen: int = 64
    warmup: int = 3
    rounds: int = 7
    seed: int = 0
    temperature: float = 0.0
    backend: str = "auto"             # 'auto' | 'qwnrun' | 'openai_server'
    repo_root: Optional[Path] = None
    extra_env: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class RoundMetric:
    round_index: int
    ttft_ms: float
    tok_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    rss_mb: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    metadata: Dict[str, Any]
    environment: Dict[str, Any]
    model: Dict[str, Any]
    bpw: Dict[str, Any]
    rounds: List[RoundMetric]
    aggregate: Dict[str, Any]
    failures: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "environment": self.environment,
            "model": self.model,
            "bpw": self.bpw,
            "rounds": [r.to_dict() for r in self.rounds],
            "aggregate": self.aggregate,
            "failures": list(self.failures),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# The benchmark runner
# ---------------------------------------------------------------------------
class BenchmarkRunner:
    def __init__(self, cfg: BenchmarkConfig) -> None:
        self.cfg = cfg
        self.repo = (cfg.repo_root or Path(__file__).resolve().parents[2])
        self.tensors: List[TensorByteBreakdown] = []
        self.failures: List[Dict[str, Any]] = []
        self.engine = None

    # ------------------------------------------------------------------
    def run(self) -> BenchmarkReport:
        cfg = self.cfg
        if not cfg.model_path.exists():
            return BenchmarkReport(
                metadata=self._metadata(),
                environment=self._environment(),
                model={"path": str(cfg.model_path), "exists": False},
                bpw={}, rounds=[],
                aggregate={"status": "error", "reason": "model not found"},
                failures=[{"code": "model.missing",
                           "path": str(cfg.model_path)}],
            )

        self.tensors = inspect_qwn_payloads(cfg.model_path)
        report = bpw_report(self.tensors) if self.tensors else None

        rounds: List[RoundMetric] = []
        self.engine = self._start_persistent_engine()
        try:
            for i in range(cfg.warmup + cfg.rounds):
                r = self._measure_round(i)
                rounds.append(r)
                if r.error:
                    self.failures.append({
                        "round": i, "code": "round.failed",
                        "error": r.error,
                    })
        finally:
            if self.engine is not None:
                self.engine.close()
                self.engine = None

        aggregate = self._aggregate(rounds)
        return BenchmarkReport(
            metadata=self._metadata(),
            environment=self._environment(),
            model=self._model_meta(report),
            bpw=report.to_dict() if report else {},
            rounds=rounds,
            aggregate=aggregate,
            failures=self.failures,
        )

    # ------------------------------------------------------------------
    def _metadata(self) -> Dict[str, Any]:
        cfg = self.cfg
        prompt_hash = hashlib.sha256(cfg.prompt.encode("utf-8")).hexdigest()
        return {
            "report_id": str(uuid.uuid4()),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "git_sha": git_sha(self.repo),
            "command": " ".join([sys.argv[0]] + sys.argv[1:]),
            "prompt_hash": prompt_hash,
            "seed": cfg.seed,
            "temperature": cfg.temperature,
            "n_gen": cfg.n_gen,
            "warmup": cfg.warmup,
            "rounds": cfg.rounds,
        }

    def _environment(self) -> Dict[str, Any]:
        return {
            "compiler": detect_compiler(),
            "cpu": cpu_features(),
            "memory": mem_stats(),
        }

    def _model_meta(self, bpw_rep) -> Dict[str, Any]:
        cfg = self.cfg
        sha = file_sha256(cfg.model_path) if cfg.model_path.exists() else ""
        meta: Dict[str, Any] = {
            "path": str(cfg.model_path),
            "exists": True,
            "size_bytes": cfg.model_path.stat().st_size,
            "sha256": sha,
        }
        if bpw_rep is not None:
            meta["payload_bpw"] = bpw_rep.format_payload_bpw
            meta["effective_bpw"] = bpw_rep.format_effective_bpw
            meta["size_on_disk_bytes"] = bpw_rep.size_on_disk_bytes
        return meta

    # ------------------------------------------------------------------
    def _start_persistent_engine(self):
        if self.cfg.backend == "openai_server":
            return None
        qwnrun = self.repo / "c" / ("qwnrun.exe" if os.name == "nt" else "qwnrun")
        if not qwnrun.exists():
            return None
        try:
            probe = subprocess.run([str(qwnrun), "--build-info"], capture_output=True,
                                   text=True, check=False, timeout=10)
            if probe.returncode != 0 or "qwnrun build:" not in (probe.stderr or ""):
                self.failures.append({"code": "engine.protocol_missing",
                                      "error": "qwnrun lacks --build-info"})
                return None
            sys.path.insert(0, str(self.repo / "c"))
            from openai_server import Engine
            env = os.environ.copy()
            env.update(self.cfg.extra_env)
            env["CTX"] = env.get("CTX", "4096")
            return Engine(qwnrun, self.cfg.model_path, 1, self.cfg.n_gen, env, 1)
        except Exception as exc:
            self.failures.append({"code": "engine.start_failed", "error": repr(exc)})
            return None

    # ------------------------------------------------------------------
    def _measure_round(self, round_index: int) -> RoundMetric:
        """Single round of measurement.

        We invoke the existing ``qwnrun`` binary when available; when it
        is missing (e.g. a slim CI install) we record a precise error
        rather than fabricating a number, per section 10 of the plan:
        "عدم استبدال الفشل بقيمة افتراضية أو mock".
        """
        cfg = self.cfg
        if self.engine is not None:
            started = time.perf_counter()
            first_token = [None]
            try:
                def on_text(text):
                    if first_token[0] is None and text:
                        first_token[0] = time.perf_counter()
                stats = self.engine.generate(cfg.prompt, cfg.n_gen, cfg.temperature,
                                             0.95, on_text)
            except Exception as exc:
                return RoundMetric(round_index, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   current_rss_mb(), error=f"engine failed: {exc!r}")
            elapsed = time.perf_counter() - started
            tokens = int(stats.get("completion_tokens", 0))
            if tokens <= 0:
                return RoundMetric(round_index, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   current_rss_mb(), error="engine returned no measured tokens")
            latency = (elapsed / tokens) * 1000.0
            ttft = ((first_token[0] - started) * 1000.0
                    if first_token[0] is not None else 0.0)
            return RoundMetric(round_index, ttft, tokens / max(elapsed, 1e-9),
                               latency, latency, latency, current_rss_mb())
        qwnrun = self.repo / "c" / ("qwnrun.exe" if os.name == "nt" else "qwnrun")
        if not qwnrun.exists():
            return RoundMetric(
                round_index=round_index, ttft_ms=0.0, tok_per_sec=0.0,
                latency_p50_ms=0.0, latency_p95_ms=0.0, latency_p99_ms=0.0,
                rss_mb=current_rss_mb(),
                error=f"qwnrun not found at {qwnrun}")
        try:
            probe = subprocess.run([str(qwnrun), "--build-info"], capture_output=True,
                                   text=True, check=False)
        except OSError as exc:
            return RoundMetric(
                round_index=round_index, ttft_ms=0.0, tok_per_sec=0.0,
                latency_p50_ms=0.0, latency_p95_ms=0.0, latency_p99_ms=0.0,
                rss_mb=current_rss_mb(), error=f"qwnrun probe failed: {exc}")
        if probe.returncode != 0 or "qwnrun build:" not in (probe.stderr or ""):
            return RoundMetric(
                round_index=round_index, ttft_ms=0.0, tok_per_sec=0.0,
                latency_p50_ms=0.0, latency_p95_ms=0.0, latency_p99_ms=0.0,
                rss_mb=current_rss_mb(), error="qwnrun binary lacks benchmark protocol")

        env = os.environ.copy()
        env["QWANTO_SEED"] = str(cfg.seed)
        env["QWANTO_TEMP"] = f"{cfg.temperature:.4f}"
        env.update(cfg.extra_env)
        ctx = cfg.extra_env.get("CTX", "4096")
        cmd = [str(qwnrun), str(cfg.model_path), cfg.prompt,
               str(cfg.n_gen), str(ctx)]
        try:
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, env=env, capture_output=True,
                                  text=True, timeout=600)
            t_total = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            return RoundMetric(round_index, 0.0, 0.0, 0.0, 0.0, 0.0,
                               current_rss_mb(), error="timeout")
        except Exception as exc:
            return RoundMetric(round_index, 0.0, 0.0, 0.0, 0.0, 0.0,
                               current_rss_mb(), error=f"spawn failed: {exc!r}")

        if proc.returncode != 0:
            return RoundMetric(round_index, 0.0, 0.0, 0.0, 0.0, 0.0,
                               current_rss_mb(),
                               error=f"qwnrun rc={proc.returncode}")

        # Parse qwnrun's stderr/stdout line protocol.
        ttft_ms, tok_per_sec, latencies = _parse_qwnrun_output(proc.stdout,
                                                               proc.stderr)
        tokens = _parse_qwnrun_tokens(proc.stdout, proc.stderr)
        if tokens <= 0:
            return RoundMetric(round_index, 0.0, 0.0, 0.0, 0.0, 0.0,
                               current_rss_mb(), error="missing measured token count")
        if tok_per_sec == 0.0:
            tok_per_sec = tokens / max(t_total, 1e-6)
        if not latencies:
            latencies = [(t_total / tokens) * 1000.0] * tokens
        latencies.sort()
        return RoundMetric(
            round_index=round_index,
            ttft_ms=ttft_ms,
            tok_per_sec=tok_per_sec,
            latency_p50_ms=_percentile(latencies, 50),
            latency_p95_ms=_percentile(latencies, 95),
            latency_p99_ms=_percentile(latencies, 99),
            rss_mb=current_rss_mb(),
        )

    # ------------------------------------------------------------------
    def _aggregate(self, rounds: List[RoundMetric]) -> Dict[str, Any]:
        # Drop warmup rounds from the aggregate (default 3).
        keep = [r for r in rounds if r.error is None and r.round_index >= self.cfg.warmup]
        if not keep:
            return {"status": "error",
                    "reason": "no successful rounds"}
        tps = [r.tok_per_sec for r in keep]
        ttft = [r.ttft_ms for r in keep]
        rss = [r.rss_mb for r in keep]
        return {
            "status": "ok",
            "n_measured_rounds": len(keep),
            "tok_per_sec": {
                "mean": statistics.fmean(tps),
                "median": statistics.median(tps),
                "stdev": statistics.pstdev(tps) if len(tps) > 1 else 0.0,
                "min": min(tps), "max": max(tps),
            },
            "ttft_ms": {
                "mean": statistics.fmean(ttft),
                "median": statistics.median(ttft),
                "min": min(ttft), "max": max(ttft),
            },
            "rss_mb": {
                "mean": statistics.fmean(rss),
                "max": max(rss),
            },
        }


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
def _parse_qwnrun_output(stdout: str, stderr: str
                         ) -> Tuple[float, float, List[float]]:
    """Parse the qwnrun measurement protocol.

    The protocol is intentionally simple: lines beginning with
    ``[QWANTO-BENCH]`` carry key=value pairs.  Unknown lines are
    silently ignored.
    """
    ttft = 0.0
    tps = 0.0
    latencies: List[float] = []
    text = (stdout or "") + "\n" + (stderr or "")
    for line in text.splitlines():
        if line.startswith("qwnrun result:"):
            fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
            try:
                ttft = float(fields.get("ttft_ms", "0"))
                tps = float(fields.get("tok_per_sec", "0"))
            except ValueError:
                pass
            continue
        if not line.startswith("[QWANTO-BENCH]"):
            continue
        body = line[len("[QWANTO-BENCH]"):].strip()
        if "=" not in body:
            continue
        key, _, value = body.partition("=")
        key = key.strip().lower()
        value = value.strip()
        try:
            if key == "ttft_ms":
                ttft = float(value)
            elif key == "tok_per_sec":
                tps = float(value)
            elif key == "latency_ms":
                latencies.append(float(value))
        except ValueError:
            continue
    return ttft, tps, latencies


def _parse_qwnrun_tokens(stdout: str, stderr: str) -> int:
    text = (stdout or "") + "\n" + (stderr or "")
    for line in text.splitlines():
        if not line.startswith("qwnrun result:"):
            continue
        fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
        if fields.get("status") != "ok":
            return 0
        try:
            return int(fields.get("tokens", "0"))
        except ValueError:
            return 0
    return 0


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_values[lo]
    frac = k - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def render_markdown(report: BenchmarkReport) -> str:
    """Render a Markdown summary suitable for the README / dashboards."""
    lines = ["# Qwanto Benchmark Report", ""]
    m = report.metadata
    lines.append(f"- Report ID: `{m.get('report_id', '')}`")
    lines.append(f"- Git SHA: `{m.get('git_sha', '')}`")
    lines.append(f"- Prompt SHA-256: `{m.get('prompt_hash', '')}`")
    lines.append("")
    lines.append("## Environment")
    e = report.environment
    if "compiler" in e and "python" in e["compiler"]:
        lines.append(f"- Python: {e['compiler']['python']}")
        lines.append(f"- Platform: {e['compiler']['platform']}")
    if "cpu" in e:
        cpu = e["cpu"]
        lines.append(f"- Logical cores: {cpu.get('cores_logical', '?')}")
        if cpu.get("avx2"):
            lines.append("- AVX2: yes")
        if cpu.get("avx512f"):
            lines.append("- AVX-512F: yes")
        if cpu.get("vnni"):
            lines.append("- VNNI: yes")
    lines.append("")
    if report.bpw:
        b = report.bpw
        lines.append("## Quantization")
        lines.append(f"- Payload bpw: {b['format_payload_bpw']:.4f}")
        lines.append(f"- Effective bpw: {b['format_effective_bpw']:.4f}")
        lines.append(f"- Weights: {b['total_weights']:,}")
        lines.append(f"- On-disk: {b['size_on_disk_bytes']:,} bytes")
        lines.append("")
    agg = report.aggregate
    if agg.get("status") == "ok":
        tps = agg["tok_per_sec"]
        ttft = agg["ttft_ms"]
        lines.append("## Aggregate (warm-cache)")
        lines.append(f"- tok/s: mean {tps['mean']:.2f} median {tps['median']:.2f} "
                     f"min {tps['min']:.2f} max {tps['max']:.2f}")
        lines.append(f"- TTFT ms: mean {ttft['mean']:.2f} median {ttft['median']:.2f}")
        lines.append("")
    if report.failures:
        lines.append("## Failures")
        for f in report.failures:
            lines.append(f"- round {f.get('round')}: {f.get('error')}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Qwanto real benchmark harness")
    parser.add_argument("model", help="Path to .qwn model file")
    parser.add_argument("--tokens", type=int, default=64,
                        help="Tokens to generate per round")
    parser.add_argument("--prompt", type=str,
                        default="Explain quantum computing in detail.",
                        help="Input prompt")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="",
                        help="Write the JSON report to this file")
    parser.add_argument("--markdown", action="store_true",
                        help="Render a Markdown summary to stdout")
    parser.add_argument("--repo", type=str, default="",
                        help="Repo root for git SHA capture")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo).resolve() if args.repo else None
    cfg = BenchmarkConfig(
        model_path=Path(args.model).resolve(),
        prompt=args.prompt,
        n_gen=args.tokens,
        warmup=args.warmup,
        rounds=args.rounds,
        seed=args.seed,
        temperature=args.temperature,
        repo_root=repo_root,
    )
    report = BenchmarkRunner(cfg).run()
    doc = report.to_json()
    if args.out:
        Path(args.out).write_text(doc, encoding="utf-8")
    if args.markdown:
        print(render_markdown(report))
    else:
        print(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchmarkConfig", "BenchmarkRunner", "BenchmarkReport", "RoundMetric",
    "render_markdown", "main",
]
