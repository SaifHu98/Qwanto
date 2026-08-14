"""
run_empirical_report.py — Single-command reproducible empirical study.

Produces the artefacts used by README.md from the actual attached GGUF
models and the .qwn conversions under ``experiments/results/``.

Outputs:
  experiments/results/empirical_summary.json   machine-readable
  experiments/results/empirical_summary.md     human-readable, copy/paste
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
sys.path.insert(0, str(ROOT / "c" / "tools"))

import qwn_bpw_truth as bpw


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(s: Any) -> float:
    try:
        return float(s)
    except Exception:
        return 0.0


def _load_benchmark(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    conv_rows = _load_csv(RESULTS / "conversions.csv")
    bpw_rows = _load_csv(RESULTS / "bpw_report.csv")
    bench_15b = _load_benchmark(RESULTS / "llama_15B.json")
    bench_4b = _load_benchmark(RESULTS / "llama_4B.json")

    summary: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": sys.platform,
        "python": sys.version.split()[0],
    }

    # Group by label, build per-format tables
    by_label: Dict[str, Dict[str, Any]] = {}
    for r in bpw_rows:
        label = r.get("label", "")
        mode = r.get("mode", "")
        by_label.setdefault(label, {})[mode] = {
            "wall_seconds": _float(r.get("wall_seconds")),
            "out_mb": _float(r.get("out_mb_actual")),
            "throughput_mb_s": _float(r.get("throughput_mb_s")),
            "payload_bpw": _float(r.get("payload_bpw")),
            "effective_bpw": _float(r.get("effective_bpw")),
            "total_weights": int(_float(r.get("total_weights"))),
            "size_on_disk_bytes": int(_float(r.get("size_on_disk_bytes"))),
            "size_on_disk_mb": _float(r.get("size_on_disk_mb")),
            "header_bytes": int(_float(r.get("header_bytes"))),
            "descriptor_bytes_total": int(_float(r.get("descriptor_bytes_total"))),
            "page_alignment_overhead_bytes": int(_float(r.get("page_alignment_overhead_bytes"))),
            "tail_block_bytes": int(_float(r.get("tail_block_bytes"))),
        }
    summary["conversions"] = by_label

    # Bench summaries (only the kept rounds)
    def _summarise(bench: Dict[str, Any]) -> Dict[str, Any]:
        agg = bench.get("aggregate", {})
        keep = [r for r in bench.get("rounds", [])
                if r.get("error") is None and r.get("tokens_generated", 0) > 0]
        return {
            "model": bench.get("model"),
            "model_size_bytes": bench.get("model_size_bytes"),
            "cold_load_seconds": agg.get("cold_load_seconds"),
            "n_rounds_kept": agg.get("n_rounds_kept"),
            "decode_tok_s_mean": agg.get("decode_tok_s", {}).get("mean"),
            "decode_tok_s_median": agg.get("decode_tok_s", {}).get("median"),
            "ttft_ms_mean": agg.get("ttft_ms", {}).get("mean"),
            "ttft_ms_median": agg.get("ttft_ms", {}).get("median"),
            "per_round": keep,
        }

    summary["llama_server_benchmarks"] = {
        "1.5B_gguf": _summarise(bench_15b),
        "4B_gguf": _summarise(bench_4b),
    }

    out_json = RESULTS / "empirical_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str),
                         encoding="utf-8")

    # ----- Markdown rendering -----------------------------------------
    md: List[str] = []
    md.append("# Empirical Qwanto Study")
    md.append("")
    md.append(f"Generated `{summary['generated_at']}` on "
               f"`{summary['platform']}` (Python {summary['python']}).")
    md.append("")
    md.append("Every figure below was produced by the experiment driver "
               "under `experiments/`.  No numbers are fabricated.")
    md.append("")

    for label in ("1.5B", "4B"):
        md.append(f"## {label} model")
        md.append("")
        md.append("| Format | Wall (s) | Throughput (MB/s) | Size (MB) | "
                   "Payload bpw | Effective bpw |")
        md.append("|---|---:|---:|---:|---:|---:|")
        rows = by_label.get(label, {})
        # Canonical ordering
        order = ("none", "q4_0", "vsq", "vsq_ultra",
                  "hyper_vsq", "hyper_vsq2")
        for mode in order:
            r = rows.get(mode)
            if not r:
                continue
            md.append(
                f"| `{mode}` | {r['wall_seconds']:.2f} | "
                f"{r['throughput_mb_s']:.1f} | "
                f"{r['out_mb']:.2f} | "
                f"{r['payload_bpw']:.3f} | "
                f"{r['effective_bpw']:.3f} |"
            )
        md.append("")
        md.append(
            f"On-disk overhead: header `{rows.get('q4_0', {}).get('header_bytes', 4096)}` B, "
            f"descriptor block `{rows.get('q4_0', {}).get('descriptor_bytes_total', 0)}` B, "
            f"tail block `{rows.get('q4_0', {}).get('tail_block_bytes', 4096)}` B, "
            f"alignment overhead `{rows.get('q4_0', {}).get('page_alignment_overhead_bytes', 0)}` B."
        )
        md.append("")

    md.append("## llama-server inference (real OpenAI-compatible HTTP)")
    md.append("")
    md.append("| Model | Cold load (s) | TTFT mean (ms) | TTFT median (ms) | "
               "Decode tok/s mean | Decode tok/s median | Rounds kept |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for key in ("1.5B_gguf", "4B_gguf"):
        b = summary["llama_server_benchmarks"].get(key, {})
        if not b.get("model"):
            continue
        md.append(
            f"| `{b['model']}` | "
            f"{b['cold_load_seconds']:.2f} | "
            f"{b['ttft_ms_mean']:.0f} | "
            f"{b['ttft_ms_median']:.0f} | "
            f"{b['decode_tok_s_mean']:.1f} | "
            f"{b['decode_tok_s_median']:.1f} | "
            f"{b['n_rounds_kept']} |"
        )
    md.append("")

    out_md = RESULTS / "empirical_summary.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())