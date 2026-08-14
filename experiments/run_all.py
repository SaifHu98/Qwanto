"""
experiments/run_all.py — Full empirical study on the two attached models.

Runs every meaningful conversion (raw passthrough + each quant format),
records real wall-clock conversion time, on-disk .qwn size, and the
true effective bpw computed by ``qwn_bpw_truth``.

Outputs:
  experiments/results/conversions.csv
  experiments/results/bpw_report.csv
  experiments/results/summary.json

The driver is read-only with respect to source models; nothing is
committed to the source .gguf files.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "c" / "tools"))

import qwn_bpw_truth as bpw
import qwn_convert as qcnv
import qwn_plan_cli as qpc


QUANT_MODES = ("none", "q4_0", "vsq", "vsq_ultra", "hyper_vsq", "hyper_vsq2")
SRC_MODELS = {
    "1.5B": ROOT / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
    "4B":   ROOT / "models" / "DeepSeek-V4-Pro-Qwen3.5-4B-MTP-BF16.gguf",
}
OUT_DIR = ROOT / "experiments" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _conv_one(src: Path, mode: str, dst: Path) -> Dict[str, object]:
    t0 = time.perf_counter()
    try:
        # convert_model returns the file size written, not a tensor count.
        n_bytes = qcnv.convert_model(str(src), str(dst), quant=mode)
    except Exception as exc:
        return {
            "src": src.name, "mode": mode, "ok": False,
            "error": f"{exc!r}", "traceback": traceback.format_exc(),
        }
    wall = time.perf_counter() - t0
    on_disk = dst.stat().st_size
    return {
        "src": src.name, "mode": mode, "ok": True,
        "wall_seconds": wall, "out_bytes": n_bytes,
        "out_bytes_actual": on_disk,
        "out_mb_actual": on_disk / 1024 ** 2,
        "src_mb": src.stat().st_size / 1024 ** 2,
        "throughput_mb_s": (on_disk / 1024 ** 2) / wall if wall > 0 else 0.0,
    }


def _inspect_and_bpw(qwn_path: Path) -> Dict[str, object]:
    info = qcnv.inspect_qwn(str(qwn_path))
    tensors: List[bpw.TensorByteBreakdown] = []
    per_format_bytes: Dict[str, int] = {}
    for t in info.get("tensors", []):
        dt = int(t.get("dtype", 0))
        bk = bpw.spec_for(dt)
        per_format_bytes[bk.name] = per_format_bytes.get(bk.name, 0) + int(t.get("payload_size", 0))
        tensors.append(bpw.TensorByteBreakdown(
            name=str(t.get("name", "")),
            numel=int(t.get("numel", 0)),
            dt_id=dt,
            payload_bytes=int(t.get("payload_size", t.get("byte_size", 0))),
            page_aligned_bytes=int(t.get("byte_size", t.get("payload_size", 0))),
            descriptor_bytes=bpw.DESC_SIZE,
        ))
    rep = bpw.report(tensors)
    return {
        "n_tensors": info.get("n_tensors", 0),
        "n_params": info.get("n_params", 0),
        "arch_dims": info.get("arch_dims", []),
        "payload_bpw": rep.format_payload_bpw,
        "effective_bpw": rep.format_effective_bpw,
        "total_weights": rep.total_weights,
        "size_on_disk_bytes": rep.size_on_disk_bytes,
        "size_on_disk_mb": rep.size_on_disk_bytes / 1024 ** 2,
        "per_format_bytes": per_format_bytes,
        "payload_bytes_total": rep.payload_bytes_total,
        "header_bytes": rep.header_bytes,
        "tail_block_bytes": rep.tail_block_bytes,
        "descriptor_bytes_total": rep.descriptor_bytes_total,
        "page_alignment_overhead_bytes": rep.page_alignment_overhead_bytes,
    }


def main() -> int:
    rows: List[Dict[str, object]] = []
    bpw_rows: List[Dict[str, object]] = []
    summary: Dict[str, object] = {"models": {}}

    for label, src in SRC_MODELS.items():
        if not src.exists():
            print(f"skip {label}: {src} not found")
            continue
        summary["models"][label] = {
            "src": src.name,
            "src_size_bytes": src.stat().st_size,
            "src_size_mb": src.stat().st_size / 1024 ** 2,
        }
        for mode in QUANT_MODES:
            dst = OUT_DIR / f"{label}_{mode}.qwn"
            print(f"==> convert {label} mode={mode}")
            row = _conv_one(src, mode, dst)
            row["label"] = label
            rows.append(row)
            if not row.get("ok"):
                print(f"    FAILED: {row.get('error')}")
                continue
            try:
                bp = _inspect_and_bpw(dst)
            except Exception as exc:
                bp = {"error": f"{exc!r}"}
            bpw_rows.append({"label": label, "mode": mode, **row, **bp})
            print(f"    size={row['out_mb_actual']:.1f}MB  "
                  f"time={row['wall_seconds']:.2f}s  "
                  f"speed={row['throughput_mb_s']:.0f} MB/s  "
                  f"payload_bpw={bp.get('payload_bpw', 'NA')}  "
                  f"effective_bpw={bp.get('effective_bpw', 'NA')}")

    # Write CSVs
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(OUT_DIR / "conversions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    if bpw_rows:
        keys = sorted({k for r in bpw_rows for k in r.keys()
                       if k != "per_format_bytes"})
        flat_rows = [{k: v for k, v in r.items() if k != "per_format_bytes"}
                     for r in bpw_rows]
        with open(OUT_DIR / "bpw_report.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(flat_rows)
        # per-format breakdown (nested) — JSON only, kept separate from CSV
        with open(OUT_DIR / "per_format.json", "w", encoding="utf-8") as f:
            json.dump([{"label": r["label"], "mode": r["mode"],
                        "per_format_bytes": r["per_format_bytes"]}
                       for r in bpw_rows], f, indent=2)

    summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary["n_conversions"] = len(rows)
    summary["n_ok"] = sum(1 for r in rows if r.get("ok"))
    summary["n_failed"] = sum(1 for r in rows if not r.get("ok"))
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nSummary:")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())