#!/usr/bin/env python3
"""Generate the QWN performance report from recorded local evidence.

This report is deliberately conservative.  Inference numbers come only from
the reproducible ``qwnrun`` evidence schema and model metadata comes from the
checked-in model manifest.  Conversion measurements are kept in a separate
scope, and external GGUF measurements are never merged into native QWN rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
UNAVAILABLE = "Unavailable"

MODE_LABELS = {
    "none": "Unquantized QWN (conversion mode)",
    "q4_0": "Q4_0",
    "vsq": "VSQ",
    "vsq_ultra": "VSQ-Ultra",
    "hyper_vsq": "HyperVSQ",
    "hyper_vsq2": "HyperVSQ-2",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _relative_path(path: Path, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _format_bytes(value: int | None) -> str:
    if value is None or value <= 0:
        return UNAVAILABLE
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    return f"{size:.2f} {unit} ({value:,} bytes)"


def _hardware_label(host: dict[str, Any]) -> str:
    parts = [str(host.get("os") or UNAVAILABLE)]
    if host.get("cpu_model"):
        parts.append(f"CPU: {host['cpu_model']}")
    gpus = host.get("gpus_detected") or []
    names = [str(gpu.get("name")) for gpu in gpus if gpu.get("name")]
    if names:
        parts.append(f"GPU: {', '.join(names)}")
    return "; ".join(parts)


def _matches_manifest(evidence: dict[str, Any], model: dict[str, Any]) -> bool:
    metadata = evidence.get("model_metadata") or {}
    expected_hash = str(model.get("target_sha256") or "").lower()
    actual_hash = str(metadata.get("sha256") or "").lower()
    if expected_hash and actual_hash and expected_hash == actual_hash:
        return True
    evidence_path = str(metadata.get("path") or "").replace("\\", "/")
    target_file = str(model.get("target_file") or "")
    return bool(target_file and evidence_path.endswith(target_file))


def _load_bpw_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {(row.get("label", ""), row.get("mode", "")): row for row in csv.DictReader(stream)}


def _conversion_artifact(label: str, mode: str) -> Path | None:
    names = {
        "none": f"{label}_none.qwn",
        "q4_0": f"{label}_q4_0.qwn",
        "vsq": f"{label}_vsq.qwn",
        "vsq_ultra": f"{label}_vsq_ultra.qwn",
        "hyper_vsq": f"{label}_hyper_vsq.qwn",
        "hyper_vsq2": f"{label}_hyper_vsq2.qwn",
    }
    name = names.get(mode)
    candidate = ROOT / "experiments" / "results" / name if name else None
    return candidate if candidate and candidate.is_file() else None


def _conversion_evidence(
    bpw_rows: dict[tuple[str, str], dict[str, str]],
    label: str = "4B",
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for (row_label, mode), row in sorted(bpw_rows.items()):
        if row_label != label or row.get("ok", "").lower() != "true":
            continue
        artifact = _conversion_artifact(row_label, mode)
        recorded_size = int(row["out_bytes_actual"])
        if artifact is None:
            excluded.append({"mode": mode, "reason": "conversion artifact is absent"})
            continue
        actual_size = artifact.stat().st_size
        if actual_size != recorded_size:
            excluded.append({
                "mode": mode,
                "reason": f"artifact size {actual_size} differs from evidence size {recorded_size}",
            })
            continue
        included.append({
            "model": f"{label} conversion fixture",
            "source_format": "GGUF source",
            "qwn_quantization": MODE_LABELS.get(mode, mode),
            "file_size_bytes": actual_size,
            "bits_per_weight": float(row["effective_bpw"]),
            "ram_vram_measurement": UNAVAILABLE,
            "ttft_ms": None,
            "tokens_per_second": None,
            "hardware": "Windows conversion host; inference hardware not recorded",
            "evidence_class": "MEASURED_CONVERSION",
            "measurement_scope": "QWN conversion only; native qwnrun inference was not measured",
            "evidence": {
                "artifact": _relative_path(artifact),
                "artifact_sha256": _sha256(artifact),
                "source": "experiments/results/bpw_report.csv",
                "conversion_wall_seconds": float(row["wall_seconds"]),
                "conversion_throughput_mb_s": float(row["throughput_mb_s"]),
            },
        })
    return included, excluded


def _native_row(model: dict[str, Any], evidence: dict[str, Any] | None, bpw_rows: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    quantization = str(model.get("quantization") or UNAVAILABLE)
    mode = {
        "hypervsq-2": "hyper_vsq2",
        "hypervsq2": "hyper_vsq2",
        "hypervsq": "hyper_vsq",
        "vsq-ultra": "vsq_ultra",
    }.get(quantization.lower().replace("qwn-", ""), quantization.lower().replace("-", "_"))
    bpw_row = bpw_rows.get(("4B", mode))
    measured = (evidence or {}).get("measured_evidence") or {}
    host = (evidence or {}).get("host_environment") or {}
    classification = str((evidence or {}).get("evidence_classification") or "UNAVAILABLE")
    ttft = measured.get("ttft_ms")
    if not isinstance(ttft, (int, float)) or ttft <= 0:
        ttft = None
    tokens_per_second = measured.get("tok_per_sec") if classification == "MEASURED" else None
    return {
        "model": model.get("model_id", UNAVAILABLE),
        "source_format": "QWN container",
        "qwn_quantization": quantization,
        "file_size_bytes": model.get("target_size_bytes"),
        "bits_per_weight": float(bpw_row["effective_bpw"]) if bpw_row else None,
        "ram_vram_measurement": (
            UNAVAILABLE
            if not measured.get("vram_allocated_gb")
            else str(measured["vram_allocated_gb"])
        ),
        "ttft_ms": ttft,
        "tokens_per_second": tokens_per_second,
        "hardware": _hardware_label(host) if evidence else UNAVAILABLE,
        "evidence_class": classification,
        "measurement_scope": "native qwnrun inference" if classification == "MEASURED" else "no valid native qwnrun evidence",
        "evidence": {
            "manifest": "docs/model-manifest.json",
            "model_sha256": model.get("target_sha256"),
            "artifact": "benchmark_evidence.json" if evidence else None,
            "benchmark_id": (evidence or {}).get("benchmark_id"),
        },
    }


def build_report(
    manifest_path: Path,
    evidence_paths: list[Path],
    bpw_path: Path,
    empirical_summary_path: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    evidence = [_read_json(path) for path in evidence_paths if path.is_file()]
    bpw_rows = _load_bpw_rows(bpw_path)
    native_models = [model for model in manifest.get("models", []) if model.get("format") == "qwn"]
    native_rows: list[dict[str, Any]] = []
    for model in native_models:
        matching = next((item for item in evidence if _matches_manifest(item, model)), None)
        native_rows.append(_native_row(model, matching, bpw_rows))

    conversions, excluded = _conversion_evidence(bpw_rows)
    external: list[dict[str, Any]] = []
    if empirical_summary_path.is_file():
        summary = _read_json(empirical_summary_path)
        for item in (summary.get("llama_server_benchmarks") or {}).values():
            external.append({
                "model": item.get("model", UNAVAILABLE),
                "source_format": "GGUF",
                "evidence_class": "EXPERIMENTAL_EXTERNAL",
                "cold_load_seconds": item.get("cold_load_seconds"),
                "ttft_ms_mean": item.get("ttft_ms_mean"),
                "tokens_per_second_mean": item.get("decode_tok_s_mean"),
                "tokens_per_second_median": item.get("decode_tok_s_median"),
                "evidence": "experiments/results/empirical_summary.json",
            })

    return {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "benchmarks/generate_performance_report.py",
        "policy": {
            "native_inference_numbers_require": "successful qwnrun evidence with matching model hash",
            "unavailable_value": UNAVAILABLE,
            "external_results_are_not_native_claims": True,
            "projections_are_not_measurements": True,
        },
        "source_files": {
            "manifest": _relative_path(manifest_path),
            "benchmark_evidence": [_relative_path(path) for path in evidence_paths if path.is_file()],
            "conversion_evidence": _relative_path(bpw_path),
            "external_evidence": _relative_path(empirical_summary_path) if empirical_summary_path.is_file() else None,
        },
        "native_qwn_rows": native_rows,
        "conversion_rows": conversions,
        "excluded_conversion_records": excluded,
        "external_gguf_rows": external,
        "format_status": [
            {"format": "FP32", "status": "implemented container dtype; no current report evidence", "evidence_class": "UNAVAILABLE"},
            {"format": "FP16", "status": "implemented container dtype; no current report evidence", "evidence_class": "UNAVAILABLE"},
            {"format": "Q4_0", "status": "implemented and container-validated; no matching native inference row", "evidence_class": "UNAVAILABLE"},
            {"format": "HyperVSQ-2", "status": "validated conversion and measured native qwnrun evidence", "evidence_class": "MEASURED"},
            {"format": "TWLA 1.58-bit", "status": "implemented/tested kernel path; no complete model evidence", "evidence_class": "EXPERIMENTAL"},
            {"format": "LittleBit", "status": "implemented/tested library path; not a QWN container dtype", "evidence_class": "EXPERIMENTAL"},
            {"format": "TurboQuant", "status": "implemented/tested KV path; no complete model evidence", "evidence_class": "EXPERIMENTAL"},
        ],
    }


def _display(value: Any, suffix: str = "") -> str:
    if value is None or value == "" or value == UNAVAILABLE:
        return UNAVAILABLE
    return f"{value}{suffix}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Generated QWN performance evidence",
        "",
        "This file is generated from the machine-readable sources listed below. It",
        "contains no fallback throughput or memory values. `Unavailable` means the",
        "runtime did not report a metric or the evidence was not comparable.",
        "",
        "## Native QWN inference",
        "",
        "| Model | Source Format | QWN Quantization | File Size | Bits/Weight if known | RAM / VRAM Measurement | TTFT | Tokens/s | Hardware | Evidence Class |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["native_qwn_rows"]:
        lines.append(
            "| {model} | {source_format} | {qwn_quantization} | {file_size} | {bpw} | {memory} | {ttft} | {tokens} | {hardware} | {evidence} |".format(
                model=row["model"],
                source_format=row["source_format"],
                qwn_quantization=row["qwn_quantization"],
                file_size=_format_bytes(row.get("file_size_bytes")),
                bpw=_display(row.get("bits_per_weight"), " bpw"),
                memory=row.get("ram_vram_measurement") or UNAVAILABLE,
                ttft=_display(row.get("ttft_ms"), " ms"),
                tokens=_display(row.get("tokens_per_second")),
                hardware=row.get("hardware") or UNAVAILABLE,
                evidence=row["evidence_class"],
            )
        )

    lines += [
        "",
        "The native row above is only a qwnrun inference claim when its evidence",
        "class is `MEASURED`. The current artifact records TTFT as unavailable",
        "because the runtime did not expose a positive first-token measurement.",
        "",
        "## Conversion evidence (not inference throughput)",
        "",
        "| Model | QWN mode | File Size | Bits/Weight | Conversion wall time | Conversion throughput | Evidence |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in report["conversion_rows"]:
        evidence = row["evidence"]
        lines.append(
            f"| {row['model']} | {row['qwn_quantization']} | {_format_bytes(row['file_size_bytes'])} | "
            f"{row['bits_per_weight']:.6f} bpw | {evidence['conversion_wall_seconds']:.6f} s | "
            f"{evidence['conversion_throughput_mb_s']:.6f} MB/s | {row['evidence_class']} |"
        )
    if report["excluded_conversion_records"]:
        lines += ["", "Excluded conversion records are retained in the JSON report with their integrity reason."]

    lines += ["", "## Format status", "", "| Format | Status | Evidence Class |", "| --- | --- | --- |"]
    for item in report["format_status"]:
        lines.append(f"| {item['format']} | {item['status']} | {item['evidence_class']} |")

    lines += [
        "",
        "## External GGUF evidence",
        "",
        "These measurements used the external local `llama-server` boundary. They",
        "are shown for provenance only and must not be read as native QWN results.",
        "",
        "| Model | Cold load | TTFT mean | Decode mean | Decode median | Evidence |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["external_gguf_rows"]:
        lines.append(
            f"| {row['model']} | {_display(row.get('cold_load_seconds'), ' s')} | "
            f"{_display(row.get('ttft_ms_mean'), ' ms')} | "
            f"{_display(row.get('tokens_per_second_mean'), ' tok/s')} | "
            f"{_display(row.get('tokens_per_second_median'), ' tok/s')} | {row['evidence_class']} |"
        )

    lines += [
        "",
        "## Sources",
        "",
        "- [QWN container format](qwn-format.md)",
        "- [Benchmark methodology](benchmark-methodology.md)",
        f"- Manifest: `{report['source_files']['manifest']}`",
        f"- Native evidence: `{', '.join(report['source_files']['benchmark_evidence']) or UNAVAILABLE}`",
        f"- Conversion evidence: `{report['source_files']['conversion_evidence']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs" / "model-manifest.json")
    parser.add_argument("--evidence", type=Path, action="append", default=None)
    parser.add_argument("--conversion-report", type=Path, default=ROOT / "experiments" / "results" / "bpw_report.csv")
    parser.add_argument("--empirical-summary", type=Path, default=ROOT / "experiments" / "results" / "empirical_summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "performance-report.json")
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "docs" / "performance-report.md")
    args = parser.parse_args()

    evidence_paths = args.evidence
    if evidence_paths is None:
        default_evidence = ROOT / "benchmark_evidence.json"
        evidence_paths = [default_evidence] if default_evidence.is_file() else []
    report = build_report(args.manifest, evidence_paths, args.conversion_report, args.empirical_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"Generated {args.output} and {args.markdown_output}")


if __name__ == "__main__":
    main()
