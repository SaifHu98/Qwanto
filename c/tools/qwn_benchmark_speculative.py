#!/usr/bin/env python3
"""Report the evidence status of speculative decoding experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def run_benchmark(
    target_model: Path,
    draft_model: Path,
    draft_lengths: list[int],
    cache_sizes: list[int],
    bidirectional_options: list[bool],
    tokens: int = 32,
    prompt: str = "",
    output_json: Path | None = None,
) -> dict:
    report = {
        "schema_version": "3.0.0",
        "benchmark_id": f"speculative-{time.time_ns()}",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_classification": "EXPERIMENTAL",
        "error_reason": "Speculative acceptance and cache behavior are not measured by the native qwnrun release harness.",
        "benchmark_parameters": {
            "target_model": str(target_model),
            "draft_model": str(draft_model),
            "draft_lengths": draft_lengths,
            "cache_sizes": cache_sizes,
            "bidirectional": bidirectional_options,
            "tokens": tokens,
            "prompt_length_chars": len(prompt),
        },
        "measured_evidence": None,
    }
    if output_json:
        output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--draft-lengths", default="3,5,8,10,15")
    parser.add_argument("--cache-size", default="64,128,256")
    parser.add_argument("--bidirectional", default="true,false")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_benchmark(
        args.target_model,
        args.draft_model,
        [int(value) for value in args.draft_lengths.split(",") if value],
        [int(value) for value in args.cache_size.split(",") if value],
        [value.lower() == "true" for value in args.bidirectional.split(",") if value],
        args.tokens,
        args.prompt,
        args.output,
    )


if __name__ == "__main__":
    main()
