#!/usr/bin/env python3
"""
Comprehensive Benchmark Harness for Qwanto Configurable Thinking Engine
Evaluates throughput, latency, and speedup ratios across LOW, MEDIUM, and HIGH modes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
ROOT_DIR = C_DIR.parent

sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(C_DIR / "tools"))

from qwn_thinking import QwnThinkingEngine, ThinkingLevel


def run_benchmark(
    model_path: Path,
    levels: List[str],
    max_tokens: int = 32,
    iterations: int = 5,
    prompt: str = "Explain the fundamental principles of quantum mechanics.",
    output_json: Optional[Path] = None,
) -> Dict[str, Any]:
    print("==========================================================================")
    print("      Qwanto Configurable Thinking Dynamic Reasoning Benchmark            ")
    print("==========================================================================")
    print(f"Model: {model_path}")
    print(f"Tokens per run: {max_tokens} | Iterations: {iterations}")
    print(f"Prompt: {prompt}\n")

    engine = QwnThinkingEngine(model_path)
    results: Dict[str, Any] = {}

    for lvl_str in levels:
        lvl = ThinkingLevel.from_value(lvl_str)
        print(f"--> Benchmarking Thinking Mode: [{lvl.value.upper()}] ({iterations} runs)...")
        bench_data = engine.benchmark(
            prompt=prompt,
            thinking_level=lvl,
            max_tokens=max_tokens,
            iterations=iterations,
        )
        results[lvl.value] = bench_data
        print(f"    Avg Throughput: {bench_data['avg_tok_per_sec']:.2f} tok/s  "
              f"(Min: {bench_data['min_tok_per_sec']:.2f}, Max: {bench_data['max_tok_per_sec']:.2f})")

    # Calculate speedup relative to HIGH mode
    high_tps = results.get("high", {}).get("avg_tok_per_sec", 0.0)
    if high_tps <= 0:
        raise RuntimeError("high thinking mode did not produce a positive measured throughput")

    summary_table = []
    print("\n==========================================================================")
    print("                        THINKING SPEEDUP SUMMARY                          ")
    print("==========================================================================")
    print(f"{'Thinking Mode':<15} | {'Avg Throughput (tok/s)':<24} | {'Speedup Factor':<15}")
    print("-" * 60)

    for lvl_name, data in results.items():
        tps = data["avg_tok_per_sec"]
        speedup = tps / high_tps
        data["speedup_factor"] = round(speedup, 2)
        summary_table.append({
            "mode": lvl_name,
            "avg_tok_per_sec": tps,
            "speedup_factor": round(speedup, 2),
        })
        print(f"{lvl_name.upper():<15} | {tps:<24.2f} | {speedup:<15.2f}x")
    print("==========================================================================\n")

    final_report = {
        "timestamp": time.time(),
        "model": str(model_path),
        "tokens_per_run": max_tokens,
        "iterations": iterations,
        "results": results,
        "summary": summary_table,
    }

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)
        print(f"[OK] Report saved to {output_json}")

    return final_report


def main():
    parser = argparse.ArgumentParser(description="Qwanto Configurable Thinking Benchmark Harness")
    parser.add_argument("--model", type=Path, default=ROOT_DIR / "experiments" / "results" / "4B_hyper_vsq2.qwn", help="Path to .qwn model")
    parser.add_argument("--levels", type=str, default="low,medium,high", help="Comma-separated thinking levels")
    parser.add_argument("--tokens", type=int, default=32, help="Tokens to generate")
    parser.add_argument("--iterations", type=int, default=5, help="Number of benchmark iterations")
    parser.add_argument("--prompt", type=str, default="Explain the theory of general relativity in simple terms.", help="Benchmark prompt")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "benchmark_thinking.json", help="Path to output JSON")
    args = parser.parse_args()

    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    run_benchmark(
        model_path=args.model,
        levels=levels,
        max_tokens=args.tokens,
        iterations=args.iterations,
        prompt=args.prompt,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
