#!/usr/bin/env python3
"""
Comprehensive Integration Benchmark Harness for Qwanto Performance Autopilot Engine
Evaluates Speedup, Quality Scores, and Memory Efficiency across all modes and task archetypes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
ROOT_DIR = C_DIR.parent

sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(C_DIR / "tools"))

from qwanto_autopilot import QwantoAutoPilot, TaskType


def run_benchmark(
    modes: List[str],
    tasks: List[str],
    iterations: int = 50,
    output_json: Optional[Path] = None,
) -> Dict[str, Any]:
    print("==========================================================================")
    print("      Qwanto Performance Autopilot Full Integration Benchmark            ")
    print("==========================================================================")
    print(f"Modes: {modes} | Tasks: {tasks} | Iterations: {iterations}\n")

    model_path = ROOT_DIR / "experiments" / "results" / "4B_hyper_vsq2.qwn"

    mode_summary = []
    task_results = []

    for mode in modes:
        pilot = QwantoAutoPilot(model_path=model_path, mode=mode)
        
        mode_speedups = []
        mode_qualities = []
        mode_mems = []

        for task in tasks:
            prompt_map = {
                "qa": "What is the boiling point of nitrogen at atmospheric pressure?",
                "code": "def quicksort(arr):\n    if len(arr) <= 1: return arr",
                "reasoning": "Explain step-by-step why the sky is blue using Rayleigh scattering theorem.",
                "agentic": "Search web for latest LLM quantization techniques and format results.",
            }
            prompt = prompt_map.get(task, f"Perform {task} benchmark.")
            tools = [{"tool": "web_search", "args": {"q": "quantization"}}] if task == "agentic" else None

            res = pilot.generate(prompt=prompt, tools=tools, max_tokens=32)

            mode_speedups.append(res.speedup)
            mode_qualities.append(res.quality_score)
            mode_mems.append(res.memory_usage_gb)

            task_results.append({
                "mode": mode,
                "task": task,
                "speedup": f"{res.speedup:.1f}x",
                "quality_score": res.quality_score,
                "memory_gb": f"{res.memory_usage_gb:.1f} GB",
                "active_optimizations": res.active_optimizations,
            })

        avg_speedup = sum(mode_speedups) / len(mode_speedups)
        avg_quality = sum(mode_qualities) / len(mode_qualities)
        avg_mem = sum(mode_mems) / len(mode_mems)

        mode_summary.append({
            "mode": mode,
            "avg_speedup": f"{avg_speedup:.1f}x",
            "quality_score": round(avg_quality, 2),
            "memory_usage": f"{avg_mem:.1f} GB",
        })

    # Print summary table
    print("==========================================================================")
    print("                   PERFORMANCE AUTOPILOT SUMMARY                          ")
    print("==========================================================================")
    print(f"{'Mode':<18} | {'Speedup':<10} | {'Quality Score':<15} | {'Memory Usage':<14}")
    print("-" * 65)
    for m in mode_summary:
        print(f"{m['mode']:<18} | {m['avg_speedup']:<10} | {m['quality_score']:<15} | {m['memory_usage']:<14}")
    print("==========================================================================\n")

    report = {
        "timestamp": time.time(),
        "iterations": iterations,
        "mode_summary": mode_summary,
        "task_details": task_results,
    }

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[OK] Report saved to {output_json}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Qwanto Autopilot Integration Benchmark Harness")
    parser.add_argument("--mode", type=str, default="max-performance,balanced,max-quality")
    parser.add_argument("--tasks", type=str, default="qa,code,reasoning,agentic")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "integration_benchmark.json")
    args = parser.parse_args()

    modes = [x.strip() for x in args.mode.split(",") if x.strip()]
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]

    run_benchmark(
        modes=modes,
        tasks=tasks,
        iterations=args.iterations,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
