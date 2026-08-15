#!/usr/bin/env python3
"""
Comprehensive Validation Suite for All Five Qwanto Optimizations
1. TurboQuant 3.5-bit KV-Cache
2. Configurable Thinking Levels
3. Saguaro SSD Speculative Decoding
4. Agentic Multi-Step Pipeline
5. Unified Performance Autopilot
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
ROOT_DIR = C_DIR.parent


def run_c_test(name: str, exe_path: Path, verbose: bool = False) -> Tuple[bool, str]:
    if not exe_path.exists():
        return False, f"Executable {exe_path.name} not found"
    try:
        proc = subprocess.run(
            [str(exe_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60.0,
        )
        passed = proc.returncode == 0
        output = proc.stdout if passed or not verbose else proc.stdout + "\n" + proc.stderr
        return passed, output
    except Exception as e:
        return False, str(e)


def run_pytest_test(name: str, test_file: Path, verbose: bool = False) -> Tuple[bool, str]:
    try:
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-q"]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60.0,
        )
        passed = proc.returncode == 0
        output = proc.stdout.strip()
        return passed, output
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Qwanto Comprehensive Validation Suite")
    parser.add_argument("--model", type=Path, default=ROOT_DIR / "experiments" / "results" / "4B_hyper_vsq2.qwn")
    parser.add_argument("--tests", type=str, default="all")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("==========================================================================")
    print("           QWANTO UNIFIED INFERENCE ENGINE VALIDATION SUITE               ")
    print("==========================================================================")
    print(f"Target Model: {args.model}")
    print(f"Test Scope:   {args.tests}\n")

    test_matrix = [
        ("HyperVSQ-2 SIMD Vectorized Kernels", C_DIR / "test_hypervsq2_kernels.exe", "c"),
        ("TurboQuant 3.5-Bit KV-Cache", C_DIR / "test_turboquant.exe", "c"),
        ("Configurable Thinking Engine", C_DIR / "test_thinking.exe", "c"),
        ("Saguaro SSD Speculative Decoding", C_DIR / "test_speculative.exe", "c"),
        ("Agentic Multi-Step Pipeline", C_DIR / "test_agentic.exe", "c"),
        ("Performance Autopilot Engine", C_DIR / "test_autopilot.exe", "c"),
        ("Autopilot Quality & Classification", C_DIR / "tests" / "test_autopilot_quality.py", "py"),
        ("Agentic Integration Quality", C_DIR / "tests" / "test_agentic_quality.py", "py"),
        ("Speculative Integration Quality", C_DIR / "tests" / "test_speculative_quality.py", "py"),
        ("Thinking Quality & Latency", C_DIR / "tests" / "test_thinking_quality.py", "py"),
    ]

    results = []
    total_passed = 0
    total_tests = len(test_matrix)

    for name, path, kind in test_matrix:
        t0 = time.perf_counter()
        if kind == "c":
            passed, detail = run_c_test(name, path, args.verbose)
        else:
            passed, detail = run_pytest_test(name, path, args.verbose)
        elapsed = time.perf_counter() - t0

        status_str = "[PASS]" if passed else "[FAIL]"
        if passed:
            total_passed += 1
        print(f"  {status_str:<7} {name:<42} ({elapsed:.2f}s)")
        if args.verbose or not passed:
            print(f"          > {detail[:150].strip()}")
        results.append((name, passed, elapsed))

    print("\n==========================================================================")
    print("                         VALIDATION SUMMARY                               ")
    print("==========================================================================")
    print(f"Total Suites Executed: {total_tests}")
    print(f"Total Suites Passed:   {total_passed} / {total_tests} ({100.0 * total_passed / total_tests:.1f}%)")
    print("==========================================================================")

    if total_passed == total_tests:
        print("[SUCCESS] ALL FIVE OPTIMIZATION MODULES FULLY VALIDATED AND PRODUCTION-READY!")
        sys.exit(0)
    else:
        print("[FAILURE] SOME VALIDATION CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
