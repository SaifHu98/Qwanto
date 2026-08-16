#!/usr/bin/env python3
"""Compatibility entry point for the evidence-producing benchmark harness."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from benchmark_reproducible import main as run_benchmark

if __name__ == "__main__":
    run_benchmark()
