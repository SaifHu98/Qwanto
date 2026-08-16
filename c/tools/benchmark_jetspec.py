#!/usr/bin/env python3
"""Speculative-decoding benchmark entry point.

The old script printed fixed throughput and acceptance values. Use the real
harness for native throughput and add a separately captured acceptance artifact
before publishing a speculative-decoding comparison.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from benchmark_reproducible import main

if __name__ == "__main__":
    main()
