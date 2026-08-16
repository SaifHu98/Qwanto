#!/usr/bin/env python3
"""Quantization benchmark entry point.

Container size and conversion timing must come from the actual conversion
artifact. Inference throughput must come from a real qwnrun evidence record.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from benchmark_reproducible import main

if __name__ == "__main__":
    main()
