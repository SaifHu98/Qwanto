#!/usr/bin/env python3
"""Run one evidence-producing local benchmark for the selected host.

A multi-scenario matrix requires separate real runs. This compatibility entry
point deliberately does not print a fabricated hardware inventory or matrix.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from benchmark_reproducible import main

if __name__ == "__main__":
    main()
