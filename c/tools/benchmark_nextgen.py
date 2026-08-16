#!/usr/bin/env python3
"""Evidence entry point for next-generation experiments.

Experimental kernels may be reported only with their own captured artifact.
This command delegates native inference measurement to the strict local
qwnrun harness and never emits projected performance values as measured data.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from benchmark_reproducible import main

if __name__ == "__main__":
    main()
