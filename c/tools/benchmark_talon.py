#!/usr/bin/env python3
"""Quality/acceptance benchmark entry point.

No model quality or throughput values are embedded here. Supply a real local
benchmark artifact and evaluate quality separately with a named dataset.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from benchmark_reproducible import main

if __name__ == "__main__":
    main()
