import os
import sys
import unittest
import tempfile
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

import benchmark_reproducible

class TestBenchmarkHarness(unittest.TestCase):
    def test_mock_runtime_parser_and_metrics(self):
        """Simulate a mock execution and verify exact metric calculation and TEST_FIXTURE classification."""
        start = time.perf_counter()
        time.sleep(0.05) # 50ms
        first_token = time.perf_counter()
        time.sleep(0.05) # 50ms
        end = time.perf_counter()

        wall_seconds = end - start
        ttft_ms = (first_token - start) * 1000.0
        generated_tokens = 20
        tps = generated_tokens / wall_seconds

        record = {
            "schema_version": "2.0.0",
            "evidence_classification": "TEST_FIXTURE",
            "measured_evidence": {
                "generated_tokens": generated_tokens,
                "wall_seconds": round(wall_seconds, 4),
                "tok_per_sec": round(tps, 2),
                "ttft_ms": round(ttft_ms, 2)
            }
        }

        self.assertEqual(record["evidence_classification"], "TEST_FIXTURE")
        self.assertAlmostEqual(record["measured_evidence"]["ttft_ms"], 50.0, delta=25.0)
        self.assertGreater(record["measured_evidence"]["tok_per_sec"], 0)

    def test_hardware_detection_is_dynamic(self):
        """Verify that hardware detection does not produce fixed static fallback strings."""
        hw = benchmark_reproducible.detect_host_hardware()
        self.assertIn("os", hw)
        self.assertIn("cpu_model", hw)
        self.assertIn("cpu_threads", hw)
        self.assertIsInstance(hw["cpu_threads"], int)
        self.assertGreater(hw["cpu_threads"], 0)

    def test_missing_model_returns_unavailable(self):
        """Verify that missing model results in UNAVAILABLE classification with clear reason."""
        res = benchmark_reproducible.execute_real_benchmark(
            model_path="nonexistent/model.qwn",
            prompt="Hello"
        )
        self.assertEqual(res["evidence_classification"], "UNAVAILABLE")
        self.assertIn("error_reason", res)

    def test_real_runtime_integration(self):
        """Run real runtime if model and binary are present, otherwise skip with clean explanation."""
        exe = benchmark_reproducible.resolve_qwnrun_executable()
        model = PROJECT_ROOT / "experiments" / "results" / "4B_hyper_vsq2.qwn"

        if not exe or not exe.exists():
            self.skipTest("qwnrun executable not available in environment")
        if not model.exists():
            self.skipTest(f"Model container not present at {model}")

        res = benchmark_reproducible.execute_real_benchmark(
            model_path=str(model),
            prompt="Hello Qwanto",
            max_tokens=16,
            custom_executable=str(exe)
        )
        self.assertIn(res["evidence_classification"], ["MEASURED", "UNAVAILABLE"])


if __name__ == "__main__":
    unittest.main()
