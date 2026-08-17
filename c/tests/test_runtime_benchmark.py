import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import benchmark_runtime_phases


class TestRuntimeBenchmarkEvidence(unittest.TestCase):
    def test_ready_stat_contains_machine_readable_load_fields(self):
        stat = benchmark_runtime_phases.parse_ready_stat(
            b"STAT 0 0.000 0.0 0.0 0 0 model_load_ms=12.5 runtime_ready_ms=14.0 pid=42\n"
        )
        self.assertEqual(stat["model_load_ms"], 12.5)
        self.assertEqual(stat["runtime_ready_ms"], 14.0)
        self.assertEqual(stat["pid"], 42)

    def test_done_keeps_legacy_stat_prefix_and_exposes_phases(self):
        done = benchmark_runtime_phases.parse_done(
            b"DONE warm-2 STAT 8 20.0 0 0 5 0 "
            b"prefill_ms=25.0 prefill_tok_per_sec=200.0 first_token_ms=30.0 "
            b"decode_wall_ms=400.0 decode_tok_per_sec=20.0 pid=99 "
            b"backend_actual=cpu kernel=avx2 gpu_matmul_count=0 "
            b"cpu_fallback_count=0 active_threads=4\n"
        )
        self.assertEqual(done["request_id"], "warm-2")
        self.assertIn("DONE warm-2 STAT", done["runtime_stat_line"])
        self.assertEqual(done["decode_wall_ms"], 400.0)
        self.assertEqual(done["pid"], 99)
        self.assertEqual(done["active_threads"], 4)

    def test_warm_decode_requires_two_requests_under_one_pid(self):
        self.assertTrue(benchmark_runtime_phases.persistent_pid_proven([42, 42], 42, 2))
        self.assertFalse(benchmark_runtime_phases.persistent_pid_proven([42], 42, 2))
        self.assertFalse(benchmark_runtime_phases.persistent_pid_proven([42, 43], 42, 2))

    def test_cuda_zero_matmuls_is_not_proven(self):
        self.assertFalse(benchmark_runtime_phases.cuda_execution_proven({
            "backend_actual": "cuda", "gpu_matmul_count": 0, "cpu_fallback_count": 0,
        }))
        self.assertTrue(benchmark_runtime_phases.cuda_execution_proven({
            "backend_actual": "cuda", "gpu_matmul_count": 1, "cpu_fallback_count": 0,
        }))

    def test_missing_local_model_is_unavailable(self):
        report = benchmark_runtime_phases.run_phase_benchmark(
            str(ROOT / "missing-model.qwn"), "cold-start", "Hello", 4
        )
        self.assertEqual(report["evidence_classification"], "UNAVAILABLE")
        self.assertIn("not found", report["error_reason"])
        json.dumps(report)

    def test_checked_in_manifest_populates_model_identity(self):
        metadata = benchmark_runtime_phases.model_manifest_metadata(
            ROOT / "experiments" / "results" / "4B_hyper_vsq2.qwn"
        )
        self.assertEqual(metadata["architecture"], "qwen35")
        self.assertEqual(metadata["qwn_dtype"], "HyperVSQ-2")


if __name__ == "__main__":
    unittest.main()
