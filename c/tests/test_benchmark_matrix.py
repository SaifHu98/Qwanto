import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

import generate_benchmark_matrix


class TestBenchmarkMatrix(unittest.TestCase):
    def test_checked_in_matrix_has_required_measured_fields(self):
        matrix = json.loads((PROJECT_ROOT / "benchmarks" / "benchmark_matrix.json").read_text(encoding="utf-8"))
        self.assertEqual(matrix["schema_version"], "1.0.0")
        self.assertTrue(matrix["rows"])
        generate_benchmark_matrix.validate_rows(matrix["rows"])
        row = matrix["rows"][0]
        for key in (
            "executable_sha256", "model_sha256", "model_architecture", "qwn_dtype",
            "backend_actual", "selected_kernel", "prompt", "context_size", "seed",
            "token_count", "decode_throughput_tok_s", "wall_seconds",
        ):
            self.assertNotIn(row[key], ("", "Unavailable", None), key)
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"{row['decode_throughput_tok_s']}", readme)
        self.assertIn("Backend Actually Used", readme)

    def test_cuda_requires_gpu_matmul_and_zero_fallback(self):
        row = {
            "benchmark_id": "cuda-fixture",
            "evidence_classification": "MEASURED",
            "qwn_version": "0.1.0",
            "git_commit": "abc123",
            "executable_sha256": "exe",
            "model_sha256": "model",
            "model_architecture": "qwen35",
            "qwn_dtype": "HyperVSQ-2",
            "backend_requested": "cuda",
            "backend_actual": "cuda",
            "selected_kernel": "hypervsq2-cuda",
            "prompt": "fixed",
            "context_size": 4096,
            "seed": 0,
            "token_count": 4,
            "wall_seconds": 1.0,
            "gpu_matmul_count": 0,
            "cpu_fallback_count": 0,
        }
        with self.assertRaisesRegex(ValueError, "GPU-only"):
            generate_benchmark_matrix.validate_rows([row | {"selected_cpu_isa_kernel": "Unavailable"}])


if __name__ == "__main__":
    unittest.main()
