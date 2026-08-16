import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

import benchmark_reproducible

HOST_FIXTURE = {"os": "TEST_FIXTURE", "cpu_threads": 1, "gpus_detected": None}


class FakeProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.killed = False

    def communicate(self, timeout=None):
        return self.stdout, self.stderr

    def kill(self):
        self.killed = True


class TestBenchmarkHarness(unittest.TestCase):
    def test_valid_qwnrun_result_is_parsed(self):
        tokens, reason = benchmark_reproducible.parse_runtime_output(
            "Prompt tokens: 3, generating up to 8 tokens...\n",
            "qwnrun result: status=ok tokens=8 wall_seconds=0.25 tok_per_sec=32\n",
        )
        self.assertEqual(tokens, 8)
        self.assertIsNone(reason)
        ttft, reason = benchmark_reproducible.parse_ttft_ms(
            "", "qwnrun result: status=ok tokens=8 ttft_ms=0.0\n"
        )
        self.assertEqual(ttft, 0.0)
        self.assertIsNone(reason)

    def test_malformed_output_is_rejected(self):
        tokens, reason = benchmark_reproducible.parse_runtime_output("hello\n", "diagnostic only\n")
        self.assertIsNone(tokens)
        self.assertIn("status", reason)

    def test_missing_model_returns_unavailable(self):
        with tempfile.NamedTemporaryFile() as executable:
            result = benchmark_reproducible.execute_real_benchmark(
                model_path="nonexistent/model.qwn",
                prompt="Hello",
                custom_executable=executable.name,
            )
        self.assertEqual(result["evidence_classification"], "UNAVAILABLE")
        self.assertIn("model container", result["error_reason"])
        self.assertIsNone(result["measured_evidence"])

    def test_missing_executable_returns_unavailable(self):
        with tempfile.NamedTemporaryFile(suffix=".qwn") as model:
            result = benchmark_reproducible.execute_real_benchmark(
                model_path=model.name,
                prompt="Hello",
                custom_executable="nonexistent/qwnrun",
            )
        self.assertEqual(result["evidence_classification"], "UNAVAILABLE")
        self.assertIn("executable", result["error_reason"])
        self.assertIsNone(result["measured_evidence"])

    def test_nonzero_runtime_exit_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "qwnrun"
            model = Path(directory) / "model.qwn"
            executable.write_bytes(b"runtime fixture")
            model.write_bytes(b"model fixture")
            process = FakeProcess(returncode=7, stderr="qwnrun open error\n")
            with patch.object(benchmark_reproducible, "detect_host_hardware", return_value=HOST_FIXTURE), patch.object(benchmark_reproducible.subprocess, "Popen", return_value=process):
                result = benchmark_reproducible.execute_real_benchmark(str(model), "Hello", custom_executable=str(executable))
        self.assertEqual(result["evidence_classification"], "INVALID")
        self.assertIn("status 7", result["error_reason"])

    def test_zero_token_runtime_result_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "qwnrun"
            model = Path(directory) / "model.qwn"
            executable.write_bytes(b"runtime fixture")
            model.write_bytes(b"model fixture")
            process = FakeProcess(stderr="qwnrun result: status=ok tokens=0 wall_seconds=0.1\n")
            with patch.object(benchmark_reproducible, "detect_host_hardware", return_value=HOST_FIXTURE), patch.object(benchmark_reproducible.subprocess, "Popen", return_value=process):
                result = benchmark_reproducible.execute_real_benchmark(str(model), "Hello", custom_executable=str(executable))
        self.assertEqual(result["evidence_classification"], "INVALID")
        self.assertIn("zero", result["error_reason"])

    def test_valid_runtime_produces_measured_evidence_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "qwnrun"
            model = Path(directory) / "model.qwn"
            executable.write_bytes(b"runtime fixture")
            model.write_bytes(b"model fixture")
            process = FakeProcess(
                stdout="Prompt tokens: 2, generating up to 8 tokens...\nanswer\n",
                stderr="qwnrun result: status=ok tokens=8 wall_seconds=0.1 ttft_ms=2.5 tok_per_sec=80\n",
            )
            with patch.object(benchmark_reproducible, "detect_host_hardware", return_value=HOST_FIXTURE), patch.object(benchmark_reproducible.subprocess, "Popen", return_value=process):
                result = benchmark_reproducible.execute_real_benchmark(str(model), "Hello", custom_executable=str(executable))

        self.assertEqual(result["evidence_classification"], "MEASURED")
        evidence = result["measured_evidence"]
        self.assertEqual(evidence["generated_tokens"], 8)
        self.assertGreater(evidence["wall_seconds"], 0)
        self.assertGreater(evidence["tok_per_sec"], 0)
        self.assertEqual(evidence["ttft_ms"], 2.5)
        for key in ("schema_version", "runtime_metadata", "model_metadata", "benchmark_parameters", "execution_evidence"):
            self.assertIn(key, result)
        json.dumps(result)

    def test_hardware_detection_uses_current_host_values(self):
        hardware = benchmark_reproducible.detect_host_hardware()
        self.assertIn("os", hardware)
        self.assertIn("cpu_threads", hardware)
        self.assertIsInstance(hardware["cpu_threads"], int)
        self.assertGreater(hardware["cpu_threads"], 0)


if __name__ == "__main__":
    unittest.main()
