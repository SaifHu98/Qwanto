import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from tools.qwn_benchmark import _stats, run_benchmark
from tools.qwn_quality_oracle import _top_k


class QwnQualityToolTests(unittest.TestCase):
    def test_top_k_is_deterministic(self):
        self.assertEqual(_top_k([0.1, 0.9, 0.3, 0.7], 3), [1, 3, 2])

    def test_stats_reads_measured_native_fields(self):
        stderr = ("qwnrun result: status=ok tokens=4 wall_seconds=2.0 "
                   "tok_per_sec=2.0 thinking_level=none\n"
                   "qwnrun result detail: backend=cpu decode_tok_per_sec=2.0 "
                   "cpu_fallback_count=0\n")
        stats = _stats(stderr)
        self.assertEqual(stats["status"], "ok")
        self.assertEqual(stats["tokens"], 4)
        self.assertEqual(stats["backend"], "cpu")
        self.assertEqual(stats["cpu_fallback_count"], 0)

    def test_benchmark_marks_failed_process_without_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake-qwnrun.py"
            fake.write_text(
                "import sys\n"
                "print('qwnrun result: status=ok tokens=1 wall_seconds=0.1 tok_per_sec=10.0', file=sys.stderr)\n"
                "print('qwnrun result detail: backend=cpu cpu_fallback_count=0', file=sys.stderr)\n",
                encoding="utf-8")
            model = root / "model.qwn"
            model.write_bytes(b"fixture")
            result = run_benchmark(Path(sys.executable), model,
                                   [{"name": "smoke", "text": "hello"}],
                                   "cpu", 8, 1, 1)
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertTrue(result["no_projected_performance_claim"])


if __name__ == "__main__":
    unittest.main()
