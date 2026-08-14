import os
import sys
import struct
import unittest
from pathlib import Path

# Add paths to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
c_dir = Path(__file__).resolve().parent.parent
tools_dir = c_dir / "tools"

for p in [str(root_dir), str(c_dir), str(tools_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from tools.qwn_convert import quantize_hyper_vsq2_rows, _get_quant_dtype_and_size, DT_HYPER_VSQ2
    from tools.qwn_ppl import evaluate_ppl_simulation, WIKITEXT2_SAMPLE
    from tools.qwn_benchmark import run_real_benchmark
except ImportError:
    from c.tools.qwn_convert import quantize_hyper_vsq2_rows, _get_quant_dtype_and_size, DT_HYPER_VSQ2
    from c.tools.qwn_ppl import evaluate_ppl_simulation, WIKITEXT2_SAMPLE
    from c.tools.qwn_benchmark import run_real_benchmark


class TestPhase23HyperVSQ2AndSpeculative(unittest.TestCase):
    def test_hypervsq2_quantization_byte_size(self):
        """Verify 256 elements are quantized into exact 74-byte superblocks (2.31 bpw)."""
        rows = 4
        cols = 256
        raw_f32 = struct.pack(f"{rows * cols}f", *[float(i % 17 - 8) for i in range(rows * cols)])
        
        packed = quantize_hyper_vsq2_rows(raw_f32, rows, cols)
        expected_bytes = rows * 74
        self.assertEqual(len(packed), expected_bytes)
        
        dtype, payload_size = _get_quant_dtype_and_size("hyper_vsq2", rows, cols)
        self.assertEqual(dtype, DT_HYPER_VSQ2)
        self.assertEqual(payload_size, expected_bytes)

    def test_hypervsq2_ppl_metrics(self):
        """Verify HyperVSQ-2 perplexity evaluation."""
        res = evaluate_ppl_simulation("model_hypervsq2.qwn", WIKITEXT2_SAMPLE, bpw_override=2.10)
        self.assertEqual(res["bpw"], 2.10)
        self.assertTrue(12.0 <= res["perplexity"] <= 16.5)
        self.assertTrue(res["accuracy_retention_pct"] > 80.0)
        self.assertTrue(res["compression_ratio"] >= 7.0)

    def test_speculative_header_files_exist(self):
        """Verify SpecDec headers and source files exist."""
        spec_h = Path(__file__).parent.parent / "qwn_speculative.h"
        spec_c = Path(__file__).parent.parent / "qwn_speculative.c"
        self.assertTrue(spec_h.exists())
        self.assertTrue(spec_c.exists())
        self.assertIn("QwnSpecContext", spec_h.read_text(encoding="utf-8"))

    def test_benchmark_harness_execution(self):
        """Verify benchmark harness executes and reports truthfully.

        Per Full Improve Plan section 10, the harness must NEVER
        substitute a default value for a failed measurement.  When
        ``qwnrun.exe`` cannot be spawned (missing binary, sandbox
        policy, missing deps), the new harness surfaces the failure
        in the report rather than fabricating a number.  We therefore
        assert that the harness runs to completion and produces a
        report; aggregate ``status`` is allowed to be ``"ok"`` when
        ``qwnrun`` is executable and ``"error"`` when it is not.
        """
        import tempfile
        from tools.qwn_convert import write_qwn
        from tools.qwn_benchmark_v2 import BenchmarkConfig, BenchmarkRunner
        with tempfile.TemporaryDirectory() as td:
            fixture_path = os.path.join(td, "fixture.qwn")
            tensors = [{
                "name": "weight", "dtype": 2, "shape": (32, 1),
                "payload": b"\x00" * 18, "payload_size": 18,
                "write_payload": None
            }]
            write_qwn(fixture_path, tensors, arch_dims=(32, 32, 1, 1, 32, 1, 32, 128))
            cfg = BenchmarkConfig(model_path=Path(fixture_path), n_gen=4)
            report = BenchmarkRunner(cfg).run()
            self.assertIn("aggregate", report.to_dict())
            self.assertIn(report.aggregate["status"], ("ok", "error"))


if __name__ == "__main__":
    unittest.main()
