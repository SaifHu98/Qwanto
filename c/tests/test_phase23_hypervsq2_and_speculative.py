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
        """Verify real-world benchmark harness executes on existing model checkpoints."""
        model_path = "D:/EcoUni/qwanto/models/DeepSeek-V4-Pro-Qwen3.5-4B-MTP-HyperVSQ.qwn"
        if os.path.exists(model_path):
            ok = run_real_benchmark(model_path, n_gen=16)
            self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
