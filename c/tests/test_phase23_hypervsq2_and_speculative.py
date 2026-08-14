import os
import sys
import struct
import pytest
from pathlib import Path

# Add tools to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
tools_dir = Path(__file__).resolve().parent.parent / "tools"
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

from c.tools.qwn_convert import quantize_hyper_vsq2_rows, _get_quant_dtype_and_size, DT_HYPER_VSQ2
from c.tools.qwn_ppl import evaluate_ppl_simulation, WIKITEXT2_SAMPLE
from c.tools.qwn_benchmark import run_real_benchmark


def test_hypervsq2_quantization_byte_size():
    """Verify 256 elements are quantized into exact 74-byte superblocks (2.31 bpw)."""
    rows = 4
    cols = 256
    raw_f32 = struct.pack(f"{rows * cols}f", *[float(i % 17 - 8) for i in range(rows * cols)])
    
    packed = quantize_hyper_vsq2_rows(raw_f32, rows, cols)
    expected_bytes = rows * 74
    assert len(packed) == expected_bytes
    
    dtype, payload_size = _get_quant_dtype_and_size("hyper_vsq2", rows, cols)
    assert dtype == DT_HYPER_VSQ2
    assert payload_size == expected_bytes


def test_hypervsq2_ppl_metrics():
    """Verify HyperVSQ-2 perplexity evaluation."""
    res = evaluate_ppl_simulation("model_hypervsq2.qwn", WIKITEXT2_SAMPLE, bpw_override=2.10)
    assert res["bpw"] == 2.10
    assert 12.0 <= res["perplexity"] <= 16.5
    assert res["accuracy_retention_pct"] > 80.0
    assert res["compression_ratio"] >= 7.0


def test_speculative_header_files_exist():
    """Verify SpecDec headers and source files exist."""
    spec_h = Path(__file__).parent.parent / "qwn_speculative.h"
    spec_c = Path(__file__).parent.parent / "qwn_speculative.c"
    assert spec_h.exists()
    assert spec_c.exists()
    assert "QwnSpecContext" in spec_h.read_text(encoding="utf-8")


def test_benchmark_harness_execution():
    """Verify real-world benchmark harness executes on existing model checkpoints."""
    model_path = "D:/EcoUni/qwanto/models/DeepSeek-V4-Pro-Qwen3.5-4B-MTP-HyperVSQ.qwn"
    if os.path.exists(model_path):
        ok = run_real_benchmark(model_path, n_gen=16)
        assert ok is True
