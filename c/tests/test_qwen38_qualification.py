"""Fail-closed tests for Qwen3.8 GGUF qualification."""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))

from tools import qwn_convert
from tools import qwen38_qualification as qualification


MODEL = HERE.parent / "models" / "Qwen3.8-27B-UD-IQ2_M.gguf"


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _write_minimal_gguf(path: Path, *, architecture: str = "", dtype: int = 18) -> None:
    metadata = [("tokenizer.ggml.tokens", 8, ["a", "b"]),
                ("tokenizer.ggml.merges", 8, [])]
    if architecture:
        metadata.append(("general.architecture", 8, architecture))
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata)))
    for key, value_type, value in metadata:
        header += _gguf_string(key)
        if isinstance(value, list):
            header += struct.pack("<IIQ", 9, value_type, len(value))
            for item in value:
                header += _gguf_string(item)
        else:
            header += struct.pack("<I", value_type) + _gguf_string(value)
    header += _gguf_string("blk.0.ssm_a")
    header += struct.pack("<I", 1) + struct.pack("<Q", 48)
    header += struct.pack("<IQ", dtype, 0)
    data_base = (len(header) + 31) & ~31
    path.write_bytes(bytes(header) + b"\0" * (data_base - len(header)) + b"\0" * 192)


class Qwen38QualificationTests(unittest.TestCase):
    def test_supported_iq_dtype_is_not_reinterpreted_as_another_source_block(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "iq.gguf"
            _write_minimal_gguf(source, dtype=18)
            report = qualification.inspect_gguf(source)
            self.assertEqual(report["dtype_summary"]["unsupported_by_current_converter"], [])
            tensor = report["tensors"][0]
            self.assertEqual(tensor["source_dtype"], "IQ3_XXS")
            self.assertNotIn("not implemented", tensor["qualification_reason"])

    def test_qwen35_architecture_is_parsed_for_native_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "hybrid.gguf"
            output = Path(directory) / "model.qwn"
            _write_minimal_gguf(source, architecture="qwen35", dtype=0)
            tensors, dims = qwn_convert._read_gguf_tensors(str(source), "q4_0")
            self.assertEqual(dims[5], 0)
            self.assertTrue(any(t["name"] == "__qwn.config" for t in tensors))

    @unittest.skipUnless(MODEL.exists(), "Qwen3.8 GGUF source is not present")
    def test_real_source_has_complete_coverage_and_hybrid_components(self):
        report = qualification.inspect_gguf(MODEL)
        self.assertEqual(report["gguf"]["tensor_count"], 866)
        self.assertEqual(len(report["tensors"]), report["gguf"]["tensor_count"])
        self.assertEqual(report["architecture"]["gated_deltanet_layer_count"], 48)
        self.assertEqual(report["architecture"]["full_attention_layer_count"], 17)
        self.assertEqual(len(report["architecture"]["mtp_tensor_names"]), 4)
        self.assertEqual(report["dtype_summary"]["general_file_type_label"], "IQ2_M mixed quantization")
        self.assertNotIn(18, report["dtype_summary"]["unsupported_by_current_converter"])
        self.assertNotIn(22, report["dtype_summary"]["unsupported_by_current_converter"])
        self.assertEqual(report["architecture"]["lm_head"],
                         "separate output.weight and token_embd.weight tensors")

    @unittest.skipUnless(MODEL.exists(), "Qwen3.8 GGUF source is not present")
    def test_qualification_reports_conversion_and_runtime_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = qualification.qualify(
                MODEL, Path(directory) / "reports",
                ram_bytes=32 * 1024 ** 3,
                vram_bytes=12 * 1024 ** 3,
                gpu_name="test GPU",
            )
            self.assertEqual(
                reports["qualification-summary.json"].read_text(encoding="utf-8")
                .split('"decision": "', 1)[1].split('"', 1)[0],
                "READY_FOR_CONVERSION_RUNTIME_GATE",
            )
            self.assertEqual(list(Path(directory).rglob("*.qwn")), [])


if __name__ == "__main__":
    unittest.main()
