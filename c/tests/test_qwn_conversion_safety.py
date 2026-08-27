import io
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from tools import qwn_convert


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _write_minimal_gguf(path: Path, name: str, dims, dtype: int, payload=b""):
    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<IQQ", 3, 1, 2)
    for key, values in (("tokenizer.ggml.tokens", ["a", "b"]),
                        ("tokenizer.ggml.merges", [])):
        header += _gguf_string(key)
        header += struct.pack("<IIQ", 9, 8, len(values))
        for value in values:
            header += _gguf_string(value)
    header += _gguf_string(name)
    header += struct.pack("<I", len(dims))
    header += struct.pack(f"<{len(dims)}Q", *dims)
    header += struct.pack("<IQ", dtype, 0)
    data_base = (len(header) + 31) & ~31
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(b"\0" * (data_base - len(header)))
        stream.write(payload)


class QwnConversionSafetyTests(unittest.TestCase):
    def test_native_k_rows_match_reference_dequantization(self):
        clang = shutil.which("clang")
        if not clang:
            self.skipTest("clang not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exe = root / ("k-row.exe" if os.name == "nt" else "k-row")
            result = subprocess.run(
                [clang, "-std=c11", "-O2", str(HERE / "tests" / "test_qwn_kernels.c"),
                 str(HERE / "qwanto_native.c"), str(HERE / "qwanto_kernels.c"), "-o", str(exe)],
                capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            for dtype, block_bytes in ((qwn_convert.DT_Q2_K, 84),
                                       (qwn_convert.DT_Q3_K, 110),
                                       (qwn_convert.DT_Q8_K, 292)):
                source_dtype = {qwn_convert.DT_Q2_K: 10,
                                qwn_convert.DT_Q3_K: 11,
                                qwn_convert.DT_Q8_K: 15}[dtype]
                payload = bytes(((i * 37 + source_dtype * 11) & 0xff)
                                for i in range(block_bytes))
                if source_dtype == 15:
                    payload = struct.pack("<f", 0.125) + payload[4:]
                model = root / f"k-{dtype}.qwn"
                qwn_convert.write_qwn(
                    str(model), [{"name": "weight", "dtype": dtype,
                                  "shape": (256, 1), "payload": payload}])
                run = subprocess.run([str(exe), str(model)], capture_output=True,
                                     text=True, check=False)
                self.assertEqual(run.returncode, 0, run.stderr)
                actual = [float(value) for value in run.stdout.split()]
                expected = qwn_convert._dequantize_k_payload(payload, source_dtype)
                self.assertEqual(len(actual), 256)
                for index, (got, want) in enumerate(zip(actual, expected)):
                    self.assertTrue(math.isfinite(got))
                    self.assertAlmostEqual(got, float(want), delta=1e-3,
                                           msg=f"dtype={dtype} index={index}")

    def test_native_iq_rows_match_reference_dequantization(self):
        clang = shutil.which("clang")
        if not clang:
            self.skipTest("clang not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exe = root / ("iq-row.exe" if os.name == "nt" else "iq-row")
            result = subprocess.run(
                [clang, "-std=c11", "-O2", str(HERE / "tests" / "test_qwn_kernels.c"),
                 str(HERE / "qwanto_native.c"), str(HERE / "qwanto_kernels.c"),
                 "-o", str(exe)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            cases = ((16, qwn_convert.DT_IQ2_XXS, 66),
                     (17, qwn_convert.DT_IQ2_XS, 74),
                     (18, qwn_convert.DT_IQ3_XXS, 98),
                     (21, qwn_convert.DT_IQ3_S, 110),
                     (22, qwn_convert.DT_IQ2_S, 82),
                     (20, qwn_convert.DT_IQ4_NL, 18),
                     (23, qwn_convert.DT_IQ4_XS, 136))
            for source_dtype, native_dtype, block_bytes in cases:
                with self.subTest(source_dtype=source_dtype):
                    payload = bytes(((i * 73 + source_dtype * 19) & 0xff)
                                    for i in range(block_bytes * (8 if source_dtype == 20 else 1)))
                    payload = bytearray(payload)
                    payload[:2] = struct.pack("<e", 0.5)
                    if source_dtype == 23:
                        payload[2:4] = struct.pack("<H", 0xAAAA)
                        payload[4:8] = b"\x11\x11\x11\x11"
                    model = root / f"iq-{source_dtype}.qwn"
                    qwn_convert.write_qwn(
                        str(model), [{"name": "weight", "dtype": native_dtype,
                                      "shape": (256, 1), "payload": bytes(payload)}])
                    run = subprocess.run([str(exe), str(model)], capture_output=True,
                                         text=True, check=False)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    actual = [float(value) for value in run.stdout.split()]
                    if source_dtype in (20, 23):
                        expected = (qwn_convert._dequantize_iq4_nl_payload(bytes(payload))
                                    if source_dtype == 20 else
                                    qwn_convert._dequantize_iq4_xs_payload(bytes(payload)))
                    else:
                        expected = qwn_convert._dequantize_iq2_iq3_payload(bytes(payload), source_dtype)
                    self.assertEqual(len(actual), 256)
                    for index, (got, want) in enumerate(zip(actual, expected)):
                        self.assertTrue(math.isfinite(got))
                        self.assertAlmostEqual(got, float(want), delta=2e-3,
                                               msg=f"dtype={source_dtype} index={index}")

    def test_gguf_shape_stays_fastest_dimension_first(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.gguf"
            _write_minimal_gguf(path, "blk.0.attn_q.weight", (2, 3), 0,
                                 struct.pack("<6f", *range(6)))
            tensors, _ = qwn_convert._read_gguf_tensors(str(path), "none")
            self.assertEqual(tensors[0]["shape"], (2, 3))

    def test_k_quant_is_dequantized_before_writing_qwn(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q4_k.gguf"
            _write_minimal_gguf(path, "blk.0.attn_q.weight", (256, 1), 12,
                                 b"\0\0\0\0" + b"\x01" * 12 + b"\x88" * 128)
            tensors, _ = qwn_convert._read_gguf_tensors(str(path), "none")
            tensor = tensors[0]
            self.assertEqual(tensor["dtype"], qwn_convert.DT_F32)
            self.assertEqual(tensor["payload_size"], 256 * 4)
            output = io.BytesIO()
            tensor["write_payload"](output)
            values = struct.unpack("<256f", output.getvalue())
            self.assertTrue(all(value == value for value in values))

    def test_k_quant_block_decoders_return_finite_full_blocks(self):
        blocks = {10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 15: 292}
        for dtype, size in blocks.items():
            raw = bytearray(size)
            if dtype == 10:
                raw[0:16] = b"\x11" * 16
                raw[16:80] = b"\x00" * 64
                raw[80:84] = struct.pack("<ee", 1.0, 0.0)
            elif dtype == 11:
                raw[32:96] = b"\x00" * 64
                raw[96:108] = b"\x00" * 12
                raw[108:110] = struct.pack("<e", 1.0)
            elif dtype in (12, 13):
                raw[0:2] = struct.pack("<e", 1.0)
                raw[4:16] = b"\x01" * 12
                raw[16:] = b"\x88" * (size - 16)
            elif dtype == 14:
                raw[192:208] = b"\x01" * 16
                raw[208:210] = struct.pack("<e", 1.0)
                raw[:192] = b"\x88" * 192
            else:
                raw[0:4] = struct.pack("<f", 1.0)
                raw[4:260] = bytes(range(256))
            values = qwn_convert._dequantize_k_payload(bytes(raw), dtype)
            self.assertEqual(values.shape, (256,))
            self.assertTrue(all(value == value for value in values))

    def test_iq4_nl_block_decoder_matches_reference_codebook(self):
        raw = struct.pack("<e", 1.0) + bytes([0x88]) * 16
        values = qwn_convert._dequantize_iq4_nl_payload(raw)
        self.assertEqual(values.shape, (32,))
        self.assertTrue(all(value == 1.0 for value in values))

    def test_iq4_xs_block_decoder_matches_reference_codebook(self):
        raw = struct.pack("<eH", 1.0, 0xAAAA) + b"\x11" * 4 + b"\x88" * 128
        values = qwn_convert._dequantize_iq4_xs_payload(raw)
        self.assertEqual(values.shape, (256,))
        self.assertTrue(all(value == 1.0 for value in values))

    def test_gguf_k_and_iq4_nl_sources_route_to_f32_writer(self):
        cases = ((10, 84), (11, 110), (15, 292), (16, 66), (17, 74),
                 (18, 98), (20, 18), (21, 110), (22, 82), (23, 136))
        for dtype, _block_size in cases:
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"source_{dtype}.gguf"
                if dtype == 10:
                    payload = b"\x11" * 16 + b"\x00" * 64 + struct.pack("<ee", 1.0, 0.0)
                elif dtype == 11:
                    payload = b"\x00" * 32 + b"\x00" * 64 + b"\x00" * 12 + struct.pack("<e", 1.0)
                elif dtype == 15:
                    payload = struct.pack("<f", 1.0) + bytes(range(256)) + b"\x00" * 32
                elif dtype == 16:
                    payload = struct.pack("<e", 1.0) + b"\x00" * 64
                elif dtype == 17:
                    payload = struct.pack("<e", 1.0) + b"\x00" * 64 + b"\x00" * 8
                elif dtype == 18:
                    payload = struct.pack("<e", 1.0) + b"\x00" * 96
                elif dtype == 23:
                    payload = struct.pack("<eH", 1.0, 0xAAAA) + b"\x11" * 4 + b"\x88" * 128
                elif dtype == 21:
                    payload = struct.pack("<e", 1.0) + b"\x00" * 64 + b"\x00" * 32 + b"\x00" * 8 + b"\x00" * 4
                elif dtype == 22:
                    payload = struct.pack("<e", 1.0) + b"\x00" * 32 + b"\x00" * 32 + b"\x00" * 8 + b"\x00" * 8
                else:
                    payload = struct.pack("<e", 1.0) + b"\x88" * 16
                _write_minimal_gguf(path, "blk.0.attn_q.weight", (32 if dtype == 20 else 256, 1), dtype, payload)
                tensors, _ = qwn_convert._read_gguf_tensors(str(path), "none")
                output = io.BytesIO()
                tensors[0]["write_payload"](output)
                expected_values = 32 if dtype == 20 else 256
                if dtype in (10, 11, 15):
                    self.assertEqual(tensors[0]["dtype"],
                                     {10: qwn_convert.DT_Q2_K,
                                      11: qwn_convert.DT_Q3_K,
                                      15: qwn_convert.DT_Q8_K}[dtype])
                    self.assertEqual(len(output.getvalue()), {10: 84, 11: 110, 15: 292}[dtype])
                elif dtype in (16, 17, 18, 21, 22, 20, 23):
                    self.assertEqual(tensors[0]["dtype"], {
                        16: qwn_convert.DT_IQ2_XXS, 17: qwn_convert.DT_IQ2_XS,
                        18: qwn_convert.DT_IQ3_XXS, 21: qwn_convert.DT_IQ3_S,
                        22: qwn_convert.DT_IQ2_S, 20: qwn_convert.DT_IQ4_NL,
                        23: qwn_convert.DT_IQ4_XS}[dtype])
                    self.assertEqual(len(output.getvalue()), _block_size)
                else:
                    self.assertEqual(len(output.getvalue()), expected_values * 4)
                if dtype not in (10, 11, 15, 16, 17, 18, 20, 21, 22, 23):
                    self.assertTrue(all(value == value for value in struct.unpack(f"<{expected_values}f", output.getvalue())))

    def test_q8_target_writer_emits_native_row_layout(self):
        raw = struct.pack("<64f", *[float(index - 32) for index in range(64)])
        payload = qwn_convert.quantize_matrix_rows(raw, 2, 32, "q8_0")
        self.assertEqual(len(payload), 2 * 34)
        for offset in (0, 34):
            scale = struct.unpack_from("<e", payload, offset)[0]
            values = struct.unpack_from("<32b", payload, offset + 2)
            self.assertGreater(scale, 0.0)
            self.assertLessEqual(max(abs(value) for value in values), 127)


if __name__ == "__main__":
    unittest.main()
