import io
import struct
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
        blocks = {12: 144, 13: 176, 14: 210}
        for dtype, size in blocks.items():
            raw = bytearray(size)
            raw[0:2] = struct.pack("<e", 1.0)
            if dtype in (12, 13):
                raw[4:16] = b"\x01" * 12
                raw[16:] = b"\x88" * (size - 16)
            else:
                raw[192:208] = b"\x01" * 16
                raw[208:210] = struct.pack("<e", 1.0)
                raw[:192] = b"\x88" * 192
            if dtype == 12:
                values = qwn_convert._dequantize_q4_k_block(bytes(raw))
            elif dtype == 13:
                values = qwn_convert._dequantize_q5_k_block(bytes(raw))
            else:
                values = qwn_convert._dequantize_q6_k_block(bytes(raw))
            self.assertEqual(values.shape, (256,))
            self.assertTrue(all(value == value for value in values))


if __name__ == "__main__":
    unittest.main()
