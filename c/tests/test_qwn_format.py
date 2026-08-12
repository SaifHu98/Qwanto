import os
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from tools.qwn_convert import (
    DESC_SIZE,
    DT_Q4_0,
    HEADER_SIZE,
    INLINE_MAX,
    inspect_qwn,
    convert_safetensors,
    quantize_q4_0_rows,
    synthetic,
    write_qwn,
)


class QwnFormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "model.qwn")

    def tearDown(self):
        self.tmp.cleanup()

    def test_inline_file_footer_and_alignment(self):
        synthetic(self.path, 3)
        info = inspect_qwn(self.path)
        self.assertEqual(info["n_tensors"], 3)
        self.assertEqual(info["inline_count"], 3)
        for tensor in info["tensors"]:
            self.assertEqual(tensor["byte_offset"] % 4096, 0)
            self.assertEqual(tensor["byte_size"] % 64, 0)
        with open(self.path, "rb") as f:
            f.seek(-8, os.SEEK_END)
            tail = struct.unpack("<Q", f.read(8))[0]
        self.assertEqual(tail, info["tail_offset"])
        self.assertEqual(tail % 4096, 0)

    def test_overflow_index_round_trip(self):
        count = INLINE_MAX + 17
        synthetic(self.path, count)
        info = inspect_qwn(self.path)
        self.assertEqual(info["n_tensors"], count)
        self.assertEqual(info["inline_count"], INLINE_MAX)
        self.assertEqual(len(info["tensors"]), count)
        self.assertEqual(info["tensors"][-1]["name"], f"tensor.{count - 1}")

    def test_k_tail_is_padded_per_row(self):
        rows, cols = 3, 37
        values = [float(i % 11 - 5) for i in range(rows * cols)]
        payload = quantize_q4_0_rows(struct.pack(f"<{len(values)}f", *values), rows, cols)
        self.assertEqual(len(payload), rows * 2 * 18)
        write_qwn(self.path, [{"name": "tail", "dtype": DT_Q4_0,
                              "shape": (cols, rows), "payload": payload}])
        tensor = inspect_qwn(self.path)["tensors"][0]
        self.assertEqual(tensor["shape"], (cols, rows))
        self.assertGreaterEqual(tensor["byte_size"], len(payload))

    def test_q4_nibbles_use_bias_eight(self):
        values = [-7.0, 0.0, 7.0] + [0.0] * 29
        payload = quantize_q4_0_rows(struct.pack("<32f", *values), 1, 32)
        first = payload[2]
        second = payload[3]
        self.assertEqual(first & 15, 1)       # -7 -> q=-7 -> stored 1
        self.assertEqual(first >> 4, 8)       #  0 -> q=0  -> stored 8
        self.assertEqual(second & 15, 15)     # +7 -> q=7  -> stored 15

    def test_rejects_bad_q4_layout(self):
        with self.assertRaises(ValueError):
            write_qwn(self.path, [{"name": "bad", "dtype": DT_Q4_0,
                                   "shape": (37, 2), "payload": b"x" * 18}])

    def test_tensor_names_are_bounded(self):
        payload = quantize_q4_0_rows(struct.pack("<32f", *([1.0] * 32)), 1, 32)
        with self.assertRaises(ValueError):
            write_qwn(self.path, [{"name": "x" * 64, "dtype": DT_Q4_0,
                                   "shape": (32, 1), "payload": payload}])

    def test_streaming_safetensors_f16_and_overflow(self):
        source = os.path.join(self.tmp.name, "model.safetensors")
        tensors = {}
        payload = bytearray()
        for i in range(INLINE_MAX + 2):
            raw = struct.pack("<37e", *[float(j % 5 - 2) for j in range(37)])
            start = len(payload); payload += raw
            tensors[f"weight.{i}"] = {"dtype": "F16", "shape": [1, 37],
                                      "data_offsets": [start, len(payload)]}
        header = json.dumps(tensors, separators=(",", ":")).encode()
        with open(source, "wb") as f:
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            f.write(payload)
        convert_safetensors(source, self.path, "q4_0")
        info = inspect_qwn(self.path)
        self.assertEqual(info["n_tensors"], INLINE_MAX + 2)
        self.assertEqual(info["tensors"][0]["shape"], (37, 1))
        self.assertTrue(all(t["byte_offset"] % 4096 == 0 for t in info["tensors"]))


if __name__ == "__main__":
    unittest.main()
