"""Tests for the Qwanto resource orchestrator (GGUF parsing + split planning)."""

import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import orchestrator
from orchestrator import kv_cache_bytes, parse_gguf_meta, plan

GB = 1 << 30


def write_gguf(path, arch="llama", n_layers=28, n_embd=1536, n_head=12,
               n_head_kv=2, ctx_train=131072, pad_bytes=0):
    """Craft a minimal valid GGUF v3 header with the metadata the parser needs."""
    def kv_str(key, val):
        k = key.encode(); v = val.encode()
        return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v

    def kv_u32(key, val):
        k = key.encode()
        return struct.pack("<Q", len(k)) + k + struct.pack("<I", 4) + struct.pack("<I", val)

    def kv_arr_str(key, items):
        k = key.encode()
        out = struct.pack("<Q", len(k)) + k + struct.pack("<I", 9)
        out += struct.pack("<I", 8) + struct.pack("<Q", len(items))
        for s in items:
            b = s.encode()
            out += struct.pack("<Q", len(b)) + b
        return out

    kvs = [
        kv_str("general.architecture", arch),
        kv_u32(f"{arch}.block_count", n_layers),
        kv_u32(f"{arch}.embedding_length", n_embd),
        kv_u32(f"{arch}.context_length", ctx_train),
        kv_u32(f"{arch}.attention.head_count", n_head),
        kv_u32(f"{arch}.attention.head_count_kv", n_head_kv),
        # a tokenizer-like array AFTER the useful keys: parser must not choke
        kv_arr_str("tokenizer.ggml.tokens", ["<s>", "</s>", "hello"]),
    ]
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    blob += b"".join(kvs) + b"\x00" * pad_bytes
    with open(path, "wb") as f:
        f.write(blob)
    return path


class GGUFParsing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "model.gguf")

    def tearDown(self):
        self.tmp.cleanup()

    def test_parses_architecture_metadata(self):
        write_gguf(self.path)
        meta = parse_gguf_meta(self.path)
        self.assertEqual(meta["arch"], "llama")
        self.assertEqual(meta["n_layers"], 28)
        self.assertEqual(meta["n_embd"], 1536)
        self.assertEqual(meta["n_head"], 12)
        self.assertEqual(meta["n_head_kv"], 2)
        self.assertEqual(meta["file_bytes"], os.path.getsize(self.path))

    def test_rejects_non_gguf(self):
        with open(self.path, "wb") as f:
            f.write(b"NOT A GGUF FILE")
        with self.assertRaises(ValueError):
            parse_gguf_meta(self.path)

    def test_missing_block_count_raises(self):
        # header with only architecture
        blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
        k, v = b"general.architecture", b"llama"
        blob += struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v
        with open(self.path, "wb") as f:
            f.write(blob)
        with self.assertRaises(ValueError):
            parse_gguf_meta(self.path)


class KVCacheMath(unittest.TestCase):
    META = {"n_layers": 28, "n_embd": 1536, "n_head": 12, "n_head_kv": 2}

    def test_f16_size(self):
        # 2 * ctx * n_head_kv * head_dim * layers * 2 bytes
        expect = 2 * 16384 * 2 * (1536 // 12) * 28 * 2
        self.assertEqual(kv_cache_bytes(self.META, 16384, "f16"), expect)

    def test_q4_0_smaller_than_f16(self):
        f16 = kv_cache_bytes(self.META, 16384, "f16")
        q4 = kv_cache_bytes(self.META, 16384, "q4_0")
        self.assertLess(q4, f16 * 0.30)  # 0.5625/2 = 0.28


class Planner(unittest.TestCase):
    META = {"arch": "llama", "n_layers": 28, "n_embd": 1536, "n_head": 12,
            "n_head_kv": 2, "ctx_train": 131072, "file_bytes": int(1.1 * GB)}

    def hw(self, vram_free_gb, n_gpus=1, cores=16, ram_avail_gb=24):
        gpus = [{"name": f"GPU{i}", "total_bytes": int(vram_free_gb * GB) + GB,
                 "free_bytes": int(vram_free_gb * GB)} for i in range(n_gpus)]
        return {"ram_total": 32 * GB, "ram_available": int(ram_avail_gb * GB),
                "gpus": gpus, "physical_cores": cores}

    def test_full_offload_when_model_fits(self):
        p = plan("x.gguf", ctx_size=16384, hw=self.hw(11), meta=self.META)
        self.assertEqual(p["ngl"], 999)
        self.assertTrue(p["full_offload"])
        self.assertEqual(p["batch"], 2048)  # bigger batch on full offload

    def test_partial_offload_on_small_vram(self):
        # 1.1 GB model, but only ~1.2 GB free VRAM -> some layers stay on CPU
        p = plan("x.gguf", ctx_size=16384, hw=self.hw(1.2), meta=self.META)
        self.assertLess(p["ngl"], 28)
        self.assertGreaterEqual(p["ngl"], 0)
        self.assertFalse(p["full_offload"])
        self.assertEqual(p["batch"], 512)

    def test_no_gpu_info_delegates_to_llamacpp(self):
        p = plan("x.gguf", ctx_size=4096, hw=self.hw(0, n_gpus=0), meta=self.META)
        self.assertEqual(p["ngl"], 999)  # unknown GPU: let llama.cpp decide

    def test_multi_gpu_tensor_split(self):
        hw = self.hw(8, n_gpus=2)
        hw["gpus"][1]["free_bytes"] = 4 * GB  # 8 GB + 4 GB -> 0.67/0.33
        p = plan("x.gguf", ctx_size=4096, hw=hw, meta=self.META)
        self.assertIsNotNone(p["tensor_split"])
        parts = [float(x) for x in p["tensor_split"].split(",")]
        self.assertEqual(len(parts), 2)
        self.assertAlmostEqual(sum(parts), 1.0, places=1)
        self.assertGreater(parts[0], parts[1])

    def test_threads_respect_cpu_limit(self):
        p = plan("x.gguf", ctx_size=4096, hw=self.hw(11, cores=16),
                 meta=self.META, cpu_limit=50)
        self.assertEqual(p["threads"], 8)

    def test_kv_quant_extends_gpu_fit(self):
        # with a huge context, quantized KV must offload >= layers than f16
        hw = self.hw(3)
        p_f16 = plan("x.gguf", ctx_size=131072, kv_cache_quant="f16", hw=hw, meta=self.META)
        p_q4 = plan("x.gguf", ctx_size=131072, kv_cache_quant="q4_0", hw=hw, meta=self.META)
        self.assertGreaterEqual(p_q4["ngl"], p_f16["ngl"])

    def test_plan_reads_real_gguf(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            path = write_gguf(os.path.join(tmp.name, "m.gguf"), pad_bytes=1 << 20)
            p = plan(path, ctx_size=4096, hw=self.hw(11))
            self.assertEqual(p["n_layers"], 28)
            self.assertEqual(p["ngl"], 999)
        finally:
            tmp.cleanup()

    def test_describe_is_printable(self):
        p = plan("x.gguf", ctx_size=16384, hw=self.hw(11), meta=self.META)
        text = orchestrator.describe(p)
        self.assertIn("ngl=999", text)
        self.assertIn("layers", text)


if __name__ == "__main__":
    unittest.main()
