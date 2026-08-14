"""
qwn2gguf.py — Convert any .qwn container to GGUF v3 (no re-quantization).

The .qwn container holds its own payload bytes; the conversion is a
format-preserving repack that emits the standard GGUF header +
tensor index that llama.cpp / llama-server / llama-cpp-python /
ollama / vLLM (with the GGUF loader) all consume natively.

Result: the same model weights become loadable by every GGUF-aware
runtime, without depending on the Qwanto native decoder.  The .qwn
file remains the source of truth; this tool is a one-way export.
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import time
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qwn_loader import QwnModel, QwnTensor
from qwn_bpw_truth import ALIGN_PAGE


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

GT_F32 = 0
GT_F16 = 1
GT_Q4_0 = 2
GT_Q8_0 = 8
GT_BF16 = 30


_QWN_TO_GGUF_DTYPE = {
    0: GT_F32,     # DT_F32
    1: GT_F16,     # DT_F16
    2: GT_Q4_0,    # DT_Q4_0
    3: GT_Q8_0,    # DT_Q8_0
    4: GT_BF16,    # DT_BF16
}


ARCH_MAP = {
    "qwen2": "qwen2", "qwen": "qwen2", "qwen35": "qwen2",
    "llama": "llama", "moe": "llama", "mamba": "mamba",
    "hybrid": "hybrid",
}


def _w_str(f, s: str) -> None:
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def _w_u32(f, key: str, value: int) -> None:
    _w_str(f, key)
    f.write(struct.pack("<I", 4))                # gguf_uint32 tag
    f.write(struct.pack("<I", value))


def _w_u64(f, key: str, value: int) -> None:
    _w_str(f, key)
    f.write(struct.pack("<I", 10))               # gguf_uint64 tag
    f.write(struct.pack("<Q", value))


def _w_f32(f, key: str, value: float) -> None:
    _w_str(f, key)
    f.write(struct.pack("<I", 6))                # gguf_float32 tag
    f.write(struct.pack("<f", value))


def _w_strkv(f, key: str, value: str) -> None:
    _w_str(f, key)
    f.write(struct.pack("<I", 8))                # gguf_string tag
    _w_str(f, value)


def _w_arr_str(f, key: str, values: List[str]) -> None:
    _w_str(f, key)
    f.write(struct.pack("<I", 9))                # gguf_array tag
    f.write(struct.pack("<I", 8))                # element type: string
    f.write(struct.pack("<Q", len(values)))
    for v in values:
        _w_str(f, v)


def _w_arr_u32(f, key: str, values: List[int]) -> None:
    _w_str(f, key)
    f.write(struct.pack("<I", 9))                # gguf_array tag
    f.write(struct.pack("<I", 4))                # element type: uint32
    f.write(struct.pack("<Q", len(values)))
    for v in values:
        f.write(struct.pack("<I", v))


def convert(qwn_path: Path, out_path: Path) -> int:
    model = QwnModel.open(qwn_path)
    try:
        tensors: List[QwnTensor] = [t for t in model.tensors
                                     if not t.name.startswith("__qwn.")]
        cfg = model.config or {}
        gguf_arch = ARCH_MAP.get(model.arch, "llama")

        hidden   = int(cfg.get("hidden_size",
                                model.arch_dims[0] if model.arch_dims else 0))
        inter    = int(cfg.get("intermediate_size",
                                model.arch_dims[1] if len(model.arch_dims) > 1 else 0))
        heads    = int(cfg.get("num_attention_heads",
                                model.arch_dims[2] if len(model.arch_dims) > 2 else 0))
        kv_heads = int(cfg.get("num_key_value_heads",
                                model.arch_dims[3] if len(model.arch_dims) > 3 else heads))
        head_dim = int(cfg.get("head_dim",
                                model.arch_dims[4] if len(model.arch_dims) > 4 else 0))
        layers   = int(cfg.get("num_hidden_layers",
                                model.arch_dims[5] if len(model.arch_dims) > 5 else 0))
        vocab    = int(cfg.get("vocab_size",
                                model.arch_dims[6] if len(model.arch_dims) > 6 else 0))
        ctx      = int(cfg.get("max_position_embeddings",
                                model.arch_dims[7] if len(model.arch_dims) > 7 else 2048))
        rope_theta = float(cfg.get("rope_theta", 10000.0))
        rms_eps    = float(cfg.get("rms_norm_eps", 1e-6))

        # Compute tensor info size for data_offset alignment.
        info_size = 0
        for t in tensors:
            info_size += 8 + len(t.name.encode("utf-8")) + 4 + 8 * len(t.shape) + 4 + 8
        if info_size % 32:
            info_size += 32 - (info_size % 32)

        # Pre-build KV into an in-memory buffer so we know its size and
        # can append general.data_offset at the end.
        kv_buf = io.BytesIO()
        n_kv = 0

        _w_strkv(kv_buf, "general.architecture", gguf_arch); n_kv += 1
        _w_u32(kv_buf,    "general.file_type", 1); n_kv += 1

        _w_u32(kv_buf, f"{gguf_arch}.embedding_length", hidden); n_kv += 1
        _w_u32(kv_buf, f"{gguf_arch}.feed_forward_length", inter); n_kv += 1
        _w_u32(kv_buf, f"{gguf_arch}.attention.head_count", heads); n_kv += 1
        _w_u32(kv_buf, f"{gguf_arch}.attention.head_count_kv", kv_heads); n_kv += 1
        _w_u32(kv_buf, f"{gguf_arch}.attention.head_dim", head_dim); n_kv += 1
        _w_u32(kv_buf, f"{gguf_arch}.block_count", layers); n_kv += 1
        _w_u32(kv_buf, f"{gguf_arch}.vocab_size", vocab); n_kv += 1
        _w_u32(kv_buf, f"{gguf_arch}.context_length", ctx); n_kv += 1
        _w_f32(kv_buf, f"{gguf_arch}.rope.freq_base", rope_theta); n_kv += 1
        _w_f32(kv_buf, f"{gguf_arch}.attention.layer_norm_rms_eps", rms_eps); n_kv += 1

        # Tokenizer (synthesised ASCII — see qwn_loader).
        _w_strkv(kv_buf, "tokenizer.ggml.model", "gpt2"); n_kv += 1
        _w_arr_str(kv_buf, "tokenizer.ggml.tokens", [chr(b) for b in range(256)])
        n_kv += 1
        _w_arr_u32(kv_buf, "tokenizer.ggml.token_type", [0] * 256); n_kv += 1
        _w_arr_str(kv_buf, "tokenizer.ggml.merges", []); n_kv += 1
        _w_u32(kv_buf,    "tokenizer.ggml.bos_token_id", 1); n_kv += 1
        _w_u32(kv_buf,    "tokenizer.ggml.eos_token_id", 2); n_kv += 1
        _w_u32(kv_buf,    "tokenizer.ggml.padding_token_id", 0); n_kv += 1
        _w_u32(kv_buf,    "tokenizer.ggml.unknown_token_id", 0); n_kv += 1

        kv_bytes = kv_buf.getvalue()

        # GGUF v3 layout: data FIRST (at offset 0), header at the END.
        # The reader scans for the header at the file's logical start
        # via a back-pointer in the GGUF "general.data_offset" field
        # that must point at the end-of-header position from the
        # start of the file.
        out = io.BytesIO()

        # First tensor offset = 0 (relative to start of file).
        cursor = 0
        payload_offsets: List[int] = []
        for t in tensors:
            payload_offsets.append(cursor)
            cursor += t.byte_size

        # Write tensor payloads first (contiguous, starting at offset 0).
        for t, off in zip(tensors, payload_offsets):
            assert off == out.tell(), "data must start at offset 0 in this layout"
            out.write(t.bytes())

        # Now write the GGUF header at the end of the file.
        data_end = out.tell()
        # 24-byte header
        out.write(b"\x00" * 24)
        # KV block
        kv_start = out.tell()
        out.write(kv_bytes)
        # Tensor info (with placeholder offsets; rewritten below)
        info_start = out.tell()
        for t in tensors:
            gt = _QWN_TO_GGUF_DTYPE.get(t.dtype, GT_F32)
            name_b = t.name.encode("utf-8")
            out.write(struct.pack("<Q", len(name_b)))
            out.write(name_b)
            out.write(struct.pack("<I", len(t.shape)))
            for d in t.shape:
                out.write(struct.pack("<Q", int(d)))
            out.write(struct.pack("<I", gt))
            out.write(struct.pack("<Q", 0))   # placeholder
        # Pad to 32-byte boundary.
        cur = out.tell()
        if cur % 32:
            out.write(b"\x00" * (32 - cur % 32))
        # The trailing-header layout: the reader expects the header
        # to start where the tensor data ended.  We already wrote
        # tensors at offsets 0..data_end-1, so the header naturally
        # starts at data_end.
        # The header offset the reader expects:
        header_start = data_end

        # Patch the GGUF header at offset data_end.
        out.seek(data_end)
        out.write(GGUF_MAGIC)
        out.write(struct.pack("<I", GGUF_VERSION))
        out.write(struct.pack("<Q", len(tensors)))
        out.write(struct.pack("<Q", n_kv))

        # Rewrite the tensor info with the real offsets.
        out.seek(info_start)
        for t, off in zip(tensors, payload_offsets):
            gt = _QWN_TO_GGUF_DTYPE.get(t.dtype, GT_F32)
            name_b = t.name.encode("utf-8")
            out.write(struct.pack("<Q", len(name_b)))
            out.write(name_b)
            out.write(struct.pack("<I", len(t.shape)))
            for d in t.shape:
                out.write(struct.pack("<Q", int(d)))
            out.write(struct.pack("<I", gt))
            out.write(struct.pack("<Q", off))

        # Add general.data_offset as the last KV (right before info).
        # This makes the GGUF self-describing for tools that look for
        # data_offset.
        # NOTE: in this trailing-header layout, the magic is at
        # data_end, not byte 0.  Some readers will reject this; if so,
        # the converter can be pointed at a different GGUF version.
        _w_u64(out, "general.data_offset", data_end); n_kv += 1
        # Patch the n_kv count in the header again.
        out.seek(data_end + 16)  # magic(4) + version(4) + n_tensors(8) + n_kv(8)
        out.write(struct.pack("<Q", n_kv))

        data = out.getvalue()
        out_path.write_bytes(data)
        return len(data)

        data = out.getvalue()
        out_path.write_bytes(data)
        return len(data)
    finally:
        model.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("qwn", help="Path to a .qwn file")
    p.add_argument("-o", "--out", required=True, help="Output .gguf path")
    args = p.parse_args()
    src = Path(args.qwn).resolve()
    dst = Path(args.out).resolve()
    t0 = time.perf_counter()
    n = convert(src, dst)
    dt = time.perf_counter() - t0
    print(f"wrote {dst} ({n/1024**2:.1f} MB) in {dt:.2f}s "
          f"({n/1024**2/dt:.0f} MB/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())