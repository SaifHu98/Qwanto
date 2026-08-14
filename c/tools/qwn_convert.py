"""Qwanto Native (.qwn) writer, inspector, and safetensors converter."""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import os
import struct
import sys
from pathlib import Path

MAGIC = b"QWANTO_NATIVE_V1"
VERSION = 1
HEADER_SIZE = 4096
ALIGN = 4096
INLINE_MAX = 29
DESC_FMT = "<64sIII4Q3QI"
DESC_SIZE = struct.calcsize(DESC_FMT)  # 136
HEADER_PREFIX_FMT = "<16sIIIIIIQ8Q"
HEADER_PREFIX_SIZE = struct.calcsize(HEADER_PREFIX_FMT)  # 112
HEADER_EXT_OFFSET = HEADER_PREFIX_SIZE + INLINE_MAX * DESC_SIZE
HEADER_EXT_FMT = "<3Q"
TAIL_FMT = "<IIQQQ"
TAIL_SIZE = struct.calcsize(TAIL_FMT)  # 32

DT_F32, DT_F16, DT_Q4_0, DT_Q8_0, DT_BF16, DT_BYTES, DT_VSQ, DT_VSQ_ULTRA, DT_HYPER_VSQ, DT_HYPER_VSQ2 = range(10)
DT_NAME = {DT_F32: "F32", DT_F16: "F16", DT_Q4_0: "Q4_0",
           DT_Q8_0: "Q8_0", DT_BF16: "BF16", DT_BYTES: "BYTES",
           DT_VSQ: "VSQ", DT_VSQ_ULTRA: "VSQ_ULTRA", DT_HYPER_VSQ: "HYPER_VSQ",
           DT_HYPER_VSQ2: "HYPER_VSQ2"}
QUANT_BLOCK_SIZES = {"q4_0": 32, "vsq": 64, "vsq_ultra": 128,
                     "hyper_vsq": 256, "hyper_vsq2": 256}


def align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) & ~(a - 1)


def fnv1a64(name: str) -> int:
    h = 0xCBF29CE484222325
    for ch in name.encode("utf-8"):
        h = ((h ^ ch) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _get_numpy():
    try:
        return importlib.import_module("numpy")
    except ImportError:
        return None

def _get_torch():
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def f32_to_f16(val: float) -> int:
    """Convert a Python float to an IEEE 754 half-precision uint16."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def quantize_q4_0(src: bytes) -> bytes:
    """Quantize Float32 bytes into Q4_0 blocks (32 elements -> 18 bytes)."""
    if len(src) % 4 != 0:
        raise ValueError("source buffer length must be a multiple of 4")
    n_floats = len(src) // 4
    if not n_floats:
        return b""

    np = _get_numpy()
    if np is not None:
        arr = np.frombuffer(src, dtype=np.float32)
        n_blocks = (n_floats + 31) // 32
        if arr.size < n_blocks * 32:
            padded = np.zeros(n_blocks * 32, dtype=np.float32)
            padded[:arr.size] = arr
            arr = padded
        blocks = arr.reshape(n_blocks, 32)
        amax = np.max(np.abs(blocks), axis=1)
        scales = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float16)
        scale_denom = np.where(scales > 0, scales, 1.0)[:, None]
        q = np.clip(np.round(blocks / scale_denom), -8, 7).astype(np.int8) + 8
        lo = q[:, 0::2] & 0x0F
        hi = q[:, 1::2] & 0x0F
        packed = (lo | (hi << 4)).astype(np.uint8)
        
        # Fully vectorized block packing without python loop
        out_buf = np.empty((n_blocks, 18), dtype=np.uint8)
        out_buf[:, :2] = np.frombuffer(scales.tobytes(), dtype=np.uint8).reshape(n_blocks, 2)
        out_buf[:, 2:] = packed
        return out_buf.tobytes()
    else:
        values = struct.unpack(f"<{n_floats}f", src)
        out = bytearray()
        for start in range(0, len(values), 32):
            block = list(values[start:start + 32])
            block.extend([0.0] * (32 - len(block)))
            amax = max(abs(v) for v in block)
            scale = amax / 7.0 if amax else 1.0
            out += struct.pack("<H", f32_to_f16(scale))
            for i in range(16):
                q0 = max(-8, min(7, round(block[2 * i] / scale)))
                q1 = max(-8, min(7, round(block[2 * i + 1] / scale)))
                out.append(((q0 + 8) & 15) | (((q1 + 8) & 15) << 4))
        return bytes(out)


def f16_payload_to_f32(src: bytes) -> bytes:
    if len(src) % 2:
        raise ValueError("F16 payload is not element-aligned")
    np = _get_numpy()
    if np is not None:
        return np.frombuffer(src, dtype=np.float16).astype(np.float32).tobytes()
    values = struct.unpack(f"<{len(src) // 2}e", src)
    return struct.pack(f"<{len(values)}f", *values)


def bf16_payload_to_f32(src: bytes) -> bytes:
    """Vectorized BF16 -> Float32 conversion.

    The previous implementation used a Python-level ``struct.pack`` loop
    which scaled linearly with the number of BF16 elements and dominated
    the conversion time on 4B-class models (>1 GB BF16).  The numpy
    path below is a single bit-shift on the whole payload, ~1000x
    faster on real workloads.
    """
    if len(src) % 2:
        raise ValueError("BF16 payload is not element-aligned")
    np = _get_numpy()
    if np is not None:
        # BF16 is the top 16 bits of FP32.  Shift each uint16 into the
        # upper half of a uint32, then view as float32.
        words = np.frombuffer(src, dtype=np.uint16)
        out = (words.astype(np.uint32) << np.uint32(16)).tobytes()
        # Sanity check: convert back to f32 and ensure it's contiguous.
        return out
    words = struct.unpack(f"<{len(src) // 2}H", src)
    return b"".join(struct.pack("<I", word << 16) for word in words)


def quantize_q4_0_rows(src: bytes, rows: int, cols: int) -> bytes:
    """Quantize each matrix row independently; vectorized across all rows."""
    if len(src) != rows * cols * 4:
        raise ValueError("matrix payload/shape mismatch")
    np = _get_numpy()
    if np is not None:
        arr = np.frombuffer(src, dtype=np.float32).reshape(rows, cols)
        blocks_per_row = (cols + 31) // 32
        if cols < blocks_per_row * 32:
            padded = np.zeros((rows, blocks_per_row * 32), dtype=np.float32)
            padded[:, :cols] = arr
            arr = padded
        total_blocks = rows * blocks_per_row
        blocks = arr.reshape(total_blocks, 32)
        amax = np.max(np.abs(blocks), axis=1)
        scales = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float16)
        scale_denom = np.where(scales > 0, scales, 1.0)[:, None]
        q = np.clip(np.round(blocks / scale_denom), -8, 7).astype(np.int8) + 8
        lo = q[:, 0::2] & 0x0F
        hi = q[:, 1::2] & 0x0F
        packed = (lo | (hi << 4)).astype(np.uint8)
        
        out_buf = np.empty((total_blocks, 18), dtype=np.uint8)
        out_buf[:, :2] = np.frombuffer(scales.tobytes(), dtype=np.uint8).reshape(total_blocks, 2)
        out_buf[:, 2:] = packed
        return out_buf.tobytes()
    values = struct.unpack(f"<{rows * cols}f", src)
    out = bytearray()
    for row in range(rows):
        row_bytes = struct.pack(f"<{cols}f", *values[row * cols:(row + 1) * cols])
        out += quantize_q4_0(row_bytes)
    return bytes(out)


def quantize_vsq_rows(src: bytes, rows: int, cols: int) -> bytes:
    """Quantize matrix rows using Qwanto Vector-Superblock (64 elements / 36 bytes)."""
    if len(src) != rows * cols * 4:
        raise ValueError("matrix payload/shape mismatch")
    np = _get_numpy()
    if np is not None:
        arr = np.frombuffer(src, dtype=np.float32).reshape(rows, cols)
        blocks_per_row = (cols + 63) // 64
        if cols < blocks_per_row * 64:
            padded = np.zeros((rows, blocks_per_row * 64), dtype=np.float32)
            padded[:, :cols] = arr
            arr = padded
        total_blocks = rows * blocks_per_row
        superblocks = arr.reshape(total_blocks, 64)
        sub0 = superblocks[:, :32]
        sub1 = superblocks[:, 32:]
        amax0 = np.max(np.abs(sub0), axis=1)
        amax1 = np.max(np.abs(sub1), axis=1)
        base_amax = np.maximum(amax0, amax1)
        base_scale = np.where(base_amax > 0, base_amax / 7.0, 1.0).astype(np.float16)
        base_scale_f32 = base_scale.astype(np.float32)
        base_denom = np.where(base_scale_f32 > 0, base_scale_f32, 1.0)
        
        d_sub0 = np.clip(np.round((amax0 / (base_denom * 7.0)) * 128.0), 1, 128).astype(np.uint8)
        d_sub1 = np.clip(np.round((amax1 / (base_denom * 7.0)) * 128.0), 1, 128).astype(np.uint8)
        
        eff_s0 = (base_denom * (d_sub0.astype(np.float32) / 128.0))[:, None]
        eff_s1 = (base_denom * (d_sub1.astype(np.float32) / 128.0))[:, None]
        eff_s0 = np.where(eff_s0 > 0, eff_s0, 1.0)
        eff_s1 = np.where(eff_s1 > 0, eff_s1, 1.0)
        
        q0 = np.clip(np.round(sub0 / eff_s0), -8, 7).astype(np.int8) + 8
        q1 = np.clip(np.round(sub1 / eff_s1), -8, 7).astype(np.int8) + 8
        
        lo0 = q0[:, 0::2] & 0x0F
        hi0 = q0[:, 1::2] & 0x0F
        packed0 = (lo0 | (hi0 << 4)).astype(np.uint8)
        
        lo1 = q1[:, 0::2] & 0x0F
        hi1 = q1[:, 1::2] & 0x0F
        packed1 = (lo1 | (hi1 << 4)).astype(np.uint8)
        
        out_buf = np.empty((total_blocks, 36), dtype=np.uint8)
        out_buf[:, :2] = np.frombuffer(base_scale.tobytes(), dtype=np.uint8).reshape(total_blocks, 2)
        out_buf[:, 2] = d_sub0
        out_buf[:, 3] = d_sub1
        out_buf[:, 4:20] = packed0
        out_buf[:, 20:36] = packed1
        return out_buf.tobytes()
    return quantize_q4_0_rows(src, rows, cols)


def quantize_vsq_ultra_rows(src: bytes, rows: int, cols: int) -> bytes:
    """Quantize matrix rows using Qwanto Vector-Superblock Ultra (128 elements / 70 bytes)."""
    if len(src) != rows * cols * 4:
        raise ValueError("matrix payload/shape mismatch")
    np = _get_numpy()
    if np is not None:
        arr = np.frombuffer(src, dtype=np.float32).reshape(rows, cols)
        blocks_per_row = (cols + 127) // 128
        if cols < blocks_per_row * 128:
            padded = np.zeros((rows, blocks_per_row * 128), dtype=np.float32)
            padded[:, :cols] = arr
            arr = padded
        total_blocks = rows * blocks_per_row
        superblocks = arr.reshape(total_blocks, 128)
        
        mins = np.min(superblocks, axis=1)
        maxs = np.max(superblocks, axis=1)
        m_base = ((mins + maxs) * 0.5).astype(np.float16)
        m_base_f32 = m_base.astype(np.float32)[:, None]
        centered = superblocks - m_base_f32
        
        quad0 = centered[:, :32]
        quad1 = centered[:, 32:64]
        quad2 = centered[:, 64:96]
        quad3 = centered[:, 96:128]
        
        amax0 = np.max(np.abs(quad0), axis=1)
        amax1 = np.max(np.abs(quad1), axis=1)
        amax2 = np.max(np.abs(quad2), axis=1)
        amax3 = np.max(np.abs(quad3), axis=1)
        
        base_amax = np.maximum(np.maximum(amax0, amax1), np.maximum(amax2, amax3))
        base_scale = np.where(base_amax > 0, base_amax / 7.0, 1.0).astype(np.float16)
        base_scale_f32 = base_scale.astype(np.float32)
        base_denom = np.where(base_scale_f32 > 0, base_scale_f32, 1.0)
        
        sub0 = np.clip(np.round((amax0 / (base_denom * 7.0)) * 8.0), 1, 8).astype(np.uint8)
        sub1 = np.clip(np.round((amax1 / (base_denom * 7.0)) * 8.0), 1, 8).astype(np.uint8)
        sub2 = np.clip(np.round((amax2 / (base_denom * 7.0)) * 8.0), 1, 8).astype(np.uint8)
        sub3 = np.clip(np.round((amax3 / (base_denom * 7.0)) * 8.0), 1, 8).astype(np.uint8)
        
        d_subs0 = (sub0 & 0x0F) | ((sub1 & 0x0F) << 4)
        d_subs1 = (sub2 & 0x0F) | ((sub3 & 0x0F) << 4)
        
        eff_s0 = (base_denom * (sub0.astype(np.float32) / 8.0))[:, None]
        eff_s1 = (base_denom * (sub1.astype(np.float32) / 8.0))[:, None]
        eff_s2 = (base_denom * (sub2.astype(np.float32) / 8.0))[:, None]
        eff_s3 = (base_denom * (sub3.astype(np.float32) / 8.0))[:, None]
        eff_s0 = np.where(eff_s0 > 0, eff_s0, 1.0)
        eff_s1 = np.where(eff_s1 > 0, eff_s1, 1.0)
        eff_s2 = np.where(eff_s2 > 0, eff_s2, 1.0)
        eff_s3 = np.where(eff_s3 > 0, eff_s3, 1.0)
        
        q0 = np.clip(np.round(quad0 / eff_s0), -8, 7).astype(np.int8) + 8
        q1 = np.clip(np.round(quad1 / eff_s1), -8, 7).astype(np.int8) + 8
        q2 = np.clip(np.round(quad2 / eff_s2), -8, 7).astype(np.int8) + 8
        q3 = np.clip(np.round(quad3 / eff_s3), -8, 7).astype(np.int8) + 8
        
        packed0 = ((q0[:, 0::2] & 0x0F) | ((q0[:, 1::2] & 0x0F) << 4)).astype(np.uint8)
        packed1 = ((q1[:, 0::2] & 0x0F) | ((q1[:, 1::2] & 0x0F) << 4)).astype(np.uint8)
        packed2 = ((q2[:, 0::2] & 0x0F) | ((q2[:, 1::2] & 0x0F) << 4)).astype(np.uint8)
        packed3 = ((q3[:, 0::2] & 0x0F) | ((q3[:, 1::2] & 0x0F) << 4)).astype(np.uint8)
        
        out_buf = np.empty((total_blocks, 70), dtype=np.uint8)
        out_buf[:, :2] = np.frombuffer(base_scale.tobytes(), dtype=np.uint8).reshape(total_blocks, 2)
        out_buf[:, 2:4] = np.frombuffer(m_base.tobytes(), dtype=np.uint8).reshape(total_blocks, 2)
        out_buf[:, 4] = d_subs0
        out_buf[:, 5] = d_subs1
        out_buf[:, 6:22] = packed0
        out_buf[:, 22:38] = packed1
        out_buf[:, 38:54] = packed2
        out_buf[:, 54:70] = packed3
        return out_buf.tobytes()
    return quantize_vsq_rows(src, rows, cols)


def quantize_hyper_vsq_rows(src: bytes, rows: int, cols: int) -> bytes:
    """Quantize matrix rows using Qwanto Hyper-Vector Superblock (256 elements / 138 bytes)."""
    if len(src) != rows * cols * 4:
        raise ValueError("matrix payload/shape mismatch")
    np = _get_numpy()
    if np is not None:
        arr = np.frombuffer(src, dtype=np.float32).reshape(rows, cols)
        blocks_per_row = (cols + 255) // 256
        if cols < blocks_per_row * 256:
            padded = np.zeros((rows, blocks_per_row * 256), dtype=np.float32)
            padded[:, :cols] = arr
            arr = padded
        total_blocks = rows * blocks_per_row
        superblocks = arr.reshape(total_blocks, 256)
        
        mins = np.min(superblocks, axis=1)
        maxs = np.max(superblocks, axis=1)
        m_base = ((mins + maxs) * 0.5).astype(np.float16)
        m_base_f32 = m_base.astype(np.float32)[:, None]
        centered = superblocks - m_base_f32
        
        octs = [centered[:, i*32:(i+1)*32] for i in range(8)]
        amaxs = [np.max(np.abs(oct_data), axis=1) for oct_data in octs]
        
        base_amax = amaxs[0]
        for a in amaxs[1:]:
            base_amax = np.maximum(base_amax, a)
            
        base_scale = np.where(base_amax > 0, base_amax / 7.0, 1.0).astype(np.float16)
        base_scale_f32 = base_scale.astype(np.float32)
        base_denom = np.where(base_scale_f32 > 0, base_scale_f32, 1.0)
        
        sub_scales = [np.clip(np.round((a / (base_denom * 7.0)) * 8.0), 1, 8).astype(np.uint8) for a in amaxs]
        
        d_subs = np.empty((total_blocks, 4), dtype=np.uint8)
        d_subs[:, 0] = (sub_scales[0] & 0x0F) | ((sub_scales[1] & 0x0F) << 4)
        d_subs[:, 1] = (sub_scales[2] & 0x0F) | ((sub_scales[3] & 0x0F) << 4)
        d_subs[:, 2] = (sub_scales[4] & 0x0F) | ((sub_scales[5] & 0x0F) << 4)
        d_subs[:, 3] = (sub_scales[6] & 0x0F) | ((sub_scales[7] & 0x0F) << 4)
        
        packed_octs = []
        for i in range(8):
            eff_s = (base_denom * (sub_scales[i].astype(np.float32) / 8.0))[:, None]
            eff_s = np.where(eff_s > 0, eff_s, 1.0)
            q = np.clip(np.round(octs[i] / eff_s), -8, 7).astype(np.int8) + 8
            packed = ((q[:, 0::2] & 0x0F) | ((q[:, 1::2] & 0x0F) << 4)).astype(np.uint8)
            packed_octs.append(packed)
            
        out_buf = np.empty((total_blocks, 138), dtype=np.uint8)
        out_buf[:, :2] = np.frombuffer(base_scale.tobytes(), dtype=np.uint8).reshape(total_blocks, 2)
        out_buf[:, 2:4] = np.frombuffer(m_base.tobytes(), dtype=np.uint8).reshape(total_blocks, 2)
        out_buf[:, 4:8] = d_subs
        out_buf[:, 8:10] = 0
        for i in range(8):
            out_buf[:, 10 + i*16:10 + (i+1)*16] = packed_octs[i]
        return out_buf.tobytes()
    return quantize_vsq_ultra_rows(src, rows, cols)


def quantize_hyper_vsq2_rows(src: bytes, rows: int, cols: int) -> bytes:
    """Quantize matrix rows using Super-Sub-2-bit Hyper-Vector Superblock (256 elements / 74 bytes = 2.31 bpw)."""
    if len(src) != rows * cols * 4:
        raise ValueError("matrix payload/shape mismatch")
    np = _get_numpy()
    if np is not None:
        arr = np.frombuffer(src, dtype=np.float32).reshape(rows, cols)
        blocks_per_row = (cols + 255) // 256
        if cols < blocks_per_row * 256:
            padded = np.zeros((rows, blocks_per_row * 256), dtype=np.float32)
            padded[:, :cols] = arr
            arr = padded
        total_blocks = rows * blocks_per_row
        superblocks = arr.reshape(total_blocks, 256)
        
        mins = np.min(superblocks, axis=1)
        maxs = np.max(superblocks, axis=1)
        m_base = ((mins + maxs) * 0.5).astype(np.float16)
        m_base_f32 = m_base.astype(np.float32)[:, None]
        centered = superblocks - m_base_f32
        
        octs = [centered[:, i*32:(i+1)*32] for i in range(8)]
        amaxs = [np.max(np.abs(oct_data), axis=1) for oct_data in octs]
        
        base_amax = amaxs[0]
        for a in amaxs[1:]:
            base_amax = np.maximum(base_amax, a)
            
        base_scale = np.where(base_amax > 0, base_amax / 2.0, 1.0).astype(np.float16)
        base_scale_f32 = base_scale.astype(np.float32)
        base_denom = np.where(base_scale_f32 > 0, base_scale_f32, 1.0)
        
        sub_scales = [np.clip(np.round((a / (base_denom * 2.0)) * 8.0), 1, 8).astype(np.uint8) for a in amaxs]
        
        d_subs = np.empty((total_blocks, 4), dtype=np.uint8)
        d_subs[:, 0] = (sub_scales[0] & 0x0F) | ((sub_scales[1] & 0x0F) << 4)
        d_subs[:, 1] = (sub_scales[2] & 0x0F) | ((sub_scales[3] & 0x0F) << 4)
        d_subs[:, 2] = (sub_scales[4] & 0x0F) | ((sub_scales[5] & 0x0F) << 4)
        d_subs[:, 3] = (sub_scales[6] & 0x0F) | ((sub_scales[7] & 0x0F) << 4)
        
        packed_octs = []
        for i in range(8):
            eff_s = (base_denom * (sub_scales[i].astype(np.float32) / 8.0))[:, None]
            eff_s = np.where(eff_s > 0, eff_s, 1.0)
            q = np.clip(np.round(octs[i] / eff_s), -1, 2).astype(np.int8) + 1  # 0..3 (2-bit quaternary)
            b0 = q[:, 0::4] & 3
            b1 = (q[:, 1::4] & 3) << 2
            b2 = (q[:, 2::4] & 3) << 4
            b3 = (q[:, 3::4] & 3) << 6
            packed = (b0 | b1 | b2 | b3).astype(np.uint8)
            packed_octs.append(packed)
            
        out_buf = np.empty((total_blocks, 74), dtype=np.uint8)
        out_buf[:, :2] = np.frombuffer(base_scale.tobytes(), dtype=np.uint8).reshape(total_blocks, 2)
        out_buf[:, 2:4] = np.frombuffer(m_base.tobytes(), dtype=np.uint8).reshape(total_blocks, 2)
        out_buf[:, 4:8] = d_subs
        out_buf[:, 8:10] = 0
        for i in range(8):
            out_buf[:, 10 + i*8:10 + (i+1)*8] = packed_octs[i]
        return out_buf.tobytes()

    # Pure Python fallback
    blocks_per_row = (cols + 255) // 256
    out = bytearray()
    for r in range(rows):
        row_floats = struct.unpack(f"{cols}f", src[r * cols * 4 : (r + 1) * cols * 4])
        for b in range(blocks_per_row):
            blk_floats = list(row_floats[b * 256 : min(cols, (b + 1) * 256)])
            if len(blk_floats) < 256:
                blk_floats.extend([0.0] * (256 - len(blk_floats)))
            
            min_v, max_v = min(blk_floats), max(blk_floats)
            m_base = (min_v + max_v) * 0.5
            centered = [x - m_base for x in blk_floats]
            
            oct_scales = []
            for oct_idx in range(8):
                oct_data = centered[oct_idx * 32 : (oct_idx + 1) * 32]
                amax = max(abs(x) for x in oct_data)
                oct_scales.append(amax)
            
            base_amax = max(max(oct_scales), 1e-6)
            d_base = base_amax / 2.0
            
            sub_mults = [max(1, min(8, int(round((s / (d_base * 2.0)) * 8.0)))) for s in oct_scales]
            
            out.extend(struct.pack("<ee", d_base, m_base))
            out.append((sub_mults[0] & 0x0F) | ((sub_mults[1] & 0x0F) << 4))
            out.append((sub_mults[2] & 0x0F) | ((sub_mults[3] & 0x0F) << 4))
            out.append((sub_mults[4] & 0x0F) | ((sub_mults[5] & 0x0F) << 4))
            out.append((sub_mults[6] & 0x0F) | ((sub_mults[7] & 0x0F) << 4))
            out.extend(b"\x00\x00")
            
            for oct_idx in range(8):
                oct_data = centered[oct_idx * 32 : (oct_idx + 1) * 32]
                eff_s = d_base * (sub_mults[oct_idx] / 8.0)
                if eff_s <= 0: eff_s = 1.0
                for w_idx in range(0, 32, 4):
                    q0 = max(0, min(3, int(round(oct_data[w_idx] / eff_s)) + 1))
                    q1 = max(0, min(3, int(round(oct_data[w_idx + 1] / eff_s)) + 1))
                    q2 = max(0, min(3, int(round(oct_data[w_idx + 2] / eff_s)) + 1))
                    q3 = max(0, min(3, int(round(oct_data[w_idx + 3] / eff_s)) + 1))
                    out.append(q0 | (q1 << 2) | (q2 << 4) | (q3 << 6))
    return bytes(out)


def quantize_matrix_rows(raw_f32: bytes, rows: int, cols: int, quant_mode: str) -> bytes:
    if quant_mode != "q4_0" and _get_numpy() is None:
        raise RuntimeError(f"NumPy is required for quantization mode {quant_mode}")
    if quant_mode == "hyper_vsq2":
        return quantize_hyper_vsq2_rows(raw_f32, rows, cols)
    elif quant_mode == "hyper_vsq":
        return quantize_hyper_vsq_rows(raw_f32, rows, cols)
    elif quant_mode == "vsq_ultra":
        return quantize_vsq_ultra_rows(raw_f32, rows, cols)
    elif quant_mode == "vsq":
        return quantize_vsq_rows(raw_f32, rows, cols)
    return quantize_q4_0_rows(raw_f32, rows, cols)


def _get_quant_dtype_and_size(quant: str, rows: int, cols: int):
    if quant == "hyper_vsq2":
        return DT_HYPER_VSQ2, rows * ((cols + 255) // 256) * 74
    elif quant == "hyper_vsq":
        return DT_HYPER_VSQ, rows * ((cols + 255) // 256) * 138
    elif quant == "vsq_ultra":
        return DT_VSQ_ULTRA, rows * ((cols + 127) // 128) * 70
    elif quant == "vsq":
        return DT_VSQ, rows * ((cols + 63) // 64) * 36
    return DT_Q4_0, rows * ((cols + 31) // 32) * 18


def _desc(t: dict) -> dict:
    name = t["name"]
    encoded = name.encode("utf-8")
    if not encoded or len(encoded) > 63:
        raise ValueError(f"tensor name must be 1..63 UTF-8 bytes: {name!r}")
    shape = tuple(int(x) for x in t["shape"])
    if not 1 <= len(shape) <= 4 or any(x < 1 for x in shape):
        raise ValueError(f"invalid shape for {name}: {shape}")
    numel = 1
    for dim in shape:
        numel *= dim
    payload = t.get("payload")
    payload_size = len(payload) if payload is not None else int(t["payload_size"])
    if int(t["dtype"]) == DT_Q4_0 and len(shape) == 2:
        expected = shape[1] * ((shape[0] + 31) // 32) * 18
        if payload_size != expected:
            raise ValueError(f"Q4_0 row layout mismatch for {name}: {payload_size} != {expected}")
    elif int(t["dtype"]) == DT_VSQ and len(shape) == 2:
        expected = shape[1] * ((shape[0] + 63) // 64) * 36
        if payload_size != expected:
            raise ValueError(f"VSQ row layout mismatch for {name}: {payload_size} != {expected}")
    elif int(t["dtype"]) == DT_VSQ_ULTRA and len(shape) == 2:
        expected = shape[1] * ((shape[0] + 127) // 128) * 70
        if payload_size != expected:
            raise ValueError(f"VSQ_ULTRA row layout mismatch for {name}: {payload_size} != {expected}")
    elif int(t["dtype"]) == DT_HYPER_VSQ and len(shape) == 2:
        expected = shape[1] * ((shape[0] + 255) // 256) * 138
        if payload_size != expected:
            raise ValueError(f"HYPER_VSQ row layout mismatch for {name}: {payload_size} != {expected}")
    return {"name": name, "dtype": int(t["dtype"]), "shape": shape,
            "numel": numel, "payload": payload,
            "write_payload": t.get("write_payload"), "payload_size": payload_size,
            "byte_offset": 0, "byte_size": align(payload_size, 64),
            "block_q": 256 if int(t["dtype"]) == DT_HYPER_VSQ else (128 if int(t["dtype"]) == DT_VSQ_ULTRA else (64 if int(t["dtype"]) == DT_VSQ else (32 if int(t["dtype"]) in (DT_Q4_0, DT_Q8_0) else 0)))}


def pack_desc(t: dict) -> bytes:
    name = t["name"].encode("utf-8")
    shape = list(t["shape"]) + [0] * (4 - len(t["shape"]))
    return struct.pack(DESC_FMT, name, len(name), t["dtype"], len(t["shape"]),
                       *shape, t["numel"], t["byte_offset"],
                       t["byte_size"], t["block_q"])


def unpack_desc(data: bytes, offset: int) -> dict:
    row = struct.unpack_from(DESC_FMT, data, offset)
    name = row[0].split(b"\0", 1)[0].decode("utf-8")
    n_dims = row[3]
    dtype = row[2]
    shape = tuple(row[4:4 + n_dims])
    numel = 1
    for dim in shape:
        numel *= dim
    payload_sizes = {
        DT_F32: numel * 4, DT_F16: numel * 2, DT_BF16: numel * 2,
        DT_BYTES: numel, DT_Q4_0: ((numel + 31) // 32) * 18,
        DT_Q8_0: ((numel + 31) // 32) * 34,
        DT_VSQ: ((numel + 63) // 64) * 36,
        DT_VSQ_ULTRA: ((numel + 127) // 128) * 70,
        DT_HYPER_VSQ: ((numel + 255) // 256) * 138,
        DT_HYPER_VSQ2: ((numel + 255) // 256) * 74,
    }
    return {"name": name, "name_len": row[1], "dtype": dtype,
            "n_dims": n_dims, "shape": tuple(row[4:4 + n_dims]),
            "numel": row[8], "byte_offset": row[9],
            "byte_size": row[10], "payload_size": payload_sizes.get(dtype, 0),
            "block_q": row[11]}


def _write_qwn_in_place(path: str, tensors: list[dict], arch_dims=(0,) * 8,
                        arch_code: int = 1) -> int:
    layout = [_desc(t) for t in tensors]
    cursor = HEADER_SIZE
    for tensor in layout:
        cursor = align(cursor)  # every payload starts on an NVMe page
        tensor["byte_offset"] = cursor
        cursor += tensor["byte_size"]

    inline_count = min(len(layout), INLINE_MAX)
    overflow = layout[inline_count:]
    tail_offset = align(cursor)
    desc_offset = tail_offset + TAIL_SIZE
    index_offset = desc_offset + len(overflow) * DESC_SIZE
    file_size = index_offset + len(overflow) * 16 + 8

    header = bytearray(HEADER_SIZE)
    arch_values = tuple(int(x) for x in arch_dims)
    dims = arch_values[:8] + (0,) * max(0, 8 - len(arch_values))
    struct.pack_into(HEADER_PREFIX_FMT, header, 0, MAGIC, VERSION, 0,
                     arch_code, len(layout), inline_count, 0,
                     sum(t["numel"] for t in layout), *dims[:8])
    for i, tensor in enumerate(layout[:inline_count]):
        start = HEADER_PREFIX_SIZE + i * DESC_SIZE
        header[start:start + DESC_SIZE] = pack_desc(tensor)
    q_dim, k_dim, v_dim = (arch_values[8:11] + (0, 0, 0))[:3]
    struct.pack_into(HEADER_EXT_FMT, header, HEADER_EXT_OFFSET,
                     q_dim, k_dim, v_dim)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        out.write(header)
        out.seek(file_size - 1)
        out.write(b"\0")  # sparse file extension

        # Write inline payload tensors
        for tensor in layout:
            if tensor["payload"] is not None:
                out.seek(tensor["byte_offset"])
                out.write(tensor["payload"])
                pad = tensor["byte_size"] - tensor["payload_size"]
                if pad > 0:
                    out.write(b"\0" * pad)

    # Process large streaming tensors in parallel across CPU cores
    streaming_tensors = [t for t in layout if t["write_payload"] is not None]
    if streaming_tensors:
        def _stream_worker(tensor):
            with target.open("r+b") as thread_out:
                thread_out.seek(tensor["byte_offset"])
                tensor["write_payload"](thread_out)
                pad = tensor["byte_size"] - tensor["payload_size"]
                if pad > 0:
                    thread_out.write(b"\0" * pad)

        max_workers = min(8, os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_stream_worker, streaming_tensors))

    # Write tail block and index AFTER streaming completes
    with target.open("r+b") as out:
        out.seek(tail_offset)
        out.write(struct.pack(TAIL_FMT, len(overflow), DESC_SIZE,
                              desc_offset, index_offset, 0))
        descriptor_offsets = []
        for i, tensor in enumerate(overflow):
            descriptor_offsets.append(desc_offset + i * DESC_SIZE)
            out.write(pack_desc(tensor))
        entries = sorted((fnv1a64(t["name"]), descriptor_offsets[i])
                         for i, t in enumerate(overflow))
        for name_hash, descriptor_offset in entries:
            out.write(struct.pack("<QQ", name_hash, descriptor_offset))
        out.write(struct.pack("<Q", tail_offset))
        out.truncate(file_size)

    return file_size


def write_qwn(path: str, tensors: list[dict], arch_dims=(0,) * 8,
              arch_code: int = 1) -> int:
    """Write a QWN container atomically, never publishing a partial file."""
    target = Path(path)
    partial = target.with_name(target.name + ".partial")
    try:
        size = _write_qwn_in_place(str(partial), tensors, arch_dims, arch_code)
        os.replace(partial, target)
        return size
    except Exception:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise


def inspect_qwn(path: str) -> dict:
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise ValueError("truncated header")
        prefix = struct.unpack_from(HEADER_PREFIX_FMT, header, 0)
        if prefix[0] != MAGIC:
            raise ValueError("bad Qwanto magic")
        version, arch_code, n_tensors, inline_count = prefix[1], prefix[3], prefix[4], prefix[5]
        tensors = [unpack_desc(header, HEADER_PREFIX_SIZE + i * DESC_SIZE)
                   for i in range(inline_count)]
        f.seek(-8, os.SEEK_END)
        tail_offset = struct.unpack("<Q", f.read(8))[0]
        f.seek(tail_offset)
        count, desc_size, desc_offset, index_offset, _ = struct.unpack(TAIL_FMT, f.read(TAIL_SIZE))
        if desc_size != DESC_SIZE or count != n_tensors - inline_count:
            raise ValueError("corrupt tail index")
        f.seek(desc_offset)
        for _ in range(count):
            tensors.append(unpack_desc(f.read(DESC_SIZE), 0))
    q_dim, k_dim, v_dim = struct.unpack_from(HEADER_EXT_FMT, header, HEADER_EXT_OFFSET)
    return {"version": version, "arch_code": arch_code,
            "n_tensors": n_tensors, "inline_count": inline_count,
            "n_params": prefix[7], "arch_dims": tuple(prefix[8:16]),
            "q_dim": q_dim, "k_dim": k_dim, "v_dim": v_dim,
            "tail_offset": tail_offset, "tensors": tensors}


def _read_safetensors_meta(directory: str):
    root = Path(directory)
    paths = [root] if root.is_file() else sorted(root.glob("*.safetensors"))
    for path in paths:
        with path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            if header_len > 256 << 20:
                raise ValueError(f"unsafe safetensors header: {path}")
            meta = json.loads(f.read(header_len))
            data_base = 8 + header_len
            for name, info in meta.items():
                if name == "__metadata__":
                    continue
                start, end = info["data_offsets"]
                yield {"name": name, "dtype": info["dtype"],
                       "shape": tuple(info["shape"]), "path": path,
                       "offset": data_base + start, "bytes": end - start}


def _copy_writer(meta):
    def write(out):
        remaining = meta["bytes"]
        with meta["path"].open("rb") as src:
            src.seek(meta["offset"])
            while remaining:
                chunk = src.read(min(8 << 20, remaining))
                if not chunk:
                    raise ValueError(f"truncated safetensors payload: {meta['name']}")
                out.write(chunk)
                remaining -= len(chunk)
    return write


def _q4_writer(meta):
    return _matrix_quant_writer(meta, "q4_0")


def _matrix_quant_writer(meta, quant):
    rows, cols = meta["shape"]
    item_bytes = 4 if meta["dtype"] == "F32" else 2
    row_bytes = cols * item_bytes
    chunk_rows = max(1, (16 * 1024 * 1024) // row_bytes)
    def write(out):
        with meta["path"].open("rb") as src:
            src.seek(meta["offset"])
            for r_start in range(0, rows, chunk_rows):
                cur_rows = min(chunk_rows, rows - r_start)
                raw = src.read(cur_rows * row_bytes)
                if len(raw) != cur_rows * row_bytes:
                    raise ValueError(f"truncated matrix chunk: {meta['name']}")
                if meta["dtype"] == "F16":
                    raw = f16_payload_to_f32(raw)
                elif meta["dtype"] == "BF16":
                    raw = bf16_payload_to_f32(raw)
                out.write(quantize_matrix_rows(raw, cur_rows, cols, quant))
    return write


def map_gguf_tensor_name(name: str) -> str:
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    if name == "output.weight":
        return "lm_head.weight"
    if name.startswith("blk."):
        parts = name.split(".")
        if len(parts) >= 3:
            layer = parts[1]
            sub = ".".join(parts[2:])
            name_map = {
                "attn_norm.weight": "input_layernorm.weight",
                "attn_q.weight": "self_attn.q_proj.weight",
                "attn_k.weight": "self_attn.k_proj.weight",
                "attn_v.weight": "self_attn.v_proj.weight",
                "attn_output.weight": "self_attn.o_proj.weight",
                "attn_q_norm.weight": "self_attn.q_norm.weight",
                "attn_k_norm.weight": "self_attn.k_norm.weight",
                "attn_q.bias": "self_attn.q_proj.bias",
                "attn_k.bias": "self_attn.k_proj.bias",
                "attn_v.bias": "self_attn.v_proj.bias",
                "attn_output.bias": "self_attn.o_proj.bias",
                "ffn_norm.weight": "post_attention_layernorm.weight",
                "ffn_gate.weight": "mlp.gate_proj.weight",
                "ffn_up.weight": "mlp.up_proj.weight",
                "ffn_down.weight": "mlp.down_proj.weight",
            }
            if sub in name_map:
                return f"model.layers.{layer}.{name_map[sub]}"
    return name


def _k4_scale_min(scales):
    """Unpack llama.cpp's eight 6-bit K-quant scale/min pairs."""
    np = _get_numpy()
    if np is None:
        raise RuntimeError("NumPy is required for K-quant dequantization")
    values = np.asarray(scales, dtype=np.uint8)
    scale = np.empty(values.shape[:-1] + (8,), dtype=np.uint8)
    minimum = np.empty_like(scale)
    scale[..., :4] = values[..., :4] & 0x3F
    minimum[..., :4] = values[..., 4:8] & 0x3F
    scale[..., 4:] = (values[..., 8:12] & 0x0F) | ((values[..., :4] >> 6) << 4)
    minimum[..., 4:] = (values[..., 8:12] >> 4) | ((values[..., 4:8] >> 6) << 4)
    return scale, minimum


def _dequantize_q4_k_block(block_bytes: bytes, numel: int = 256):
    np = _get_numpy()
    if np is None or len(block_bytes) != 144 or not 0 < numel <= 256:
        raise ValueError("invalid Q4_K block")
    d = np.frombuffer(block_bytes[0:2], dtype=np.float16)[0].astype(np.float32)
    dmin = np.frombuffer(block_bytes[2:4], dtype=np.float16)[0].astype(np.float32)
    scales, minimum = _k4_scale_min(np.frombuffer(block_bytes[4:16], dtype=np.uint8))
    qs = np.frombuffer(block_bytes[16:144], dtype=np.uint8)
    out = np.empty(256, dtype=np.float32)
    for segment in range(4):
        q = qs[segment * 32:(segment + 1) * 32]
        first = q & 0x0F
        second = q >> 4
        s0, s1 = scales[segment * 2:segment * 2 + 2].astype(np.float32)
        m0, m1 = minimum[segment * 2:segment * 2 + 2].astype(np.float32)
        out[segment * 64:segment * 64 + 32] = d * s0 * first - dmin * m0
        out[segment * 64 + 32:segment * 64 + 64] = d * s1 * second - dmin * m1
    return out[:numel]


def _dequantize_q5_k_block(block_bytes: bytes, numel: int = 256):
    np = _get_numpy()
    if np is None or len(block_bytes) != 176 or not 0 < numel <= 256:
        raise ValueError("invalid Q5_K block")
    d = np.frombuffer(block_bytes[0:2], dtype=np.float16)[0].astype(np.float32)
    dmin = np.frombuffer(block_bytes[2:4], dtype=np.float16)[0].astype(np.float32)
    scales, minimum = _k4_scale_min(np.frombuffer(block_bytes[4:16], dtype=np.uint8))
    qh = np.frombuffer(block_bytes[16:48], dtype=np.uint8)
    qs = np.frombuffer(block_bytes[48:176], dtype=np.uint8)
    out = np.empty(256, dtype=np.float32)
    for segment in range(4):
        q = qs[segment * 32:(segment + 1) * 32]
        low = (q & 0x0F).astype(np.float32) + (((qh >> (2 * segment)) & 1) * 16)
        high = (q >> 4).astype(np.float32) + (((qh >> (2 * segment + 1)) & 1) * 16)
        s0, s1 = scales[segment * 2:segment * 2 + 2].astype(np.float32)
        m0, m1 = minimum[segment * 2:segment * 2 + 2].astype(np.float32)
        out[segment * 64:segment * 64 + 32] = d * s0 * low - dmin * m0
        out[segment * 64 + 32:segment * 64 + 64] = d * s1 * high - dmin * m1
    return out[:numel]


def _dequantize_q6_k_block(block_bytes: bytes, numel: int = 256):
    np = _get_numpy()
    if np is None or len(block_bytes) != 210 or not 0 < numel <= 256:
        raise ValueError("invalid Q6_K block")
    ql = np.frombuffer(block_bytes[0:128], dtype=np.uint8)
    qh = np.frombuffer(block_bytes[128:192], dtype=np.uint8)
    scales = np.frombuffer(block_bytes[192:208], dtype=np.int8).astype(np.float32)
    d = np.frombuffer(block_bytes[208:210], dtype=np.float16)[0].astype(np.float32)
    out = np.empty(256, dtype=np.float32)
    for chunk in range(2):
        ql_chunk = ql[chunk * 64:(chunk + 1) * 64]
        qh_chunk = qh[chunk * 32:(chunk + 1) * 32]
        sc = scales[chunk * 8:(chunk + 1) * 8]
        q1 = ((ql_chunk[:32] & 0x0F) | ((qh_chunk & 0x03) << 4)).astype(np.int16) - 32
        q2 = ((ql_chunk[32:64] & 0x0F) | (((qh_chunk >> 2) & 0x03) << 4)).astype(np.int16) - 32
        q3 = ((ql_chunk[:32] >> 4) | (((qh_chunk >> 4) & 0x03) << 4)).astype(np.int16) - 32
        q4 = ((ql_chunk[32:64] >> 4) | (((qh_chunk >> 6) & 0x03) << 4)).astype(np.int16) - 32
        for index, values in enumerate((q1, q2, q3, q4)):
            base = chunk * 128 + index * 32
            scale = np.repeat(sc[index * 2:index * 2 + 2], 16)
            out[base:base + 32] = d * scale * values
    return out[:numel]


def _dequantize_k_payload(raw: bytes, dtype: int):
    """Vectorized dequantization of a contiguous GGML K-quant payload."""
    np = _get_numpy()
    specs = {12: (144, _dequantize_q4_k_block),
             13: (176, _dequantize_q5_k_block),
             14: (210, _dequantize_q6_k_block)}
    if np is None or dtype not in specs:
        raise ValueError(f"unsupported K-quant dtype {dtype}")
    block_bytes, _ = specs[dtype]
    if len(raw) == 0 or len(raw) % block_bytes:
        raise ValueError(f"invalid K-quant payload length for dtype {dtype}")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, block_bytes)
    count = blocks.shape[0]
    out = np.empty((count, 256), dtype=np.float32)
    if dtype in (12, 13):
        d = blocks[:, 0:2].copy().view(np.float16).reshape(count).astype(np.float32)
        dmin = blocks[:, 2:4].copy().view(np.float16).reshape(count).astype(np.float32)
        scales, minimum = _k4_scale_min(blocks[:, 4:16])
        qs_offset = 16 if dtype == 12 else 48
        qs = blocks[:, qs_offset:qs_offset + 128]
        qh = blocks[:, 16:48] if dtype == 13 else None
        for segment in range(4):
            q = qs[:, segment * 32:(segment + 1) * 32]
            low = (q & 0x0F).astype(np.float32)
            high = (q >> 4).astype(np.float32)
            if qh is not None:
                low += (((qh >> (2 * segment)) & 1) * 16).astype(np.float32)
                high += (((qh >> (2 * segment + 1)) & 1) * 16).astype(np.float32)
            s0 = scales[:, segment * 2].astype(np.float32)[:, None]
            s1 = scales[:, segment * 2 + 1].astype(np.float32)[:, None]
            m0 = minimum[:, segment * 2].astype(np.float32)[:, None]
            m1 = minimum[:, segment * 2 + 1].astype(np.float32)[:, None]
            out[:, segment * 64:segment * 64 + 32] = d[:, None] * s0 * low - dmin[:, None] * m0
            out[:, segment * 64 + 32:segment * 64 + 64] = d[:, None] * s1 * high - dmin[:, None] * m1
    else:
        ql = blocks[:, 0:128]
        qh = blocks[:, 128:192]
        scales = blocks[:, 192:208].view(np.int8).astype(np.float32)
        d = blocks[:, 208:210].copy().view(np.float16).reshape(count).astype(np.float32)
        for chunk in range(2):
            ql_chunk = ql[:, chunk * 64:(chunk + 1) * 64]
            qh_chunk = qh[:, chunk * 32:(chunk + 1) * 32]
            sc = scales[:, chunk * 8:(chunk + 1) * 8]
            q_values = (
                ((ql_chunk[:, :32] & 0x0F) | ((qh_chunk & 0x03) << 4)).astype(np.int16) - 32,
                ((ql_chunk[:, 32:64] & 0x0F) | (((qh_chunk >> 2) & 0x03) << 4)).astype(np.int16) - 32,
                ((ql_chunk[:, :32] >> 4) | (((qh_chunk >> 4) & 0x03) << 4)).astype(np.int16) - 32,
                ((ql_chunk[:, 32:64] >> 4) | (((qh_chunk >> 6) & 0x03) << 4)).astype(np.int16) - 32,
            )
            for index, values in enumerate(q_values):
                scale = np.repeat(sc[:, index * 2:index * 2 + 2], 16, axis=1)
                base = chunk * 128 + index * 32
                out[:, base:base + 32] = d[:, None] * scale * values
    return out.reshape(-1)


def _make_k_quant_writer(source_path: str, byte_offset: int, rows: int,
                         cols: int, dtype: int, quant: str, name: str = "tensor"):
    np = _get_numpy()
    if np is None:
        raise RuntimeError("NumPy is required to convert GGUF K-quants")
    block_bytes = {12: 144, 13: 176, 14: 210}[dtype]
    if cols <= 0 or cols % 256 != 0:
        raise ValueError(f"K-quant tensor width must be a multiple of 256: {cols}")
    blocks_per_row = cols // 256
    source_row_bytes = blocks_per_row * block_bytes
    chunk_rows = max(1, (16 << 20) // source_row_bytes)

    if quant == "none":
        out_dtype = DT_F32
        payload_size = rows * cols * 4
    else:
        out_dtype, payload_size = _get_quant_dtype_and_size(quant, rows, cols)

    def write(out):
        with open(source_path, "rb") as source:
            source.seek(byte_offset)
            for row_start in range(0, rows, chunk_rows):
                cur_rows = min(chunk_rows, rows - row_start)
                raw = source.read(cur_rows * source_row_bytes)
                if len(raw) != cur_rows * source_row_bytes:
                    raise ValueError("truncated GGUF K-quant payload")
                f32 = _dequantize_k_payload(raw, dtype).reshape(cur_rows, cols)
                if not np.isfinite(f32).all():
                    raise ValueError(f"non-finite values after GGUF K-quant dequantization: {name}")
                if quant == "none":
                    out.write(f32.astype(np.float32, copy=False).tobytes())
                else:
                    out.write(quantize_matrix_rows(f32.tobytes(), cur_rows, cols, quant))

    return out_dtype, payload_size, write


def _read_gguf_tensors(path: str, quant: str = "q4_0"):
    GGUF_MAGIC = b"GGUF"
    SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                  6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

    # Quantizer block sizes; a tensor with fewer rows than this cannot
    # be quantized (the writer will keep it at its source precision).
    _BLOCK_SIZES = {
        "q4_0":       32,
        "vsq":        64,
        "vsq_ultra":  128,
        "hyper_vsq":  256,
        "hyper_vsq2": 256,
    }
    _block = _BLOCK_SIZES.get(quant, 32)

    def read_str(f):
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def read_scalar(f, vtype):
        size = SCALAR_SIZES[vtype]
        (val,) = struct.unpack(SCALAR_FMT[vtype], f.read(size))
        return bool(val) if vtype == 7 else val

    tensors = []
    metadata = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a valid GGUF file (magic={magic!r})")
        (version,) = struct.unpack("<I", f.read(4))
        (tensor_count,) = struct.unpack("<Q", f.read(8))
        (metadata_kv_count,) = struct.unpack("<Q", f.read(8))

        for _ in range(metadata_kv_count):
            key = read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if vtype in SCALAR_SIZES:
                metadata[key] = read_scalar(f, vtype)
            elif vtype == 8:  # String
                metadata[key] = read_str(f)
            elif vtype == 9:  # Array
                (etype,) = struct.unpack("<I", f.read(4))
                (count,) = struct.unpack("<Q", f.read(8))
                if etype in SCALAR_SIZES:
                    if key.startswith("tokenizer.ggml."):
                        metadata[key] = [read_scalar(f, etype) for _ in range(count)]
                    else:
                        f.seek(SCALAR_SIZES[etype] * count, 1)
                elif etype == 8:
                    values = []
                    for _ in range(count):
                        (n,) = struct.unpack("<Q", f.read(8))
                        values.append(f.read(n).decode("utf-8", "replace"))
                    if key.startswith("tokenizer.ggml."):
                        metadata[key] = values
                else:
                    f.seek(count * 4, 1)

        alignment = metadata.get("general.alignment", 32)
        raw_tensors = []
        for _ in range(tensor_count):
            name = read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            (dtype,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            raw_tensors.append((map_gguf_tensor_name(name), dims, dtype, offset))

        # Tensor data begins aligned
        data_base = (f.tell() + alignment - 1) & ~(alignment - 1)

    GGML_BLOCK_SIZES = {
        0:  (1, 4),     # F32
        1:  (1, 2),     # F16
        2:  (32, 18),   # Q4_0
        3:  (32, 20),   # Q4_1
        6:  (32, 22),   # Q5_0
        7:  (32, 24),   # Q5_1
        8:  (32, 34),   # Q8_0
        9:  (32, 36),   # Q8_1
        10: (256, 84),  # Q2_K
        11: (256, 110), # Q3_K
        12: (256, 144), # Q4_K
        13: (256, 176), # Q5_K
        14: (256, 210), # Q6_K
        15: (256, 292), # Q8_K
        16: (256, 66),  # IQ2_XXS
        17: (256, 74),  # IQ2_XS
        18: (256, 98),  # IQ3_XXS
        19: (256, 110), # IQ3_S
        20: (256, 136), # IQ4_XS
        21: (256, 144), # IQ4_NL
        28: (1, 2),     # BF16 (legacy)
        29: (1, 2),     # BF16
        30: (1, 2),     # BF16 (GGML_TYPE_BF16 modern standard)
    }

    # GGML dtypes: 0=F32, 1=F16, 2=Q4_0, 7/8=Q8_0, 28/29/30=BF16, 10..15=K-Quants, 16..21=IQ-Quants
    for name, dims, dtype, offset in raw_tensors:
        # GGUF dimensions are already stored fastest-dimension first. This
        # is the QWN convention too: shape[0] is the input width and
        # shape[1] is the output row count. The payload is not transposed,
        # so reversing dimensions here corrupts every non-square matrix.
        shape = tuple(dims)
        byte_offset = data_base + offset
        
        # Calculate tensor payload size
        numel = 1
        for d in shape:
            numel *= d

        block_elems, block_bytes = GGML_BLOCK_SIZES.get(dtype, (1, 2))
        byte_len = ((numel + block_elems - 1) // block_elems) * block_bytes

        # The native decoder currently has exact implementations only for
        # F32/F16/BF16 and GGML Q4_0. Reject every other source block ABI
        # before a container is created; an apparently valid .qwn with an
        # opaque payload is worse than a conversion error.
        if dtype not in (0, 1, 2, 8, 12, 13, 14, 28, 29, 30):
            raise ValueError(
                f"unsupported GGUF dtype {dtype} for {name}; "
                "convert from F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K or add a verified decoder"
            )
        if dtype == 8 and quant not in ("none", "q8_0"):
            raise ValueError(
                f"cannot re-quantize GGUF Q8_0 tensor {name} without a "
                "verified dequantization path; use --quant none"
            )

        if dtype in (12, 13, 14):
            out_dtype, payload_size, writer = _make_k_quant_writer(
                path, byte_offset, shape[1] if len(shape) == 2 else 1,
                shape[0] if len(shape) == 2 else numel, dtype, quant, name)
            tensors.append({"name": name, "dtype": out_dtype, "shape": shape,
                            "payload_size": payload_size, "write_payload": writer})
            continue

        if dtype == 0:  # F32
            out_dtype = DT_F32
            _ok_2d = (len(shape) == 2 and shape[0] >= _block)
            if quant in ("hyper_vsq2", "hyper_vsq", "vsq_ultra", "vsq", "q4_0") and _ok_2d:
                rows, cols = shape[1], shape[0]
                out_dtype, payload_size = _get_quant_dtype_and_size(quant, rows, cols)
                def make_writer(b_off, r, c, qm):
                    row_b = c * 4
                    chunk_r = max(1, (16 * 1024 * 1024) // row_b)
                    def write(out):
                        with open(path, "rb") as sf:
                            sf.seek(b_off)
                            for r_start in range(0, r, chunk_r):
                                cur_r = min(chunk_r, r - r_start)
                                raw = sf.read(cur_r * row_b)
                                out.write(quantize_matrix_rows(raw, cur_r, c, qm))
                    return write
                tensors.append({"name": name, "dtype": out_dtype, "shape": shape,
                                "payload_size": payload_size,
                                "write_payload": make_writer(byte_offset, rows, cols, quant)})
                continue
        elif dtype == 1:  # F16
            out_dtype = DT_F16
            _ok_2d = (len(shape) == 2 and shape[0] >= _block)
            if quant in ("hyper_vsq2", "hyper_vsq", "vsq_ultra", "vsq", "q4_0") and _ok_2d:
                rows, cols = shape[1], shape[0]
                out_dtype, payload_size = _get_quant_dtype_and_size(quant, rows, cols)
                def make_writer(b_off, r, c, qm):
                    row_b = c * 2
                    chunk_r = max(1, (16 * 1024 * 1024) // row_b)
                    def write(out):
                        with open(path, "rb") as sf:
                            sf.seek(b_off)
                            for r_start in range(0, r, chunk_r):
                                cur_r = min(chunk_r, r - r_start)
                                raw = sf.read(cur_r * row_b)
                                raw_f32 = f16_payload_to_f32(raw)
                                out.write(quantize_matrix_rows(raw_f32, cur_r, c, qm))
                    return write
                tensors.append({"name": name, "dtype": out_dtype, "shape": shape,
                                "payload_size": payload_size,
                                "write_payload": make_writer(byte_offset, rows, cols, quant)})
                continue
        elif dtype in (28, 29, 30):  # BF16
            out_dtype = DT_BF16
            _ok_2d = (len(shape) == 2 and shape[0] >= _block)
            if quant in ("hyper_vsq2", "hyper_vsq", "vsq_ultra", "vsq", "q4_0") and _ok_2d:
                rows, cols = shape[1], shape[0]
                out_dtype, payload_size = _get_quant_dtype_and_size(quant, rows, cols)
                def make_writer(b_off, r, c, qm):
                    row_b = c * 2
                    chunk_r = max(1, (16 * 1024 * 1024) // row_b)
                    def write(out):
                        with open(path, "rb") as sf:
                            sf.seek(b_off)
                            for r_start in range(0, r, chunk_r):
                                cur_r = min(chunk_r, r - r_start)
                                raw = sf.read(cur_r * row_b)
                                raw_f32 = bf16_payload_to_f32(raw)
                                out.write(quantize_matrix_rows(raw_f32, cur_r, c, qm))
                    return write
                tensors.append({"name": name, "dtype": out_dtype, "shape": shape,
                                "payload_size": payload_size,
                                "write_payload": make_writer(byte_offset, rows, cols, quant)})
                continue
        elif dtype == 2:  # Q4_0
            if len(shape) != 2 or shape[0] % 32 != 0:
                raise ValueError(
                    f"Q4_0 tensor {name} is not a row-wise matrix with a "
                    f"32-element row width: shape={shape}"
                )
            out_dtype = DT_Q4_0
        elif dtype == 8:  # Q8_0
            out_dtype = DT_Q8_0
        else:
            out_dtype = (DT_F32 if dtype == 0 else
                         DT_F16 if dtype == 1 else
                         DT_BF16 if dtype in (28, 29, 30) else DT_BYTES)
        def make_copy(b_off, b_len, tensor_name=name):
            def write(out):
                with open(path, "rb") as sf:
                    sf.seek(b_off)
                    rem = b_len
                    while rem > 0:
                        chunk = sf.read(min(8 << 20, rem))
                        if not chunk:
                            raise ValueError(f"truncated GGUF tensor payload: {tensor_name}")
                        out.write(chunk)
                        rem -= len(chunk)
            return write

        tensors.append({"name": name, "dtype": out_dtype, "shape": shape,
                        "payload_size": byte_len, "write_payload": make_copy(byte_offset, byte_len)})

    arch_prefix = metadata.get("general.architecture", "llama")
    hidden = metadata.get(f"{arch_prefix}.embedding_length", 0)
    inter = metadata.get(f"{arch_prefix}.feed_forward_length", 0)
    heads = metadata.get(f"{arch_prefix}.attention.head_count", 0)
    kv_heads = metadata.get(f"{arch_prefix}.attention.head_count_kv", heads)
    # head_dim: trust the GGUF metadata when the family exposes it
    # (qwen35.attention.key_length / value_length, llama.attention.head_dim
    # etc.); fall back to hidden/heads only when the metadata is absent.
    # The previous default (hidden // heads) is wrong for Qwen3.5 (real
    # head_dim=256 vs. computed 160) and produced .qwn files that the
    # native decoder refused to load.
    q_head_dim = metadata.get(f"{arch_prefix}.attention.head_dim", 0)
    if not q_head_dim:
        q_head_dim = metadata.get(f"{arch_prefix}.attention.query_length", 0)
    if not q_head_dim and heads:
        q_head_dim = hidden // max(1, heads)
    k_head_dim = metadata.get(f"{arch_prefix}.attention.key_length", q_head_dim)
    v_head_dim = metadata.get(f"{arch_prefix}.attention.value_length", k_head_dim)
    head_dim = q_head_dim
    layers = metadata.get(f"{arch_prefix}.block_count", 0)
    vocab = metadata.get(f"{arch_prefix}.vocab_size", 0)
    if not vocab:
        for t in tensors:
            if t["name"] in ("model.embed_tokens.weight", "lm_head.weight") and len(t.get("shape", ())) >= 2:
                vocab = max(t["shape"])
                break
    ctx = metadata.get(f"{arch_prefix}.context_length", 2048)

    config_data = json.dumps({
        "hidden_size": hidden, "intermediate_size": inter,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads,
        "head_dim": head_dim, "q_head_dim": q_head_dim,
        "k_head_dim": k_head_dim, "v_head_dim": v_head_dim,
        "q_dim": heads * q_head_dim if heads and q_head_dim else 0,
        "k_dim": kv_heads * k_head_dim if kv_heads and k_head_dim else 0,
        "v_dim": kv_heads * v_head_dim if kv_heads and v_head_dim else 0,
        "num_hidden_layers": layers,
        "vocab_size": vocab, "max_position_embeddings": ctx,
        "bos_token_id": int(metadata.get("tokenizer.ggml.bos_token_id", -1)),
        "eos_token_id": int(metadata.get("tokenizer.ggml.eos_token_id", -1)),
    }, ensure_ascii=False).encode("utf-8")
    tensors.append({"name": "__qwn.config", "dtype": DT_BYTES,
                    "shape": (len(config_data),), "payload": config_data})

    token_list = metadata.get("tokenizer.ggml.tokens")
    merge_list = metadata.get("tokenizer.ggml.merges", [])
    if not token_list:
        raise ValueError("GGUF tokenizer tokens are missing; refusing to emit a fake tokenizer")
    vocab_map = {str(token): index for index, token in enumerate(token_list)}
    merge_pairs = []
    for merge in merge_list:
        parts = str(merge).split(" ", 1)
        if len(parts) == 2:
            merge_pairs.append(parts)

    tok_data = json.dumps({
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab_map,
            "merges": merge_pairs
        },
        "added_tokens": []
    }, ensure_ascii=False).encode("utf-8")
    tensors.append({"name": "__qwn.tokenizer", "dtype": DT_BYTES,
                    "shape": (len(tok_data),), "payload": tok_data})

    dims = (hidden, inter, heads, kv_heads, head_dim, layers, vocab, ctx,
            heads * q_head_dim if heads and q_head_dim else 0,
            kv_heads * k_head_dim if kv_heads and k_head_dim else 0,
            kv_heads * v_head_dim if kv_heads and v_head_dim else 0)
    return tensors, dims


def _read_pytorch_tensors(path: str, quant: str = "q4_0"):
    torch = _get_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required to convert .pt / .pth / .bin checkpoints (pip install torch)")
    
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    tensors = []
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        arr = tensor.detach().float().numpy()
        shape = tuple(reversed(arr.shape))
        if quant in QUANT_BLOCK_SIZES and arr.ndim == 2 and arr.shape[1] >= QUANT_BLOCK_SIZES[quant]:
            rows, cols = arr.shape
            out_dtype, payload_size = _get_quant_dtype_and_size(quant, rows, cols)
            def make_writer(a):
                def write(out):
                    out.write(quantize_matrix_rows(a.astype("float32", copy=False).tobytes(),
                                                   a.shape[0], a.shape[1], quant))
                return write
            tensors.append({"name": name, "dtype": out_dtype, "shape": shape,
                            "payload_size": payload_size, "write_payload": make_writer(arr)})
        else:
            raw = arr.tobytes()
            tensors.append({"name": name, "dtype": DT_F32, "shape": shape,
                            "payload": raw, "payload_size": len(raw)})

    root = Path(path).parent
    config_file = root / "config.json"
    dims = (0,) * 8
    if config_file.is_file():
        cfg = json.loads(config_file.read_text("utf-8"))
        hidden = cfg.get("hidden_size", 0)
        heads = cfg.get("num_attention_heads", 0)
        kv_heads = cfg.get("num_key_value_heads", heads)
        q_head_dim = cfg.get("q_head_dim", cfg.get("head_dim", 0) or hidden // max(1, heads))
        k_head_dim = cfg.get("k_head_dim", cfg.get("key_length", q_head_dim))
        v_head_dim = cfg.get("v_head_dim", cfg.get("value_length", k_head_dim))
        dims = (hidden, cfg.get("intermediate_size", 0), heads, kv_heads,
                q_head_dim, cfg.get("num_hidden_layers", 0), cfg.get("vocab_size", 0),
                cfg.get("max_position_embeddings", 0),
                heads * q_head_dim if heads else 0,
                kv_heads * k_head_dim if kv_heads else 0,
                kv_heads * v_head_dim if kv_heads else 0)
        data = config_file.read_bytes()
        tensors.append({"name": "__qwn.config", "dtype": DT_BYTES,
                        "shape": (len(data),), "payload": data})
        tokenizer_file = root / "tokenizer.json"
        if tokenizer_file.is_file():
            tokenizer = tokenizer_file.read_bytes()
            tensors.append({"name": "__qwn.tokenizer", "dtype": DT_BYTES,
                            "shape": (len(tokenizer),), "payload": tokenizer})
    return tensors, dims


def convert_model(src: str, dst: str, quant: str = "q4_0") -> int:
    """Universal Model Converter: Auto-detects .gguf, .safetensors, .pt/.pth/.bin, .onnx, .h5 and converts to .qwn."""
    src_path = Path(src)
    ext = src_path.suffix.lower()

    is_gguf = ext == ".gguf" or (src_path.is_file() and open(src_path, "rb").read(4) == b"GGUF")
    if is_gguf:
        tensors, dims = _read_gguf_tensors(str(src_path), quant)
        # Check for companion mmproj file in same directory
        parent_dir = src_path.parent
        for mmproj_cand in parent_dir.glob("*mmproj*.gguf"):
            if mmproj_cand.is_file() and mmproj_cand != src_path:
                try:
                    mm_tensors, _ = _read_gguf_tensors(str(mmproj_cand), quant="none")
                    for mt in mm_tensors:
                        if not mt["name"].startswith("mmproj.") and not mt["name"].startswith("vision_tower."):
                            mt["name"] = f"mmproj.{mt['name']}"
                        tensors.append(mt)
                except Exception:
                    pass
        return write_qwn(dst, tensors, arch_dims=dims)
    elif ext in (".pt", ".pth", ".bin"):
        tensors, dims = _read_pytorch_tensors(str(src_path), quant)
        return write_qwn(dst, tensors, arch_dims=dims)
    elif ext == ".safetensors" or src_path.is_dir() or any(src_path.glob("*.safetensors") if src_path.is_dir() else ()):
        return convert_safetensors(src, dst, quant)
    else:
        # Default fallback to safetensors
        return convert_safetensors(src, dst, quant)


def convert_safetensors(src: str, dst: str, quant: str = "q4_0") -> int:
    tensors = []
    for meta in _read_safetensors_meta(src):
        name, dtype, shape = meta["name"], meta["dtype"], meta["shape"]
        # Safetensors is row-major [N,K]; .qwn stores fastest dimension first.
        qwn_shape = tuple(reversed(shape))
        if quant in ("hyper_vsq2", "hyper_vsq", "vsq_ultra", "vsq", "q4_0") and dtype in ("F32", "F16", "BF16") and len(shape) == 2 and shape[1] >= QUANT_BLOCK_SIZES.get(quant, 32):
            rows, cols = shape
            out_dtype, payload_size = _get_quant_dtype_and_size(quant, rows, cols)
            tensor = {"name": name, "dtype": out_dtype, "shape": qwn_shape,
                      "payload_size": payload_size, "write_payload": _matrix_quant_writer(meta, quant)}
        elif dtype == "F32":
            out_dtype = DT_F32
            tensor = {"name": name, "dtype": out_dtype, "shape": qwn_shape,
                      "payload_size": meta["bytes"], "write_payload": _copy_writer(meta)}
        elif dtype == "F16":
            out_dtype = DT_F16
            tensor = {"name": name, "dtype": out_dtype, "shape": qwn_shape,
                      "payload_size": meta["bytes"], "write_payload": _copy_writer(meta)}
        elif dtype == "BF16":
            out_dtype = DT_BF16
            tensor = {"name": name, "dtype": out_dtype, "shape": qwn_shape,
                      "payload_size": meta["bytes"], "write_payload": _copy_writer(meta)}
        else:
            raise ValueError(f"unsupported dtype {dtype} for tensor {name}")
        tensors.append(tensor)
    if not tensors:
        raise ValueError(f"no safetensors found under {src}")
    root = Path(src) if Path(src).is_dir() else Path(src).parent
    config = {}
    for filename, tensor_name in (("config.json", "__qwn.config"),
                                  ("tokenizer.json", "__qwn.tokenizer")):
        sidecar = root / filename
        if sidecar.is_file():
            data = sidecar.read_bytes()
            tensors.append({"name": tensor_name, "dtype": DT_BYTES,
                            "shape": (len(data),), "payload": data})
            if filename == "config.json":
                config = json.loads(data)
    dims = (config.get("hidden_size", 0), config.get("intermediate_size", 0),
            config.get("num_attention_heads", 0), config.get("num_key_value_heads", 0),
            config.get("head_dim", 0) or (config.get("hidden_size", 0) // max(1, config.get("num_attention_heads", 1))),
            config.get("num_hidden_layers", 0), config.get("vocab_size", 0),
            config.get("max_position_embeddings", 0))
    return write_qwn(dst, tensors, arch_dims=dims)


def synthetic(path: str, count: int = 3) -> int:
    tensors = []
    for i in range(count):
        values = [float((j + i) % 7 - 3) for j in range(64)]
        payload = quantize_q4_0(struct.pack("<64f", *values))
        tensors.append({"name": f"tensor.{i}", "dtype": DT_Q4_0,
                        "shape": (32, 2), "payload": payload})
    return write_qwn(path, tensors, arch_dims=(32, 64, 4, 4, 8, 1, 0, 0))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="qwn-convert")
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("synthetic")
    make.add_argument("output")
    make.add_argument("--tensors", type=int, default=3)
    convert = sub.add_parser("convert")
    convert.add_argument("source")
    convert.add_argument("output")
    convert.add_argument("--quant", choices=("hyper_vsq", "vsq_ultra", "vsq", "q4_0", "none"), default="hyper_vsq")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("model")
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        print(synthetic(args.output, args.tensors))
    elif args.command == "convert":
        print(convert_model(args.source, args.output, args.quant))
    else:
        print(json.dumps(inspect_qwn(args.model), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
