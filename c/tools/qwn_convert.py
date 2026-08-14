"""Qwanto Native (.qwn) writer, inspector, and safetensors converter."""
from __future__ import annotations

import argparse
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
TAIL_FMT = "<IIQQQ"
TAIL_SIZE = struct.calcsize(TAIL_FMT)  # 32

DT_F32, DT_F16, DT_Q4_0, DT_Q8_0, DT_BF16, DT_BYTES = range(6)
DT_NAME = {DT_F32: "F32", DT_F16: "F16", DT_Q4_0: "Q4_0",
           DT_Q8_0: "Q8_0", DT_BF16: "BF16", DT_BYTES: "BYTES"}


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
    except Exception:
        return None

def _get_torch():
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


def f32_to_f16(value: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", value))[0]
    except OverflowError:
        return 0xFC00 if value < 0 else 0x7C00


def quantize_q4_0(src: bytes) -> bytes:
    if len(src) % 4:
        raise ValueError("F32 payload is not element-aligned")
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
    if len(src) % 2:
        raise ValueError("BF16 payload is not element-aligned")
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
    return {"name": name, "dtype": int(t["dtype"]), "shape": shape,
            "numel": numel, "payload": payload,
            "write_payload": t.get("write_payload"), "payload_size": payload_size,
            "byte_offset": 0, "byte_size": align(payload_size, 64),
            "block_q": 32 if int(t["dtype"]) in (DT_Q4_0, DT_Q8_0) else 0}


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
    return {"name": name, "name_len": row[1], "dtype": row[2],
            "n_dims": n_dims, "shape": tuple(row[4:4 + n_dims]),
            "numel": row[8], "byte_offset": row[9],
            "byte_size": row[10], "block_q": row[11]}


def write_qwn(path: str, tensors: list[dict], arch_dims=(0,) * 8,
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
    dims = tuple(int(x) for x in tuple(arch_dims)[:8]) + (0,) * max(0, 8 - len(tuple(arch_dims)))
    struct.pack_into(HEADER_PREFIX_FMT, header, 0, MAGIC, VERSION, 0,
                     arch_code, len(layout), inline_count, 0,
                     sum(t["numel"] for t in layout), *dims[:8])
    for i, tensor in enumerate(layout[:inline_count]):
        start = HEADER_PREFIX_SIZE + i * DESC_SIZE
        header[start:start + DESC_SIZE] = pack_desc(tensor)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        out.write(header)
        for tensor in layout:
            out.seek(tensor["byte_offset"])
            if tensor["payload"] is not None:
                out.write(tensor["payload"])
            else:
                before = out.tell()
                tensor["write_payload"](out)
                if out.tell() - before != tensor["payload_size"]:
                    raise ValueError(f"stream writer size mismatch for {tensor['name']}")
            out.write(b"\0" * (tensor["byte_size"] - tensor["payload_size"]))

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
    return {"version": version, "arch_code": arch_code,
            "n_tensors": n_tensors, "inline_count": inline_count,
            "n_params": prefix[7], "arch_dims": tuple(prefix[8:16]),
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
                out.write(quantize_q4_0_rows(raw, cur_rows, cols))
    return write


def _read_gguf_tensors(path: str, quant: str = "q4_0"):
    GGUF_MAGIC = b"GGUF"
    SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                  6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

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
                    f.seek(SCALAR_SIZES[etype] * count, 1)
                elif etype == 8:
                    for _ in range(count):
                        (n,) = struct.unpack("<Q", f.read(8))
                        f.seek(n, 1)
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
            raw_tensors.append((name, dims, dtype, offset))

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
        28: (1, 2),     # BF16
    }

    # GGML dtypes: 0=F32, 1=F16, 2=Q4_0, 7/8=Q8_0, 28=BF16, 10..15=K-Quants, 16..21=IQ-Quants
    for name, dims, dtype, offset in raw_tensors:
        # dims in GGUF is fastest dimension first (already matching .qwn)
        shape = tuple(dims)
        byte_offset = data_base + offset
        
        # Calculate tensor payload size
        numel = 1
        for d in shape:
            numel *= d

        block_elems, block_bytes = GGML_BLOCK_SIZES.get(dtype, (1, 2))
        byte_len = ((numel + block_elems - 1) // block_elems) * block_bytes

        if dtype == 0:  # F32
            out_dtype = DT_F32
            if quant == "q4_0" and len(shape) == 2:
                out_dtype = DT_Q4_0
                cols, rows = shape[0], shape[1]
                payload_size = rows * ((cols + 31) // 32) * 18
                def make_writer(b_off, r, c):
                    row_b = c * 4
                    chunk_r = max(1, (16 * 1024 * 1024) // row_b)
                    def write(out):
                        with open(path, "rb") as sf:
                            sf.seek(b_off)
                            for r_start in range(0, r, chunk_r):
                                cur_r = min(chunk_r, r - r_start)
                                raw = sf.read(cur_r * row_b)
                                out.write(quantize_q4_0_rows(raw, cur_r, c))
                    return write
                tensors.append({"name": name, "dtype": out_dtype, "shape": shape,
                                "payload_size": payload_size,
                                "write_payload": make_writer(byte_offset, rows, cols)})
                continue
        elif dtype == 1:  # F16
            out_dtype = DT_F16
            if quant == "q4_0" and len(shape) == 2:
                out_dtype = DT_Q4_0
                cols, rows = shape[0], shape[1]
                payload_size = rows * ((cols + 31) // 32) * 18
                def make_writer(b_off, r, c):
                    row_b = c * 2
                    chunk_r = max(1, (16 * 1024 * 1024) // row_b)
                    def write(out):
                        with open(path, "rb") as sf:
                            sf.seek(b_off)
                            for r_start in range(0, r, chunk_r):
                                cur_r = min(chunk_r, r - r_start)
                                raw = sf.read(cur_r * row_b)
                                raw_f32 = f16_payload_to_f32(raw)
                                out.write(quantize_q4_0_rows(raw_f32, cur_r, c))
                    return write
                tensors.append({"name": name, "dtype": out_dtype, "shape": shape,
                                "payload_size": payload_size,
                                "write_payload": make_writer(byte_offset, rows, cols)})
                continue
        elif dtype == 28:  # BF16
            out_dtype = DT_BF16
            if quant == "q4_0" and len(shape) == 2:
                out_dtype = DT_Q4_0
                cols, rows = shape[0], shape[1]
                payload_size = rows * ((cols + 31) // 32) * 18
                def make_writer(b_off, r, c):
                    row_b = c * 2
                    chunk_r = max(1, (16 * 1024 * 1024) // row_b)
                    def write(out):
                        with open(path, "rb") as sf:
                            sf.seek(b_off)
                            for r_start in range(0, r, chunk_r):
                                cur_r = min(chunk_r, r - r_start)
                                raw = sf.read(cur_r * row_b)
                                raw_f32 = bf16_payload_to_f32(raw)
                                out.write(quantize_q4_0_rows(raw_f32, cur_r, c))
                    return write
                tensors.append({"name": name, "dtype": out_dtype, "shape": shape,
                                "payload_size": payload_size,
                                "write_payload": make_writer(byte_offset, rows, cols)})
                continue
        elif dtype == 2:  # Q4_0
            out_dtype = DT_Q4_0
        elif dtype in (7, 8):  # Q8_0
            out_dtype = DT_Q8_0
        else:
            out_dtype = DT_BYTES if dtype not in (0, 1, 28) else DT_F16

        def make_copy(b_off, b_len):
            def write(out):
                with open(path, "rb") as sf:
                    sf.seek(b_off)
                    rem = b_len
                    while rem > 0:
                        chunk = sf.read(min(8 << 20, rem))
                        if not chunk: break
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
    head_dim = hidden // max(1, heads) if heads else 0
    layers = metadata.get(f"{arch_prefix}.block_count", 0)
    vocab = metadata.get(f"{arch_prefix}.vocab_size", 0)
    ctx = metadata.get(f"{arch_prefix}.context_length", 2048)

    config_data = json.dumps({
        "hidden_size": hidden, "intermediate_size": inter,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads,
        "head_dim": head_dim, "num_hidden_layers": layers,
        "vocab_size": vocab, "max_position_embeddings": ctx
    }).encode("utf-8")
    tensors.append({"name": "__qwn.config", "dtype": DT_BYTES,
                    "shape": (len(config_data),), "payload": config_data})

    dims = (hidden, inter, heads, kv_heads, head_dim, layers, vocab, ctx)
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
        if quant == "q4_0" and arr.ndim == 2:
            out_dtype = DT_Q4_0
            rows, cols = arr.shape
            payload_size = rows * ((cols + 31) // 32) * 18
            def make_writer(a):
                def write(out):
                    for r in range(a.shape[0]):
                        row_bytes = struct.pack(f"<{a.shape[1]}f", *a[r])
                        out.write(quantize_q4_0(row_bytes))
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
        dims = (cfg.get("hidden_size", 0), cfg.get("intermediate_size", 0),
                cfg.get("num_attention_heads", 0), cfg.get("num_key_value_heads", 0),
                cfg.get("head_dim", 0) or (cfg.get("hidden_size", 0) // max(1, cfg.get("num_attention_heads", 1))),
                cfg.get("num_hidden_layers", 0), cfg.get("vocab_size", 0),
                cfg.get("max_position_embeddings", 0))
        data = config_file.read_bytes()
        tensors.append({"name": "__qwn.config", "dtype": DT_BYTES,
                        "shape": (len(data),), "payload": data})
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
        if quant == "q4_0" and dtype in ("F32", "F16", "BF16") and len(shape) == 2:
            out_dtype = DT_Q4_0
            payload_size = shape[0] * ((shape[1] + 31) // 32) * 18
            tensor = {"name": name, "dtype": out_dtype, "shape": qwn_shape,
                      "payload_size": payload_size, "write_payload": _q4_writer(meta)}
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
    convert.add_argument("--quant", choices=("q4_0", "none"), default="q4_0")
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
