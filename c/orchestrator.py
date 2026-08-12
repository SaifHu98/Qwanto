#!/usr/bin/env python3
"""Qwanto Resource Orchestrator — built from scratch, dependency-free.

Profiles the machine (GPU VRAM, RAM, CPU, disk), reads the GGUF model
structure directly from the file header, and computes the optimal split of
the model across GPU + RAM + CPU + NVMe so llama.cpp's optimized kernels are
driven at maximum useful speed:

  - how many layers go to VRAM (-ngl), the rest stay in RAM/mmap (NVMe)
  - multi-GPU proportional tensor split (-ts)
  - batch/ubatch sizing depending on full vs partial offload
  - KV-cache memory accounting (with quantization factors)
  - thread count from physical cores and the user's CPU limit

The hot path stays inside llama.cpp's hand-tuned CUDA/Vulkan/SIMD kernels;
this module is the control plane that decides *where* every byte lives.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys

# ---------------------------------------------------------------- GGUF parsing

_GGUF_MAGIC = b"GGUF"

# value-type -> fixed byte size (None = variable)
_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_T_STRING, _T_ARRAY = 8, 9

_SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _read_scalar(f, vtype):
    size = _SCALAR_SIZES[vtype]
    (val,) = struct.unpack(_SCALAR_FMT[vtype], f.read(size))
    return bool(val) if vtype == 7 else val


def _skip_value(f, vtype):
    if vtype in _SCALAR_SIZES:
        f.seek(_SCALAR_SIZES[vtype], 1)
    elif vtype == _T_STRING:
        (n,) = struct.unpack("<Q", f.read(8))
        f.seek(n, 1)
    elif vtype == _T_ARRAY:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if etype in _SCALAR_SIZES:
            f.seek(_SCALAR_SIZES[etype] * count, 1)
        elif etype == _T_STRING:
            for _ in range(count):
                (n,) = struct.unpack("<Q", f.read(8))
                f.seek(n, 1)
        else:
            raise ValueError(f"nested arrays unsupported (etype={etype})")
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def parse_gguf_meta(path: str) -> dict:
    """Read architecture metadata from a GGUF file header (fast, no tensors).

    Returns: {arch, n_layers, n_embd, n_head, n_head_kv, ctx_train, file_bytes}
    Raises ValueError on non-GGUF files.
    """
    file_bytes = os.path.getsize(path)
    wanted_suffixes = ("block_count", "embedding_length",
                       "attention.head_count", "attention.head_count_kv",
                       "context_length")
    out = {"arch": "", "file_bytes": file_bytes}
    found = {}
    with open(path, "rb") as f:
        if f.read(4) != _GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError(f"GGUF v{version} unsupported")
        f.seek(8, 1)  # tensor_count
        (n_kv,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            interesting = (key == "general.architecture" or
                           any(key.endswith("." + s) for s in wanted_suffixes))
            if interesting and vtype != _T_ARRAY:
                val = (_read_str(f) if vtype == _T_STRING
                       else _read_scalar(f, vtype))
                if key == "general.architecture":
                    out["arch"] = str(val)
                else:
                    found[key.rsplit(".", 1)[-1] if not key.endswith("head_count_kv")
                          else "head_count_kv"] = val
                    if key.endswith("attention.head_count"):
                        found["head_count"] = val
            else:
                _skip_value(f, vtype)
            # Early exit once we have everything (tokenizer arrays come later)
            if out["arch"] and all(
                    k in found for k in ("block_count", "embedding_length",
                                         "head_count", "head_count_kv")):
                break
    out["n_layers"] = int(found.get("block_count", 0)) or None
    out["n_embd"] = int(found.get("embedding_length", 0)) or None
    out["n_head"] = int(found.get("head_count", 0)) or None
    out["n_head_kv"] = int(found.get("head_count_kv", found.get("head_count", 0))) or None
    out["ctx_train"] = int(found.get("context_length", 0)) or None
    if not out["n_layers"]:
        raise ValueError("GGUF metadata missing block_count")
    return out


# ------------------------------------------------------------ hardware profile

def _ram_bytes() -> tuple[int, int]:
    """(total, available) RAM in bytes."""
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return int(st.ullTotalPhys), int(st.ullAvailPhys)
    try:
        with open("/proc/meminfo") as fh:
            info = fh.read()
        import re
        total = int(re.search(r"MemTotal:\s+(\d+)", info).group(1)) * 1024
        avail = int(re.search(r"MemAvailable:\s+(\d+)", info).group(1)) * 1024
        return total, avail
    except Exception:
        return 0, 0


def _gpus() -> list[dict]:
    """[{name, total_bytes, free_bytes}] — NVIDIA via nvidia-smi; else empty."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"name": parts[0],
                             "total_bytes": int(float(parts[1])) * 1024 * 1024,
                             "free_bytes": int(float(parts[2])) * 1024 * 1024})
        return gpus
    except Exception:
        return []


def _physical_cores() -> int:
    try:
        from resource_plan import physical_cpu_count
        return physical_cpu_count()
    except Exception:
        n = os.cpu_count() or 4
        return max(1, n // 2)


def profile_hardware() -> dict:
    total, avail = _ram_bytes()
    return {"ram_total": total, "ram_available": avail,
            "gpus": _gpus(), "physical_cores": _physical_cores()}


# ---------------------------------------------------------------- the planner

# bytes per element for KV cache types (per-element average incl. block scales)
_KV_BPE = {"f16": 2.0, "q8_0": 1.0625, "q5_1": 0.75, "q5_0": 0.6875,
           "q4_1": 0.625, "q4_0": 0.5625}

_VRAM_SAFETY = 0.92          # usable fraction of reported free VRAM
_COMPUTE_OVERHEAD = 900 << 20  # graph/compute buffers reserve per GPU (bytes)


def kv_cache_bytes(meta: dict, ctx: int, kv_quant: str = "f16") -> int:
    """Total KV cache size for `ctx` tokens across all layers."""
    n_layers = meta["n_layers"]
    n_embd = meta.get("n_embd") or 2048
    n_head = meta.get("n_head") or 16
    n_head_kv = meta.get("n_head_kv") or n_head
    head_dim = n_embd // max(1, n_head)
    bpe = _KV_BPE.get(kv_quant, 2.0)
    return int(2 * ctx * n_head_kv * head_dim * n_layers * bpe)


def plan(model_path: str, ctx_size: int = 16384, kv_cache_quant: str = "q4_0",
         cpu_limit: int = 100, hw: dict | None = None,
         meta: dict | None = None) -> dict:
    """Compute the optimal GPU/RAM/CPU/NVMe split for a GGUF model.

    hw/meta can be injected for testing; otherwise measured live.
    """
    meta = meta or parse_gguf_meta(model_path)
    hw = hw or profile_hardware()
    notes: list[str] = []

    n_layers = meta["n_layers"]
    model_bytes = meta["file_bytes"]
    # +1 accounts for token embeddings / output head kept alongside layers
    layer_bytes = model_bytes / (n_layers + 1)
    kv_total = kv_cache_bytes(meta, ctx_size, kv_cache_quant)
    kv_per_layer = kv_total / n_layers

    gpus = hw.get("gpus") or []
    vram_free = sum(g["free_bytes"] for g in gpus)

    if not gpus:
        # Unknown GPU (AMD/Intel via Vulkan or none): let llama.cpp decide.
        ngl = 999
        notes.append("GPU VRAM unknown (no nvidia-smi) — delegating layer "
                     "placement to llama.cpp (-ngl 999)")
    else:
        usable = vram_free * _VRAM_SAFETY - _COMPUTE_OVERHEAD * len(gpus)
        per_layer_cost = layer_bytes + kv_per_layer
        fit = int(usable / per_layer_cost) if per_layer_cost > 0 else 0
        if fit >= n_layers + 1:
            ngl = 999
            notes.append(f"model + KV fit fully in VRAM "
                         f"({model_bytes / 1e9:.1f} GB model, "
                         f"{kv_total / 1e9:.2f} GB KV) — full GPU offload")
        else:
            ngl = max(0, fit)
            notes.append(f"partial offload: {ngl}/{n_layers} layers to GPU, "
                         f"rest served from RAM/NVMe via mmap")

    # multi-GPU proportional split by free VRAM
    tensor_split = None
    if len(gpus) > 1 and vram_free > 0:
        ratios = [g["free_bytes"] / vram_free for g in gpus]
        tensor_split = ",".join(f"{r:.2f}" for r in ratios)
        notes.append(f"tensor split across {len(gpus)} GPUs: {tensor_split}")

    full_offload = ngl == 999
    batch, ubatch = (2048, 512) if full_offload else (512, 512)

    threads = max(1, int(hw.get("physical_cores", 4) * max(1, min(100, cpu_limit)) / 100))

    # RAM sanity: if the CPU-resident part exceeds available RAM, NVMe mmap
    # streaming kicks in — inform, don't block.
    if not full_offload and gpus:
        cpu_part = model_bytes - ngl * layer_bytes
        if cpu_part > hw.get("ram_available", 0) > 0:
            notes.append("CPU-resident weights exceed free RAM — NVMe mmap "
                         "streaming will be used (slower, but runs)")

    return {"ngl": ngl, "tensor_split": tensor_split, "batch": batch,
            "ubatch": ubatch, "threads": threads, "n_layers": n_layers,
            "model_bytes": model_bytes, "kv_cache_bytes": kv_total,
            "vram_free_bytes": vram_free, "full_offload": full_offload,
            "notes": notes}


def describe(p: dict) -> str:
    gb = 1 << 30
    lines = [f"model {p['model_bytes'] / gb:.2f} GB · {p['n_layers']} layers · "
             f"KV {p['kv_cache_bytes'] / gb:.2f} GB",
             f"plan: ngl={p['ngl']} batch={p['batch']} ubatch={p['ubatch']} "
             f"threads={p['threads']}"
             + (f" tensor-split={p['tensor_split']}" if p["tensor_split"] else "")]
    lines += p["notes"]
    return "\n".join(lines)
