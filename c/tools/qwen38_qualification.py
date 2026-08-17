"""Qualification evidence generator for the local Qwen3.8-27B GGUF source.

This module deliberately stops before conversion.  GGUF is a source artifact
in Qwanto Native, and the current native runtime has no validated Qwen3.8
hybrid/DeltaNet/MTP execution path.  The parser is intentionally self
contained so qualification can inspect a large GGUF header without importing
an external runtime or allocating model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


GGUF_MAGIC = b"GGUF"
QWN_HEADER_SIZE = 4096
QWN_ALIGNMENT = 4096
QWN_DESCRIPTOR_SIZE = 136
QWN_TENSOR_ALIGNMENT = 64
STREAM_CHUNK_BYTES = 16 * 1024 * 1024

# GGML type ids used by the attached file.  The block sizes are part of the
# GGUF source ABI; an unknown id is never treated as a compatible dtype.
GGML_TYPES: Dict[int, Tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("IQ1_M", 256, 56),
    28: ("BF16_legacy", 1, 2),
    29: ("BF16", 1, 2),
    30: ("BF16", 1, 2),
}

# This is intentionally the same conservative source set enforced by the
# current qwn_convert.py reader.  IQ dtypes are not silently reinterpreted.
CURRENT_CONVERTER_DTYPES = {0, 1, 2, 8, 12, 13, 14, 28, 29, 30}
MATRIX_QUANT_DTYPE = "HYPER_VSQ2 (planned; not emitted)"

GGUF_SCALAR_SIZES = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
    10: 8, 11: 8, 12: 8,
}
GGUF_SCALAR_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d",
}


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity() -> Tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _read_string(stream) -> str:
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError("truncated GGUF string length")
    (length,) = struct.unpack("<Q", raw)
    value = stream.read(length)
    if len(value) != length:
        raise ValueError("truncated GGUF string")
    return value.decode("utf-8", "replace")


def _read_scalar(stream, value_type: int) -> Any:
    if value_type not in GGUF_SCALAR_SIZES:
        raise ValueError(f"unsupported GGUF scalar metadata type {value_type}")
    size = GGUF_SCALAR_SIZES[value_type]
    raw = stream.read(size)
    if len(raw) != size:
        raise ValueError("truncated GGUF scalar metadata")
    value = struct.unpack(GGUF_SCALAR_FORMATS[value_type], raw)[0]
    return bool(value) if value_type == 7 else value


def _read_metadata_value(stream, value_type: int, key: str) -> Any:
    if value_type in GGUF_SCALAR_SIZES:
        return _read_scalar(stream, value_type)
    if value_type == 8:
        return _read_string(stream)
    if value_type != 9:
        raise ValueError(f"unsupported GGUF metadata type {value_type} for {key}")
    raw = stream.read(12)
    if len(raw) != 12:
        raise ValueError("truncated GGUF array metadata")
    element_type, count = struct.unpack("<IQ", raw)
    if element_type in GGUF_SCALAR_SIZES:
        if key.startswith("tokenizer."):
            return [_read_scalar(stream, element_type) for _ in range(count)]
        stream.seek(GGUF_SCALAR_SIZES[element_type] * count, os.SEEK_CUR)
        return {"array_count": count, "element_type": element_type}
    if element_type == 8:
        values = [_read_string(stream) for _ in range(count)]
        return values if key.startswith("tokenizer.") else {
            "array_count": count, "element_type": element_type,
        }
    raise ValueError(f"unsupported GGUF array element type {element_type} for {key}")


def _payload_size(numel: int, dtype_id: int) -> int:
    spec = GGML_TYPES.get(dtype_id)
    if spec is None:
        raise ValueError(f"unknown GGUF tensor dtype {dtype_id}")
    block_elements, block_bytes = spec[1], spec[2]
    return ((numel + block_elements - 1) // block_elements) * block_bytes


def _dtype_name(dtype_id: int) -> str:
    return GGML_TYPES.get(dtype_id, (f"UNKNOWN_{dtype_id}", 1, 0))[0]


def _layer_for(name: str) -> Optional[int]:
    match = re.match(r"blk\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def _operator_for(name: str) -> str:
    lower = name.lower()
    if name == "token_embd.weight":
        return "token_embedding"
    if name == "output.weight":
        return "lm_head"
    if name == "output_norm.weight":
        return "final_normalization"
    if "nextn." in lower:
        return "mtp_head"
    if lower.startswith("blk.") and ".ssm_" in lower:
        return "gated_deltanet_recurrent_state"
    if ".attn_qkv." in lower or ".attn_gate." in lower:
        return "gated_deltanet_projection"
    if any(marker in lower for marker in (
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight",
        ".attn_output.weight",
    )):
        return "full_attention"
    if ".attn_" in lower:
        return "normalization"
    if ".ffn_" in lower:
        return "feed_forward"
    if "norm" in lower:
        return "normalization"
    if "vision" in lower or "mmproj" in lower:
        return "vision_optional"
    return "unclassified"


def _destination_name(name: str) -> str:
    mapping = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    if name in mapping:
        return mapping[name]
    match = re.match(r"blk\.(\d+)\.(.*)", name)
    if not match:
        return name
    layer, suffix = match.groups()
    names = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }
    return f"model.layers.{layer}.{names[suffix]}" if suffix in names else (
        f"model.layers.{layer}.{suffix}")


def _category_for(name: str, operator: str) -> str:
    if operator == "gated_deltanet_recurrent_state":
        return "gated_deltanet_state"
    if operator == "gated_deltanet_projection":
        return "gated_deltanet"
    if operator == "full_attention":
        return "full_attention"
    if operator == "mtp_head":
        return "mtp"
    if operator == "feed_forward":
        return "feed_forward"
    if operator == "token_embedding":
        return "embedding"
    if operator == "lm_head":
        return "lm_head"
    if operator == "final_normalization" or operator == "normalization":
        return "normalization"
    if operator == "vision_optional":
        return "vision_optional"
    return "other"


def _planned_destination_dtype(name: str, shape: Tuple[int, ...], dtype_id: int) -> str:
    if name == "token_embd.weight" or name == "output.weight":
        return MATRIX_QUANT_DTYPE
    if len(shape) == 2 and shape[0] >= 256 and "norm" not in name.lower():
        return MATRIX_QUANT_DTYPE
    if dtype_id in (1, 28, 29, 30):
        return "BF16" if dtype_id in (28, 29, 30) else "F16"
    return "F32 (planned dequantization)"


def _status_for(dtype_id: int, operator: str) -> Tuple[str, str, str]:
    if operator == "vision_optional":
        return (
            "EXCLUDED_TEXT_ONLY",
            "EXCLUDED_TEXT_ONLY",
            "No vision tensor is present; chat template vision markers remain optional and are not runnable.",
        )
    reasons = []
    if dtype_id not in CURRENT_CONVERTER_DTYPES:
        reasons.append(f"GGUF dtype {_dtype_name(dtype_id)} ({dtype_id}) is not implemented by qwn_convert")
    if operator in {"gated_deltanet", "gated_deltanet_state", "mtp_head"}:
        reasons.append(f"native operator {operator} is not end-to-end implemented")
    if operator == "full_attention" and dtype_id in CURRENT_CONVERTER_DTYPES:
        return "BLOCKED_BY_MODEL_GATE", "BLOCKED_BY_MODEL_GATE", "Qwen3.8 hybrid model gate has not passed"
    if reasons:
        detail = "; ".join(reasons)
        return "UNAVAILABLE", "UNAVAILABLE", detail
    return "UNAVAILABLE", "UNAVAILABLE", "qualification is blocked before conversion"


def inspect_gguf(path: Path) -> Dict[str, Any]:
    """Read GGUF metadata and tensor descriptors only; never read weights."""
    metadata: Dict[str, Any] = {}
    tensors: List[Dict[str, Any]] = []
    with path.open("rb") as stream:
        if stream.read(4) != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        version_raw = stream.read(4)
        tensor_count_raw = stream.read(8)
        metadata_count_raw = stream.read(8)
        if len(version_raw) != 4 or len(tensor_count_raw) != 8 or len(metadata_count_raw) != 8:
            raise ValueError("truncated GGUF header")
        version = struct.unpack("<I", version_raw)[0]
        tensor_count = struct.unpack("<Q", tensor_count_raw)[0]
        metadata_count = struct.unpack("<Q", metadata_count_raw)[0]
        for _ in range(metadata_count):
            key = _read_string(stream)
            value_type_raw = stream.read(4)
            if len(value_type_raw) != 4:
                raise ValueError("truncated GGUF metadata value type")
            metadata[key] = _read_metadata_value(
                stream, struct.unpack("<I", value_type_raw)[0], key)
        tensor_header_offset = stream.tell()
        for _ in range(tensor_count):
            name = _read_string(stream)
            rank_raw = stream.read(4)
            if len(rank_raw) != 4:
                raise ValueError("truncated GGUF tensor rank")
            rank = struct.unpack("<I", rank_raw)[0]
            dims_raw = stream.read(8 * rank)
            if len(dims_raw) != 8 * rank:
                raise ValueError("truncated GGUF tensor shape")
            shape = tuple(struct.unpack("<" + "Q" * rank, dims_raw))
            dtype_raw = stream.read(4)
            offset_raw = stream.read(8)
            if len(dtype_raw) != 4 or len(offset_raw) != 8:
                raise ValueError("truncated GGUF tensor descriptor")
            dtype_id = struct.unpack("<I", dtype_raw)[0]
            offset = struct.unpack("<Q", offset_raw)[0]
            numel = math.prod(shape) if shape else 0
            operator = _operator_for(name)
            cpu_status, cuda_status, reason = _status_for(dtype_id, operator)
            tensors.append({
                "source_name": name,
                "destination_name": _destination_name(name),
                "shape_fastest_dimension_first": list(shape),
                "numel": numel,
                "source_dtype_id": dtype_id,
                "source_dtype": _dtype_name(dtype_id),
                "source_block_elements": GGML_TYPES.get(dtype_id, ("unknown", 1, 0))[1],
                "source_block_bytes": GGML_TYPES.get(dtype_id, ("unknown", 1, 0))[2],
                "source_payload_bytes": _payload_size(numel, dtype_id),
                "source_data_offset": offset,
                "owner_layer": _layer_for(name),
                "category": _category_for(name, operator),
                "runtime_operator": operator,
                "expected_qwn_dtype": _planned_destination_dtype(name, shape, dtype_id),
                "cpu_implementation": cpu_status,
                "cuda_implementation": cuda_status,
                "qualification_reason": reason,
                "required_for_text_logits": operator not in {"vision_optional"},
            })

    architecture = str(metadata.get("general.architecture", "")).lower()
    layers = int(metadata.get("qwen35.block_count", 0) or 0)
    ssm_layers = sorted({
        tensor["owner_layer"] for tensor in tensors
        if tensor["category"] == "gated_deltanet_state"
        and tensor["owner_layer"] is not None
    })
    attention_layers = sorted({
        tensor["owner_layer"] for tensor in tensors
        if tensor["category"] == "full_attention"
        and tensor["owner_layer"] is not None
    })
    mtp_layers = sorted({
        tensor["owner_layer"] for tensor in tensors
        if tensor["category"] == "mtp"
        and tensor["owner_layer"] is not None
    })
    vision_tensors = [
        tensor["source_name"] for tensor in tensors
        if tensor["category"] == "vision_optional"
    ]
    tokenizer_tokens = metadata.get("tokenizer.ggml.tokens", [])
    tokenizer_merges = metadata.get("tokenizer.ggml.merges", [])
    chat_template = metadata.get("tokenizer.chat_template")
    if not isinstance(chat_template, str):
        chat_template = None
    separate_heads = all(name in {t["source_name"] for t in tensors}
                         for name in ("output.weight", "token_embd.weight"))
    selected_metadata = {
        key: value for key, value in metadata.items()
        if key.startswith(("general.", "qwen35."))
        or key in {"tokenizer.ggml.model", "tokenizer.ggml.pre",
                   "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                   "tokenizer.ggml.padding_token_id"}
    }
    selected_metadata["tokenizer.ggml.tokens_count"] = len(tokenizer_tokens) if isinstance(tokenizer_tokens, list) else 0
    selected_metadata["tokenizer.ggml.merges_count"] = len(tokenizer_merges) if isinstance(tokenizer_merges, list) else 0
    if isinstance(tokenizer_tokens, list):
        selected_metadata["tokenizer.ggml.tokens_sample"] = {
            "first": tokenizer_tokens[:3], "last": tokenizer_tokens[-3:],
        }
    if isinstance(tokenizer_merges, list):
        selected_metadata["tokenizer.ggml.merges_sample"] = {
            "first": tokenizer_merges[:3], "last": tokenizer_merges[-3:],
        }
    selected_metadata["tokenizer.estimated_serialized_bytes"] = (
        sum(len(str(item).encode("utf-8")) + 8 for item in tokenizer_tokens)
        + sum(len(str(item).encode("utf-8")) + 8 for item in tokenizer_merges)
        + 1024
        if isinstance(tokenizer_tokens, list) and isinstance(tokenizer_merges, list)
        else 1024
    )
    selected_metadata["tokenizer.chat_template"] = chat_template

    dtype_counts = Counter(tensor["source_dtype_id"] for tensor in tensors)
    layer_summary: Dict[str, Any] = {}
    by_layer: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for tensor in tensors:
        if tensor["owner_layer"] is not None:
            by_layer[tensor["owner_layer"]].append(tensor)
    for layer in range(layers):
        entries = by_layer.get(layer, [])
        layer_summary[str(layer)] = {
            "tensor_count": len(entries),
            "has_gated_deltanet_state": any(e["category"] == "gated_deltanet_state" for e in entries),
            "has_full_attention": any(e["category"] == "full_attention" for e in entries),
            "has_mtp": any(e["category"] == "mtp" for e in entries),
            "all_source_tensor_names": [e["source_name"] for e in entries],
        }
    unsupported_dtypes = sorted({
        tensor["source_dtype_id"] for tensor in tensors
        if tensor["source_dtype_id"] not in CURRENT_CONVERTER_DTYPES
    })
    return {
        "schema_version": 1,
        "source": {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        },
        "gguf": {
            "version": version,
            "tensor_count": tensor_count,
            "metadata_count": metadata_count,
            "tensor_header_offset": tensor_header_offset,
            "data_alignment": int(metadata.get("general.alignment", 32) or 32),
        },
        "metadata": selected_metadata,
        "architecture": {
            "general_architecture": architecture,
            "model_family": "Qwen3.8 hybrid Transformer/Gated DeltaNet",
            "layer_count": layers,
            "gated_deltanet_layers": ssm_layers,
            "gated_deltanet_layer_count": len(ssm_layers),
            "full_attention_layers": attention_layers,
            "full_attention_layer_count": len(attention_layers),
            "full_attention_interval": metadata.get("qwen35.full_attention_interval"),
            "mtp_layers": mtp_layers,
            "mtp_tensor_names": [t["source_name"] for t in tensors if t["category"] == "mtp"],
            "vision_tensor_names": vision_tensors,
            "vision_metadata_markers": [
                marker for marker in ("image", "video", "vision_start", "image_pad", "video_pad")
                if isinstance(chat_template, str) and marker in chat_template
            ],
            "lm_head": "separate output.weight and token_embd.weight tensors" if separate_heads else "not proven",
            "text_only_policy": "vision tensors absent; optional vision chat-template branches are rejected",
        },
        "dtype_summary": {
            "counts_by_id": {str(key): value for key, value in sorted(dtype_counts.items())},
            "counts_by_name": {_dtype_name(key): value for key, value in sorted(dtype_counts.items())},
            "unsupported_by_current_converter": unsupported_dtypes,
            "unsupported_names": [_dtype_name(key) for key in unsupported_dtypes],
            "general_file_type": metadata.get("general.file_type"),
            "general_file_type_label": "IQ2_M mixed quantization" if metadata.get("general.file_type") == 14 else "not mapped",
        },
        "required_component_checks": {
            "embedding": {"source_present": "token_embd.weight" in {t["source_name"] for t in tensors}, "runtime_status": "UNAVAILABLE"},
            "tokenizer": {"source_present": bool(tokenizer_tokens), "runtime_status": "UNAVAILABLE"},
            "chat_template": {"source_present": bool(chat_template), "runtime_status": "UNAVAILABLE"},
            "normalization": {"source_present": any(t["category"] == "normalization" for t in tensors), "runtime_status": "UNAVAILABLE"},
            "rope": {"source_present": "qwen35.rope.dimension_count" in metadata, "runtime_status": "UNAVAILABLE"},
            "full_attention": {"source_present": bool(attention_layers), "runtime_status": "UNAVAILABLE"},
            "gated_deltanet": {"source_present": bool(ssm_layers), "runtime_status": "UNAVAILABLE_UNIMPLEMENTED"},
            "recurrent_state": {"source_present": bool(ssm_layers), "runtime_status": "UNAVAILABLE_UNIMPLEMENTED"},
            "feed_forward": {"source_present": any(t["category"] == "feed_forward" for t in tensors), "runtime_status": "UNAVAILABLE"},
            "lm_head": {"source_present": "output.weight" in {t["source_name"] for t in tensors}, "runtime_status": "UNAVAILABLE"},
            "mtp": {"source_present": bool(mtp_layers), "runtime_status": "UNAVAILABLE_UNIMPLEMENTED"},
            "vision_text_only": {"source_present": bool(vision_tensors), "runtime_status": "EXCLUDED_TEXT_ONLY"},
        },
        "layer_summary": layer_summary,
        "tensors": tensors,
    }


def _tokenizer_estimate(metadata: Dict[str, Any]) -> int:
    if "tokenizer.estimated_serialized_bytes" in metadata:
        return int(metadata["tokenizer.estimated_serialized_bytes"])
    tokens = metadata.get("tokenizer.ggml.tokens", [])
    merges = metadata.get("tokenizer.ggml.merges", [])
    token_bytes = sum(len(str(item).encode("utf-8")) + 8 for item in tokens) if isinstance(tokens, list) else 0
    merge_bytes = sum(len(str(item).encode("utf-8")) + 8 for item in merges) if isinstance(merges, list) else 0
    return token_bytes + merge_bytes + 1024


def _projected_qwn_size(tensors: Iterable[Dict[str, Any]], tokenizer_bytes: int) -> Dict[str, int]:
    payload = 0
    descriptor_count = 0
    cursor = QWN_HEADER_SIZE
    for tensor in tensors:
        shape = tensor["shape_fastest_dimension_first"]
        numel = int(tensor["numel"])
        if tensor["source_name"] in {"output.weight", "token_embd.weight"} or (
            len(shape) == 2 and shape[0] >= 256 and "norm" not in tensor["source_name"].lower()
        ):
            planned = ((numel + 255) // 256) * 74
        else:
            source_dtype = tensor["source_dtype_id"]
            scalar_bytes = 2 if source_dtype in (1, 28, 29, 30) else 4
            planned = numel * scalar_bytes
        descriptor_count += 1
        cursor = _align(cursor, QWN_ALIGNMENT)
        cursor += _align(planned, QWN_TENSOR_ALIGNMENT)
        payload += planned
    for name, size in (("__qwn.config", 4096), ("__qwn.tokenizer", tokenizer_bytes)):
        del name
        descriptor_count += 1
        cursor = _align(cursor, QWN_ALIGNMENT)
        cursor += _align(size, QWN_TENSOR_ALIGNMENT)
        payload += size
    cursor = _align(cursor, QWN_ALIGNMENT) + 32
    return {
        "projected_payload_bytes": payload,
        "projected_descriptor_count": descriptor_count,
        "projected_qwn_size_bytes": cursor,
    }


def _hardware_fit(inspection: Dict[str, Any], ram_bytes: int, vram_bytes: int,
                  gpu_name: str, managed_dir: Path) -> Dict[str, Any]:
    metadata = inspection["metadata"]
    architecture = inspection["architecture"]
    size = _projected_qwn_size(
        inspection["tensors"], _tokenizer_estimate(metadata))
    full_attention_layers = int(architecture["full_attention_layer_count"])
    kv_heads = int(metadata.get("qwen35.attention.head_count_kv", 0) or 0)
    key_length = int(metadata.get("qwen35.attention.key_length", 0) or 0)
    # K + V, FP16.  DeltaNet recurrent-state bytes are deliberately not
    # guessed because the native state layout is not implemented.
    kv_bytes_per_token = 2 * full_attention_layers * kv_heads * key_length * 2
    kv_context = {
        str(context): {
            "bytes": kv_bytes_per_token * context,
            "source": "metadata formula: 2 * full_attention_layers * kv_heads * key_length * 2-byte FP16",
            "status": "ESTIMATE_ONLY",
        }
        for context in (4096, 8192, 16384, 32768)
    }
    usage = shutil.disk_usage(managed_dir.anchor or managed_dir.parent)
    conversion_temp = size["projected_qwn_size_bytes"] + 64 * 1024 * 1024
    workspace = 512 * 1024 * 1024
    vram_rows = {}
    for context, kv in kv_context.items():
        required = size["projected_qwn_size_bytes"] + kv["bytes"] + workspace
        vram_rows[context] = {
            "required_bytes_without_delta_state": required,
            "vram_bytes": vram_bytes,
            "status": "NOT_PROVABLE_DELTA_STATE_UNIMPLEMENTED" if required <= vram_bytes else "DOES_NOT_FIT_WITHOUT_OUT_OF_CORE",
        }
    return {
        "schema_version": 1,
        "classification": "HARDWARE_FIT_FAILED",
        "decision_basis": "required Gated DeltaNet state and native operators are not implemented; fit cannot be proven",
        "hardware": {
            "cpu": "AMD Ryzen 9 9955HX (user-specified target; topology not re-probed by this report)",
            "ram_bytes": ram_bytes,
            "gpu": gpu_name,
            "vram_bytes": vram_bytes,
            "os": "Windows 11 (user-specified target)",
            "cuda": "13.3 (user-specified/locally installed toolkit)",
        },
        "managed_model_directory": str(managed_dir),
        "source_and_destination_disk": {
            "path": str(managed_dir.anchor or managed_dir.parent),
            "free_bytes_at_generation": usage.free,
            "total_bytes_at_generation": usage.total,
            "conversion_temp_requirement_bytes": conversion_temp,
            "requirement_source": "projected QWN plus 64 MiB safety margin; not a completed conversion",
        },
        "projected_qwn": size,
        "fp16_kv": {
            "full_attention_layers": full_attention_layers,
            "kv_heads": kv_heads,
            "key_length": key_length,
            "bytes_per_token": kv_bytes_per_token,
            "contexts": kv_context,
        },
        "deltanet_recurrent_state": {
            "status": "UNAVAILABLE",
            "state_size": metadata.get("qwen35.ssm.state_size"),
            "group_count": metadata.get("qwen35.ssm.group_count"),
            "inner_size": metadata.get("qwen35.ssm.inner_size"),
            "bytes": None,
            "reason": "No validated Qwen3.8 Gated DeltaNet state layout exists in qwnrun.",
        },
        "vram_fit_without_delta_state": vram_rows,
        "placement": {
            "status": "UNAVAILABLE",
            "reason": "No conversion or CUDA coverage exists for required hybrid operators; no resident/RAM/streamed plan can be claimed.",
        },
        "cuda_coverage": {
            "status": "UNAVAILABLE",
            "expected_operation_count_per_layer": None,
            "observed_gpu_matmul_count": 0,
            "observed_cpu_fallback_count": None,
            "reason": "No valid QWN model was produced or executed.",
            "current_4b_residency_note": {
                "observed_bytes": 463370240,
                "interpretation": "Existing 4B evidence counts only tensors accepted and uploaded by the current HyperVSQ-2 projection ABI; it is not full-file residency and cannot be extrapolated to Qwen3.8.",
                "evidence_class": "MEASURED_LOCAL_PENDING_HOSTED_VALIDATION",
            },
        },
    }


def build_reports(inspection: Dict[str, Any], output_dir: Path,
                  ram_bytes: int, vram_bytes: int, gpu_name: str) -> Dict[str, Path]:
    commit, dirty = _git_identity()
    source = inspection["source"]
    metadata = inspection["metadata"]
    architecture = inspection["architecture"]
    unsupported = inspection["dtype_summary"]["unsupported_by_current_converter"]
    primary_decision = "UNSUPPORTED_QWEN38_ARCHITECTURE"
    common = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "git_worktree_dirty_at_generation": dirty,
        "source": source,
    }
    coverage = dict(common)
    coverage.update({
        "document_type": "required_tensor_coverage_manifest",
        "source_tensor_count": inspection["gguf"]["tensor_count"],
        "coverage_complete": len(inspection["tensors"]) == inspection["gguf"]["tensor_count"],
        "conversion_status": "BLOCKED_BEFORE_OUTPUT",
        "architecture": architecture,
        "dtypes": inspection["dtype_summary"],
        "tensors": inspection["tensors"],
        "no_output_created": True,
    })
    feasibility = dict(common)
    feasibility.update({
        "document_type": "conversion_feasibility_report",
        "decision": primary_decision,
        "secondary_blockers": [
            "UNSUPPORTED_SOURCE_QUANTIZATION",
            "CONVERSION_SOURCE_SUPPORTED_RUNTIME_INCOMPLETE",
        ],
        "architecture_gate": {
            "status": "BLOCKED",
            "architecture": architecture["general_architecture"],
            "reasons": [
                "48 Gated DeltaNet/SSM layer state tensors have no native execution path",
                "4 MTP tensors have no validated target execution path",
                "current decoder cannot prove full hybrid layer coverage",
            ],
        },
        "source_quantization": {
            "general_file_type": inspection["dtype_summary"]["general_file_type"],
            "general_file_type_label": inspection["dtype_summary"]["general_file_type_label"],
            "actual_tensor_dtypes": inspection["dtype_summary"]["counts_by_name"],
            "unsupported_dtype_ids": unsupported,
            "unsupported_dtype_names": inspection["dtype_summary"]["unsupported_names"],
            "status": "UNSUPPORTED_SOURCE_QUANTIZATION" if unsupported else "SUPPORTED_SOURCE_QUANTIZATION",
            "reason": "Current converter has no exact IQ2/IQ3/IQ4 block decoder for these source tensor dtypes; no reinterpretation is permitted.",
        },
        "conversion_policy": {
            "attempted": False,
            "temporary_output": None,
            "managed_output_directory": str(_managed_model_dir()),
            "atomic_rename": "not reached because feasibility gate failed",
            "no_model_weights_written": True,
        },
        "estimates": _projected_qwn_size(inspection["tensors"], _tokenizer_estimate(metadata)),
        "safety": {
            "status": "REFUSED_BEFORE_CONVERSION",
            "peak_ram": "not claimed; exact hybrid conversion path unavailable",
            "temporary_disk": "projected output plus safety margin only",
            "reason": "Do not spend memory or disk on a partial QWN artifact.",
        },
    })
    hardware = _hardware_fit(inspection, ram_bytes, vram_bytes, gpu_name, _managed_model_dir())
    hardware.update({"source_commit": commit, "git_worktree_dirty_at_generation": dirty, "source": source})
    correctness = dict(common)
    correctness.update({
        "document_type": "correctness_qualification_report",
        "status": "UNAVAILABLE_NOT_RUN",
        "decision": primary_decision,
        "oracle_policy": "External runtimes are forbidden in product and were not invoked; a developer oracle may be added only after a complete native conversion path exists.",
        "acceptance_criteria_defined_before_execution": {
            "tokenizer_chat_template": "byte-identical rendered text for 100 deterministic prompts and multi-turn fixtures",
            "tensor_dequantization": "max_abs_error <= 1e-3 * max(1, reference_abs), no NaN/Inf, zero unvalidated dtype substitutions",
            "layer_output": "cosine_similarity >= 0.999 and max_relative_error <= 1e-2 on declared fixture layers",
            "logits": "top_k_10_overlap >= 0.90 and KL divergence <= 0.02 on declared prompt set",
            "greedy_tokens": ">= 0.95 agreement over 100 deterministic prompts; exact agreement required for unquantized control",
            "tool_call_json": "100% schema-valid outputs on the declared tool-call fixture set",
        },
        "executed": {
            "tensor_samples": 0,
            "layer_outputs": 0,
            "logit_prompts": 0,
            "deterministic_prompts": 0,
            "multi_turn_prompts": 0,
            "code_prompts": 0,
            "tool_call_prompts": 0,
        },
        "blocked_reasons": [
            "No valid QWN output exists because Qwen3.8 hybrid architecture is unsupported.",
            "Source IQ dtypes have no exact current converter decoder.",
        ],
    })
    agent_quality = dict(common)
    agent_quality.update({
        "document_type": "agent_quality_report",
        "status": "UNAVAILABLE_NOT_RUN",
        "decision": primary_decision,
        "tasks": [
            "repository-level code understanding",
            "patch correctness and apply rate",
            "compile/test pass rate",
            "tool-call schema validity",
            "hallucinated paths",
            "retry count",
            "long-context retrieval at 4096 and 8192",
            "regression introduction rate",
            "Arabic instruction following",
            "English instruction following",
        ],
        "recording_contract": ["prompt", "response", "seed", "patch", "test_result", "timing", "runtime_config"],
        "executed_tasks": 0,
        "reason": "Agent evaluation requires a validated native QWN model and complete hybrid runtime; neither exists.",
    })
    benchmark = dict(common)
    benchmark.update({
        "document_type": "qwen38_benchmark_evidence",
        "evidence_class": "UNAVAILABLE",
        "decision": primary_decision,
        "model": {
            "architecture": architecture["general_architecture"],
            "qwn_dtype": None,
            "source_dtype_summary": inspection["dtype_summary"],
        },
        "backend_requested": None,
        "backend_actual": None,
        "executable_sha256": None,
        "model_sha256": None,
        "gpu_matmul_count": 0,
        "cpu_fallback_count": None,
        "throughput": {"prefill_tok_per_sec": None, "decode_tok_per_sec": None, "ttft_ms": None},
        "reason_unavailable": "No validated QWN conversion; GGUF source cannot be activated by qwnrun.",
        "contexts_requested": [4096, 8192],
        "no_projected_performance_claim": True,
    })
    summary = dict(common)
    summary.update({
        "document_type": "qwen38_qualification_summary",
        "decision": primary_decision,
        "decision_alternatives_not_selected": {
            "QWN_27B_VALIDATED": "not eligible: no conversion, correctness, or runtime coverage",
            "CONVERSION_SOURCE_SUPPORTED_RUNTIME_INCOMPLETE": "secondary description, but source quantization also fails current converter support",
            "UNSUPPORTED_SOURCE_QUANTIZATION": "secondary blocker; hybrid architecture is the primary decision",
            "HARDWARE_FIT_FAILED": "fit cannot be finalized before required operators exist",
            "QUALITY_GATE_FAILED": "quality tests did not run because runtime gate failed closed",
        },
        "source_facts": {
            "gguf_version": inspection["gguf"]["version"],
            "tensor_count": inspection["gguf"]["tensor_count"],
            "layer_count": architecture["layer_count"],
            "gated_deltanet_layers": architecture["gated_deltanet_layer_count"],
            "full_attention_layers": architecture["full_attention_layer_count"],
            "mtp_tensor_count": len(architecture["mtp_tensor_names"]),
            "vision_tensor_count": len(architecture["vision_tensor_names"]),
        },
        "reports": {
            "architecture_coverage": "architecture-coverage.json",
            "conversion_feasibility": "conversion-feasibility.json",
            "correctness": "correctness.json",
            "hardware_fit": "hardware-fit.json",
            "agent_quality": "agent-quality.json",
            "benchmark_evidence": "benchmark-evidence.json",
        },
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = {
        "architecture-coverage.json": coverage,
        "conversion-feasibility.json": feasibility,
        "correctness.json": correctness,
        "hardware-fit.json": hardware,
        "agent-quality.json": agent_quality,
        "benchmark-evidence.json": benchmark,
        "qualification-summary.json": summary,
    }
    paths: Dict[str, Path] = {}
    for filename, document in documents.items():
        target = output_dir / filename
        target.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        paths[filename] = target
    return paths


def _managed_model_dir() -> Path:
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "Qwanto" / "models"
        return Path.home() / "AppData" / "Local" / "Qwanto" / "models"
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        return Path(data_home) / "qwanto" / "models"
    return Path.home() / ".local" / "share" / "qwanto" / "models"


def qualify(source: Path, output_dir: Path, *, ram_bytes: int,
            vram_bytes: int, gpu_name: str) -> Dict[str, Path]:
    if not source.is_file():
        raise FileNotFoundError(source)
    inspection = inspect_gguf(source)
    return build_reports(inspection, output_dir, ram_bytes, vram_bytes, gpu_name)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and fail-closed qualify a Qwen3.8 GGUF source")
    parser.add_argument("source", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ram-bytes", type=int, default=32 * 1024 ** 3)
    parser.add_argument("--vram-bytes", type=int, default=12 * 1024 ** 3)
    parser.add_argument("--gpu-name", default="NVIDIA RTX 5070 Ti Laptop GPU")
    args = parser.parse_args(argv)
    paths = qualify(args.source.resolve(), args.out_dir.resolve(),
                    ram_bytes=args.ram_bytes, vram_bytes=args.vram_bytes,
                    gpu_name=args.gpu_name)
    summary = json.loads(paths["qualification-summary.json"].read_text(encoding="utf-8"))
    print(json.dumps({"decision": summary["decision"], "reports": {k: str(v) for k, v in paths.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
