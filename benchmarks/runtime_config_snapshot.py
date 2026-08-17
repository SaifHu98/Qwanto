"""Canonical, JSON-safe runtime configuration snapshots for benchmark evidence."""

from __future__ import annotations

import hashlib
from typing import Any


def make_runtime_config_snapshot(
    *,
    backend: str,
    context_size: int,
    max_tokens: int,
    seed: int,
    prompt: str,
    threads: int | None,
    warmup_tokens: int,
    thinking_mode: str = "none",
    decode_function: str = "qwn_decoder_generate",
    temperature: float = 0.0,
    top_p: float = 1.0,
    gpu_device: int | str = "auto",
    kv_cache_mode: str = "fp16",
    quantization: str = "auto",
    kernel_requested: str = "auto",
    speculative_decoding: bool = False,
    fused_kernel: bool = False,
) -> dict[str, Any]:
    prompt_bytes = prompt.encode("utf-8")
    return {
        "backend_requested": backend,
        "backend_actual": "Unavailable",
        "gpu_device": gpu_device,
        "cpu_threads_requested": threads if threads is not None else "auto",
        "cpu_threads_active": "Unavailable",
        "context_size": context_size,
        "max_tokens": max_tokens,
        "seed": seed,
        "prompt": prompt,
        "prompt_length_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "warmup_tokens": warmup_tokens,
        "thinking_mode": thinking_mode,
        "decode_function": decode_function,
        "kv_cache_mode": kv_cache_mode,
        "quantization": quantization,
        "kernel_requested": kernel_requested,
        "temperature": temperature,
        "top_p": top_p,
        "batch_size": 1,
        "sampler": {
            "temperature": temperature,
            "top_p": top_p,
            "greedy": temperature <= 0.0,
        },
        "speculative_decoding": speculative_decoding,
        "fused_kernel": fused_kernel,
    }


def comparable_runtime_config(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return fields that must agree when one-shot and serve flags are equivalent."""
    ignored = {"backend_actual", "cpu_threads_active"}
    return {key: value for key, value in snapshot.items() if key not in ignored}


def update_runtime_config_snapshot(snapshot: dict[str, Any], fields: dict[str, Any]) -> None:
    """Overlay only values emitted by qwnrun; absent values remain unavailable."""
    mapping = {
        "backend_actual": "backend_actual",
        "active_threads": "cpu_threads_active",
        "kernel": "kernel_selected",
        "model_dtype": "model_dtype",
        "dispatch_reason": "dispatch_reason",
        "thinking_mode": "thinking_mode",
        "decode_function": "decode_function",
        "temperature": "temperature",
        "top_p": "top_p",
        "config_backend": "backend_requested",
        "context_size": "context_size",
        "max_tokens": "max_tokens",
        "seed": "seed",
        "kv_cache_mode": "kv_cache_mode",
        "quantization": "quantization",
        "kernel_requested": "kernel_requested",
    }
    for source, target in mapping.items():
        if source in fields:
            snapshot[target] = fields[source]

