"""
qwn_bpw_truth.py — Realistic bits-per-weight (bpw) and disk-size accounting
==========================================================================

The README, benchmark, and convert scripts historically printed hand-written
"effective bpw" numbers (e.g. ``2.70 bpw`` for QWN-HyperVSQ, ``2.10 bpw`` for
QWN-HyperVSQ-2, ``2.625 bpw`` for QWN-VSQ-Ultra).  Those figures **ignored
metadata, container padding, header overhead, outlier sidecars, and the fact
that some tensors (norms, biases, embeddings, the LM head, router weights,
SSM ``A/D/dt``, MTP heads) are stored at higher precision than the bulk
matrix quant format.**

The ``Full Improve Plan.md`` (section 1) calls out the same problem:
``HyperVSQ2`` is in fact ``74 * 8 / 256 = 2.3125 bpw`` and ``HyperVSQ``
is ``138 * 8 / 256 = 4.3125 bpw``; the "sub-2-bit" promise is unachievable
without sparsity / entropy coding / binary weights (section 6 of the plan).

This module replaces every hand-written constant in the project with a single
auto-derived computation:

* payload_bpw    = sum(payload_bytes) * 8 / sum(numel)
* effective_bpw  = sum(bytes_on_disk_for_weights) * 8 / sum(numel)
* size_on_disk   = header + tensor descriptors + sorted overflow index +
                   sum(aligned_payload_bytes) + tail block

The class :class:`QuantFormatSpec` declares the per-format constants in one
place so that **no part of the codebase needs to hard-code a bpw number
ever again**.  The :func:`report` helper produces a JSON-serialisable
``bpw_report`` that can be embedded in benchmark output, in
``quant_plan.json``, or in the README build.

This file is dependency-free and safe to import anywhere in the project.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Container invariants from qwn_convert.py (single source of truth).
# Mirrors the constants used by the on-disk writer; do not edit without
# also editing c/tools/qwn_convert.py.
# ---------------------------------------------------------------------------
HEADER_SIZE: int = 4096            # 4 KiB fixed header
INLINE_MAX: int = 29               # tensor descriptors that fit inline
DESC_SIZE: int = 96                # one inline tensor descriptor
ALIGN_PAGE: int = 4096             # every payload starts on a page boundary
ALIGN_TAIL: int = 64               # payload padded to 64-byte boundary

# Format ids from c/tools/qwn_convert.py.  Mirror only the IDs that the
# project currently writes; unknown ids fall back to F32.
DT_F32 = 0
DT_F16 = 1
DT_BF16 = 2
DT_Q4_0 = 3
DT_Q8_0 = 4
DT_VSQ = 5           # QWN-VSQ       (64-elem superblocks, 4.125 bpw payload)
DT_VSQ_ULTRA = 6     # QWN-VSQ-Ultra (128-elem superblocks, 3.375 bpw payload)
DT_HYPER_VSQ = 7     # QWN-HyperVSQ  (256-elem superblocks, 4.3125 bpw payload)
DT_HYPER_VSQ2 = 8    # QWN-HyperVSQ2 (256-elem superblocks, 2.3125 bpw payload)
DT_RAW = 9           # opaque bytes  (counted at 8 bpw)


@dataclass(frozen=True)
class QuantFormatSpec:
    """Per-format geometric constants.  Bytes-per-block * 8 / block_size = bpw."""

    name: str
    dt_id: int
    block_size: int          # number of weights per block (e.g. 32 for Q4_0)
    block_bytes: int         # total bytes for one block (header + payload)
    payload_bpw: float       # payload bytes * 8 / weights, metadata excluded
    notes: str = ""

    @property
    def bits_per_weight(self) -> float:
        """Alias used in some reports; identical to ``payload_bpw``."""
        return self.payload_bpw


# Single source of truth.  The numbers below come straight from the
# quantizers in c/tools/qwn_convert.py (see quantize_vsq_rows /
# quantize_vsq_ultra_rows / quantize_hyper_vsq_rows /
# quantize_hyper_vsq2_rows and ``_get_quant_dtype_and_size``).  They
# MUST match the byte sizes those functions actually emit; do not edit
# without editing the quantizers at the same time.
Q4_0 = QuantFormatSpec(
    name="Q4_0", dt_id=DT_Q4_0, block_size=32, block_bytes=18,
    payload_bpw=18 * 8 / 32,
    notes="32 values / FP16 scale / 16 packed nibbles per block (4.5 bpw payload)",
)
Q8_0 = QuantFormatSpec(
    name="Q8_0", dt_id=DT_Q8_0, block_size=32, block_bytes=34,
    payload_bpw=34 * 8 / 32,
    notes="32 values / FP16 scale / 32 int8 values per block (8.5 bpw payload)",
)
VSQ = QuantFormatSpec(
    name="QWN-VSQ", dt_id=DT_VSQ, block_size=64, block_bytes=36,
    payload_bpw=36 * 8 / 64,
    notes="QWN-VSQ dual-scale 64-element superblock (4.5 bpw payload)",
)
VSQ_ULTRA = QuantFormatSpec(
    name="QWN-VSQ-Ultra", dt_id=DT_VSQ_ULTRA, block_size=128, block_bytes=70,
    payload_bpw=70 * 8 / 128,
    notes="QWN-VSQ-Ultra 128-element quad-quadrant superblock (4.375 bpw payload)",
)
HYPER_VSQ = QuantFormatSpec(
    name="QWN-HyperVSQ", dt_id=DT_HYPER_VSQ, block_size=256, block_bytes=138,
    payload_bpw=138 * 8 / 256,
    notes="QWN-HyperVSQ 256-element octa-quadrant superblock (4.3125 bpw payload)",
)
HYPER_VSQ2 = QuantFormatSpec(
    name="QWN-HyperVSQ-2", dt_id=DT_HYPER_VSQ2, block_size=256, block_bytes=74,
    payload_bpw=74 * 8 / 256,
    notes="QWN-HyperVSQ-2 256-element sub-2-bit superblock (2.3125 bpw payload)",
)
F16 = QuantFormatSpec(
    name="F16", dt_id=DT_F16, block_size=1, block_bytes=2,
    payload_bpw=16.0, notes="IEEE float16",
)
BF16 = QuantFormatSpec(
    name="BF16", dt_id=DT_BF16, block_size=1, block_bytes=2,
    payload_bpw=16.0, notes="bfloat16",
)
F32 = QuantFormatSpec(
    name="F32", dt_id=DT_F32, block_size=1, block_bytes=4,
    payload_bpw=32.0, notes="IEEE float32",
)
RAW = QuantFormatSpec(
    name="RAW", dt_id=DT_RAW, block_size=1, block_bytes=1,
    payload_bpw=8.0, notes="opaque byte blob",
)

# Convenience map: name -> spec, dt_id -> spec.
SPECS_BY_NAME: Dict[str, QuantFormatSpec] = {
    s.name: s for s in (Q4_0, Q8_0, VSQ, VSQ_ULTRA, HYPER_VSQ, HYPER_VSQ2, F16, BF16, F32, RAW)
}
SPECS_BY_DT: Dict[int, QuantFormatSpec] = {s.dt_id: s for s in SPECS_BY_NAME.values()}


def spec_for(dt_id: int) -> QuantFormatSpec:
    """Return the spec for a dtype id; fall back to F32 for unknown ids."""
    return SPECS_BY_DT.get(int(dt_id), F32)


# ---------------------------------------------------------------------------
# Per-tensor contribution to the container.
# ---------------------------------------------------------------------------
@dataclass
class TensorByteBreakdown:
    """Raw byte counts for a single tensor as the writer sees them."""
    name: str
    numel: int               # number of weights
    dt_id: int
    payload_bytes: int       # exact payload size (pre-padding)
    # After alignment (header pads each tensor to ALIGN_TAIL within its
    # 4 KiB-aligned page).  We never expose these as the *effective* bpw
    # because the rounding is sub-1% but we still report them in the JSON.
    page_aligned_bytes: int
    descriptor_bytes: int    # 96 bytes per inline tensor; 0 once overflow kicks in

    @property
    def spec(self) -> QuantFormatSpec:
        return spec_for(self.dt_id)

    @property
    def payload_bpw(self) -> float:
        if self.numel <= 0:
            return 0.0
        return self.payload_bytes * 8.0 / self.numel


@dataclass
class BpwReport:
    """Result of :func:`report`.  All fields are JSON-serialisable."""
    format_payload_bpw: float            # weighted by numel across the model
    format_effective_bpw: float          # weighted by bytes-on-disk per tensor
    total_weights: int
    payload_bytes_total: int
    descriptor_bytes_total: int
    header_bytes: int
    page_alignment_overhead_bytes: int
    overflow_index_bytes: int
    tail_block_bytes: int
    size_on_disk_bytes: int              # everything, end-to-end
    per_tensor: List[Dict[str, object]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_dict(self) -> Dict[str, object]:
        return {
            "format_payload_bpw": self.format_payload_bpw,
            "format_effective_bpw": self.format_effective_bpw,
            "total_weights": self.total_weights,
            "payload_bytes_total": self.payload_bytes_total,
            "descriptor_bytes_total": self.descriptor_bytes_total,
            "header_bytes": self.header_bytes,
            "page_alignment_overhead_bytes": self.page_alignment_overhead_bytes,
            "overflow_index_bytes": self.overflow_index_bytes,
            "tail_block_bytes": self.tail_block_bytes,
            "size_on_disk_bytes": self.size_on_disk_bytes,
            "per_tensor": list(self.per_tensor),
            "notes": list(self.notes),
        }


def _round_up(n: int, alignment: int) -> int:
    if alignment <= 1:
        return int(n)
    return int((int(n) + alignment - 1) // alignment) * alignment


def _descriptor_bytes_for(n_inline: int, n_overflow: int) -> int:
    """Cost of the inline descriptor array + the overflow FNV-1a hash index.

    The on-disk layout (see ``write_qwn`` in qwn_convert.py) reserves 29
    inline descriptors by default.  The overflow uses 32-byte entries
    (8 byte hash + 8 byte offset + 16 byte name).  These constants are
    duplicated here on purpose to keep this module dependency-free; the
    ``test_bpw_truth.py`` test pins them.
    """
    inline_cost = min(n_inline + n_overflow, INLINE_MAX) * DESC_SIZE
    overflow_cost = max(0, (n_inline + n_overflow) - INLINE_MAX) * 32
    return inline_cost + overflow_cost


def report(tensors: Sequence[TensorByteBreakdown]) -> BpwReport:
    """Aggregate a per-tensor breakdown into a single model-level report.

    Parameters
    ----------
    tensors
        Each entry is the ``TensorByteBreakdown`` as it will actually be
        written to the ``.qwn`` container.

    Notes
    -----
    * ``payload_bpw`` weights by ``numel`` across the whole model.
    * ``effective_bpw`` weights by ``page_aligned_bytes``, which is the
      figure end users actually see on disk once padding is included.
    * Outlier sidecars (``outlier_bytes``) are *not* accounted for here —
      they are still considered weight bytes by this function, so the
      resulting ``payload_bpw`` already reflects the mixed-precision cost.
      Callers that want a "naive baseline" without sidecars should pass
      the post-sidecar ``payload_bytes`` directly.
    """
    total_weights = 0
    payload_total = 0
    page_total = 0
    per_tensor: List[Dict[str, object]] = []

    for t in tensors:
        total_weights += int(t.numel)
        payload_total += int(t.payload_bytes)
        page_total += int(t.page_aligned_bytes)
        per_tensor.append({
            "name": t.name,
            "numel": int(t.numel),
            "dt_id": int(t.dt_id),
            "format": t.spec.name,
            "payload_bytes": int(t.payload_bytes),
            "page_aligned_bytes": int(t.page_aligned_bytes),
            "payload_bpw": round(t.payload_bpw, 6),
        })

    if total_weights == 0:
        payload_bpw = 0.0
        effective_bpw = 0.0
    else:
        payload_bpw = payload_total * 8.0 / total_weights
        effective_bpw = page_total * 8.0 / total_weights

    descriptor_bytes = sum(t.descriptor_bytes for t in tensors)
    descriptor_bytes_total = (
        descriptor_bytes if descriptor_bytes > 0
        else _descriptor_bytes_for(len(tensors), 0)
    )
    overflow_index_bytes = max(
        0, (descriptor_bytes_total // 32) - INLINE_MAX
    ) * 0 + 0  # already counted above; kept for report readability
    header_bytes = HEADER_SIZE
    page_alignment_overhead = max(0, page_total - payload_total)

    # Tail block: the writer pads the final payload to a 4 KiB page and
    # writes the absolute tail-block offset in the last 8 bytes.  This is
    # always at most one extra page, so we charge it conservatively when
    # there are any tensors.
    tail_block_bytes = ALIGN_PAGE if tensors else 0

    size_on_disk = (
        header_bytes
        + descriptor_bytes_total
        + page_total
        + tail_block_bytes
    )

    notes: List[str] = []
    if not tensors:
        notes.append("empty model; no tensors reported")
    if effective_bpw - payload_bpw > 0.05:
        notes.append(
            f"alignment adds {effective_bpw - payload_bpw:.3f} bpw beyond payload "
            f"(page={ALIGN_PAGE}B, tail={ALIGN_TAIL}B)"
        )

    return BpwReport(
        format_payload_bpw=round(payload_bpw, 6),
        format_effective_bpw=round(effective_bpw, 6),
        total_weights=total_weights,
        payload_bytes_total=payload_total,
        descriptor_bytes_total=descriptor_bytes_total,
        header_bytes=header_bytes,
        page_alignment_overhead_bytes=page_alignment_overhead,
        overflow_index_bytes=overflow_index_bytes,
        tail_block_bytes=tail_block_bytes,
        size_on_disk_bytes=size_on_disk,
        per_tensor=per_tensor,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Convenience: estimate file size in GB / MB from a parameter count and a
# single payload_bpw figure, used by the CLI ``--size-forecast`` path.
# ---------------------------------------------------------------------------
def forecast_size(num_params: int, payload_bpw: float,
                  header_bytes: int = HEADER_SIZE,
                  n_tensors: int = 256) -> Dict[str, float]:
    """Return ``{bytes_on_disk, gb, mb}`` for a model of ``num_params`` weights.

    ``n_tensors`` defaults to 256 because typical dense Llama/Qwen models
    expose ~250-400 tensors in the safetensors index.  Caller overrides
    when they know better.
    """
    payload_bytes = int(round(num_params * payload_bpw / 8.0))
    descriptor_bytes_total = _descriptor_bytes_for(n_tensors, 0)
    page_aligned = _round_up(payload_bytes, ALIGN_PAGE)
    tail_block = ALIGN_PAGE if payload_bytes else 0
    bytes_on_disk = header_bytes + descriptor_bytes_total + page_aligned + tail_block
    return {
        "bytes_on_disk": int(bytes_on_disk),
        "gb": bytes_on_disk / (1024 ** 3),
        "mb": bytes_on_disk / (1024 ** 2),
    }


def format_payload_bpw_for_name(name: str) -> float:
    """Helper used by the benchmark / CLI to look up the payload bpw."""
    return SPECS_BY_NAME[name].payload_bpw


__all__ = [
    "HEADER_SIZE", "INLINE_MAX", "DESC_SIZE", "ALIGN_PAGE", "ALIGN_TAIL",
    "DT_F32", "DT_F16", "DT_BF16", "DT_Q4_0", "DT_Q8_0",
    "DT_VSQ", "DT_VSQ_ULTRA", "DT_HYPER_VSQ", "DT_HYPER_VSQ2", "DT_RAW",
    "QuantFormatSpec",
    "Q4_0", "Q8_0", "VSQ", "VSQ_ULTRA", "HYPER_VSQ", "HYPER_VSQ2",
    "F16", "BF16", "F32", "RAW",
    "SPECS_BY_NAME", "SPECS_BY_DT", "spec_for",
    "TensorByteBreakdown", "BpwReport", "report", "forecast_size",
    "format_payload_bpw_for_name",
]