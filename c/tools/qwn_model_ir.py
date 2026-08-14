"""
qwn_model_ir.py — QWN Intermediate Representation (QWN-IR)
==========================================================

Per section 2 and 3 of ``Full Improve Plan.md`` the engine cannot be made
universal by relying on tensor names alone.  The right design is:

* a single uniform :class:`ModelIR` produced by an :class:`ArchAdapter`
* the adapter is selected by an :class:`ArchRegistry` (see
  ``qwn_arch_registry.py``)
* each tensor is annotated with a :class:`TensorRole` produced by
  ``qwn_roles.py``
* quantization decisions are made off the IR by ``qwn_quant_plan.py``

This module only defines the data classes.  Everything here is dependency-
free, JSON-serialisable, and uses ``dataclasses`` to match the style of
``c/tools/qwn_convert.py``.

Schema (informal)
-----------------
::

    Confidence       { score: 0..1, evidence: [str], hard_constraints: [str] }
    TensorRole       enum (EMBED_TOK, ATTN_Q, ATTN_K, ... MTP_HEAD, UNKNOWN)
    CacheLayout      { kind: 'paged_kv' | 'ssm_state' | 'mla', block_tokens, ... }
    MTPPlan          { enabled, depth, head_names, expected_speedup_range }
    ValidationIssue  { severity: 'error'|'warn', tensor, code, message }
    ValidationReport { issues: [ValidationIssue], ok: bool }
    ModelIR          { arch, dims, nodes: [TensorNode], kv_layout, mtp_plan,
                       adapter_name, confidence, validation }

TensorNode
    { name, shape: [int], dtype_id, role: TensorRole, role_confidence,
      layer_index, expert_index, fused_with, evidence, hard_constraints }
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Tensor roles.  Centralised here so every consumer (planner, role
# classifier, benchmark) references the same enum.  Numeric values are
# stable for JSON round-trips.
# ---------------------------------------------------------------------------
class TensorRole(str, enum.Enum):
    # embeddings / output head
    EMBED_TOK = "embed_tok"
    LM_HEAD = "lm_head"
    TIED_EMBED = "tied_embed"           # share storage with EMBED_TOK
    # attention projections
    ATTN_Q = "attn_q"
    ATTN_K = "attn_k"
    ATTN_V = "attn_v"
    ATTN_O = "attn_o"
    ATTN_QKV_FUSED = "attn_qkv_fused"
    # normalisation and biases
    NORM = "norm"                       # RMSNorm/gamma weight
    BIAS = "bias"
    # MLP / FFN
    FFN_GATE = "ffn_gate"
    FFN_UP = "ffn_up"
    FFN_DOWN = "ffn_down"
    FFN_GATE_UP_FUSED = "ffn_gate_up_fused"
    # MoE
    ROUTER = "router"
    SHARED_EXPERT = "shared_expert"
    ROUTED_EXPERT = "routed_expert"
    # SSM (Mamba / Mamba-2)
    SSM_A = "ssm_a"
    SSM_D = "ssm_d"
    SSM_DT = "ssm_dt"
    SSM_IN = "ssm_in"
    SSM_OUT = "ssm_out"
    SSM_CONV = "ssm_conv"
    SSM_STATE = "ssm_state"
    # MTP / Medusa-like heads
    MTP_HEAD = "mtp_head"
    # RoPE tables (rarely stored, but a few HF checkpoints include them)
    ROPE = "rope"
    # KV cache projections (rarely explicit; recorded for completeness)
    KV_PROJ = "kv_proj"
    # MLA-specific
    MLA_KV_COMPRESS = "mla_kv_compress"
    MLA_Q_COMPRESS = "mla_q_compress"
    # position embeddings / misc
    POS_EMBED = "pos_embed"
    # fallback
    UNKNOWN = "unknown"


# Roles that must keep at least their current precision (used by the
# quant planner as a "do not go below X" hint).  See plan section 5.
PROTECTED_ROLES = frozenset({
    TensorRole.EMBED_TOK, TensorRole.LM_HEAD, TensorRole.TIED_EMBED,
    TensorRole.NORM, TensorRole.BIAS, TensorRole.ROUTER,
    TensorRole.SSM_A, TensorRole.SSM_D, TensorRole.SSM_DT,
    TensorRole.MTP_HEAD, TensorRole.ROPE, TensorRole.POS_EMBED,
})

ATTENTION_ROLES = frozenset({
    TensorRole.ATTN_Q, TensorRole.ATTN_K, TensorRole.ATTN_V,
    TensorRole.ATTN_O, TensorRole.ATTN_QKV_FUSED,
})

FFN_ROLES = frozenset({
    TensorRole.FFN_GATE, TensorRole.FFN_UP, TensorRole.FFN_DOWN,
    TensorRole.FFN_GATE_UP_FUSED,
})

MOE_ROLES = frozenset({
    TensorRole.ROUTER, TensorRole.SHARED_EXPERT, TensorRole.ROUTED_EXPERT,
})

SSM_ROLES = frozenset({
    TensorRole.SSM_A, TensorRole.SSM_D, TensorRole.SSM_DT,
    TensorRole.SSM_IN, TensorRole.SSM_OUT, TensorRole.SSM_CONV,
    TensorRole.SSM_STATE,
})


# ---------------------------------------------------------------------------
# Confidence + validation
# ---------------------------------------------------------------------------
@dataclass
class Confidence:
    """How sure we are about an inference (architecture, tensor role, ...)."""
    score: float = 0.0                  # 0..1
    evidence: List[str] = field(default_factory=list)
    hard_constraints: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        score = float(self.score)
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        self.score = score

    @property
    def is_weak(self) -> bool:
        """``True`` below the 0.90 threshold that disables aggressive Q2."""
        return self.score < 0.90

    def to_dict(self) -> Dict[str, object]:
        return {"score": self.score, "evidence": list(self.evidence),
                "hard_constraints": list(self.hard_constraints)}


@dataclass
class ValidationIssue:
    severity: str                       # 'error' | 'warn' | 'info'
    tensor: str = ""
    code: str = ""
    message: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {"severity": self.severity, "tensor": self.tensor,
                "code": self.code, "message": self.message}


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warn"]

    def add(self, severity: str, tensor: str, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(severity, tensor, code, message))

    def to_dict(self) -> Dict[str, object]:
        return {"ok": self.ok,
                "issues": [i.to_dict() for i in self.issues]}


# ---------------------------------------------------------------------------
# Cache layouts
# ---------------------------------------------------------------------------
@dataclass
class CacheLayout:
    """KV cache or SSM state plan, consumed by ``qwn_paged_kv.c`` /
    ``qwn_state_pool.c``.

    ``kind`` selects the runtime path:

    * ``paged_kv``      – classic transformer KV cache (paged).
    * ``mla``           – DeepSeek-style Multi-Latent Attention.
    * ``ssm_state``     – Mamba-2 state pool (no KV at all).
    * ``hybrid``        – transformer + SSM mixed per layer.
    """
    kind: str = "paged_kv"
    block_tokens: int = 16             # logical tokens per paged block
    page_bytes: int = 4096             # physical page size for the allocator
    bytes_per_token: int = 0           # derived (KV path)
    layers: int = 0
    kv_heads: int = 0
    head_dim: int = 0
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "block_tokens": self.block_tokens,
            "page_bytes": self.page_bytes,
            "bytes_per_token": self.bytes_per_token,
            "layers": self.layers,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "extra": dict(self.extra),
        }


# ---------------------------------------------------------------------------
# MTP plan
# ---------------------------------------------------------------------------
@dataclass
class MTPPlan:
    enabled: bool = False
    depth: int = 0
    head_names: List[str] = field(default_factory=list)
    expected_speedup_range: List[float] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "depth": self.depth,
            "head_names": list(self.head_names),
            "expected_speedup_range": list(self.expected_speedup_range),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Tensor node (one per safetensors / GGUF tensor)
# ---------------------------------------------------------------------------
@dataclass
class TensorNode:
    name: str
    shape: List[int]
    dtype_id: int = 0
    role: TensorRole = TensorRole.UNKNOWN
    role_confidence: Confidence = field(default_factory=Confidence)
    layer_index: int = -1              # -1 if not a per-layer tensor
    expert_index: int = -1             # -1 if not an expert tensor
    fused_with: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    hard_constraints: List[str] = field(default_factory=list)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype_id": int(self.dtype_id),
            "role": self.role.value,
            "role_confidence": self.role_confidence.to_dict(),
            "layer_index": int(self.layer_index),
            "expert_index": int(self.expert_index),
            "fused_with": list(self.fused_with),
            "evidence": list(self.evidence),
            "hard_constraints": list(self.hard_constraints),
        }


# ---------------------------------------------------------------------------
# Model-wide IR
# ---------------------------------------------------------------------------
@dataclass
class ModelDims:
    """Architecture-level hyperparameters (no behaviour, just numbers)."""
    hidden_size: int = 0
    intermediate_size: int = 0
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    rope_theta: float = 0.0
    rope_scaling: Dict[str, object] = field(default_factory=dict)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    max_position_embeddings: int = 0
    ssm_state_size: int = 0
    ssm_conv_kernel: int = 0
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "rope_theta": self.rope_theta,
            "rope_scaling": dict(self.rope_scaling),
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "max_position_embeddings": self.max_position_embeddings,
            "ssm_state_size": self.ssm_state_size,
            "ssm_conv_kernel": self.ssm_conv_kernel,
            "extra": dict(self.extra),
        }


@dataclass
class ModelIR:
    """QWN-IR.  The single object every consumer downstream reads."""
    arch: str = "unknown"
    family: str = "unknown"             # 'dense' | 'moe' | 'ssm' | 'hybrid'
    dims: ModelDims = field(default_factory=ModelDims)
    nodes: List[TensorNode] = field(default_factory=list)
    kv_layout: CacheLayout = field(default_factory=CacheLayout)
    mtp_plan: MTPPlan = field(default_factory=MTPPlan)
    adapter_name: str = ""
    confidence: Confidence = field(default_factory=Confidence)
    validation: ValidationReport = field(default_factory=ValidationReport)

    # ---- convenience accessors -----------------------------------------
    def by_role(self, role: TensorRole) -> List[TensorNode]:
        return [n for n in self.nodes if n.role == role]

    def by_layer(self, layer_index: int) -> List[TensorNode]:
        return [n for n in self.nodes if n.layer_index == layer_index]

    def total_params(self, role_predicate=None) -> int:
        total = 0
        for n in self.nodes:
            if role_predicate is None or role_predicate(n.role):
                total += n.numel
        return total

    def to_dict(self) -> Dict[str, object]:
        return {
            "arch": self.arch,
            "family": self.family,
            "dims": self.dims.to_dict(),
            "nodes": [n.to_dict() for n in self.nodes],
            "kv_layout": self.kv_layout.to_dict(),
            "mtp_plan": self.mtp_plan.to_dict(),
            "adapter_name": self.adapter_name,
            "confidence": self.confidence.to_dict(),
            "validation": self.validation.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


__all__ = [
    "TensorRole",
    "PROTECTED_ROLES", "ATTENTION_ROLES", "FFN_ROLES", "MOE_ROLES", "SSM_ROLES",
    "Confidence", "ValidationIssue", "ValidationReport",
    "CacheLayout", "MTPPlan", "TensorNode", "ModelDims", "ModelIR",
]