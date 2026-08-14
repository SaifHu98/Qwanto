"""
qwn_roles.py — Tensor role classifier
=====================================

Implements section 4 of ``Full Improve Plan.md``:

    1. موقع العملية في الرسم الحسابي (graph position) — strongest
    2. metadata الخاصة بالمعمارية — strong
    3. علاقات الأبعاد مع بقية التنسورات — medium
    4. الاسم كدليل أخير — fallback

The classifier is the consumer of the arch adapter's :class:`ModelIR`
graph.  For each :class:`TensorNode` it produces a tuple of
``(TensorRole, Confidence)`` and stamps the result back onto the node.

The classifier is intentionally pure-functional: it does not mutate the
input metadata and never reads files, so it can be reused by the
quant planner, the converter, and the benchmark harness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from qwn_model_ir import (
    ATTENTION_ROLES,
    FFN_ROLES,
    MOE_ROLES,
    SSM_ROLES,
    Confidence,
    ModelIR,
    ModelDims,
    TensorNode,
    TensorRole,
)


# ---------------------------------------------------------------------------
# Layer / expert extraction
# ---------------------------------------------------------------------------
_LAYER_PATTERNS = [
    re.compile(r"(?:^|\.)layers?\.(\d+)\."),
    re.compile(r"(?:^|\.)h\.(\d+)\."),
    re.compile(r"(?:^|\.)block\.(\d+)\."),
    re.compile(r"(?:^|\.)layer\.(\d+)\."),
]
_EXPERT_PATTERNS = [
    re.compile(r"(?:^|\.)experts?\.(\d+)\."),
    re.compile(r"(?:^|\.)expert\.(\d+)\."),
    re.compile(r"(?:^|\.)mlp\.experts\.(\d+)\."),
]


def _extract_layer(name: str) -> int:
    for p in _LAYER_PATTERNS:
        m = p.search(name)
        if m:
            return int(m.group(1))
    return -1


def _extract_expert(name: str) -> int:
    for p in _EXPERT_PATTERNS:
        m = p.search(name)
        if m:
            return int(m.group(1))
    return -1


# ---------------------------------------------------------------------------
# Evidence accumulator
# ---------------------------------------------------------------------------
def _add(node: TensorNode, role: TensorRole, score: float,
         reason: str) -> Confidence:
    """Append evidence and bump the confidence of ``node``.

    Returns the *previous* confidence so callers can decide whether to
    override or accumulate.
    """
    node.role = role
    node.role_confidence.score = max(node.role_confidence.score, score)
    node.evidence.append(reason)
    return node.role_confidence


# ---------------------------------------------------------------------------
# Per-rule classification helpers
# ---------------------------------------------------------------------------
def _classify_by_name(node: TensorNode) -> Optional[Tuple[TensorRole, float, str]]:
    n = node.name.lower()
    # MTP / Medusa-like heads
    if "mtp" in n and ("head" in n or "proj" in n or n.endswith(".weight")):
        return TensorRole.MTP_HEAD, 0.85, "name contains 'mtp' and head/proj/weight"
    # Specialised MTP path: many heads sit at the top of the checkpoint.
    # Routers
    if "router" in n or "gate_proj" in n and "mlp" in n and "shared" in n:
        # MLPGate that is actually a router (DeepSeek uses mlp.gate.weight as router)
        if "mlp.gate" in n and any(".experts." in t.name for t in _global_tensors):
            return TensorRole.ROUTER, 0.85, "mlp.gate weight alongside expert tensors"
    if n.endswith("mlp.gate.weight") and any("experts" in t.name for t in _global_tensors):
        return TensorRole.ROUTER, 0.80, "mlp.gate.weight sibling of experts"
    if "router" in n or "routing" in n:
        return TensorRole.ROUTER, 0.90, "name contains 'router'/'routing'"
    if "gate.weight" == n.split(".")[-2:]:
        return None  # too generic; shape classifier handles it
    # MoE expert tensors.  Shared experts must be matched *before* the
    # generic `.experts.` rule because `.shared_experts.` contains that
    # substring as a suffix.
    if ".shared_experts." in n or ".shared_expert." in n:
        return TensorRole.SHARED_EXPERT, 0.92, "name contains '.shared_experts.'"
    if ".experts." in n or ".expert." in n:
        return TensorRole.ROUTED_EXPERT, 0.90, "name contains '.experts.' / '.expert.'"
    # Attention Q/K/V/O
    if n.endswith(".q_proj.weight") or ".q_proj" in n:
        return TensorRole.ATTN_Q, 0.90, "name contains q_proj"
    if n.endswith(".k_proj.weight") or ".k_proj" in n:
        return TensorRole.ATTN_K, 0.90, "name contains k_proj"
    if n.endswith(".v_proj.weight") or ".v_proj" in n:
        return TensorRole.ATTN_V, 0.90, "name contains v_proj"
    if n.endswith(".o_proj.weight") or ".o_proj" in n:
        return TensorRole.ATTN_O, 0.90, "name contains o_proj"
    # fused QKV (Llama/Qwen sometimes pack qkv into one matrix)
    if n.endswith(".qkv_proj.weight") or "qkv_proj" in n:
        return TensorRole.ATTN_QKV_FUSED, 0.90, "name contains qkv_proj"
    # FFN
    if n.endswith(".gate_proj.weight") or "mlp.gate_proj" in n:
        return TensorRole.FFN_GATE, 0.90, "name contains gate_proj"
    if n.endswith(".up_proj.weight") or "mlp.up_proj" in n:
        return TensorRole.FFN_UP, 0.90, "name contains up_proj"
    if n.endswith(".down_proj.weight") or "mlp.down_proj" in n:
        return TensorRole.FFN_DOWN, 0.90, "name contains down_proj"
    if "gate_up_proj" in n:
        return TensorRole.FFN_GATE_UP_FUSED, 0.85, "name contains gate_up_proj"
    # Norm / bias
    if any(k in n for k in ("input_layernorm", "post_attention_layernorm",
                             "pre_feedforward_layernorm", "post_feedforward_layernorm",
                             "norm.weight", "attn_norm", "mlp_norm", "norm")):
        if "norm" in n:
            return TensorRole.NORM, 0.85, "name contains 'norm'"
    if n.endswith(".bias"):
        return TensorRole.BIAS, 0.80, "name ends with .bias"
    # Embeddings / LM head
    if "embed_tokens" in n or "tok_embeddings" in n:
        return TensorRole.EMBED_TOK, 0.95, "name contains embed_tokens/tok_embeddings"
    if "lm_head" in n or "output_proj" in n:
        return TensorRole.LM_HEAD, 0.95, "name contains lm_head/output_proj"
    if "wte" in n or "shared.weight" in n:
        # wte is the token embedding; tied embeddings also match this rule.
        return TensorRole.EMBED_TOK, 0.70, "name contains wte / shared.weight"
    # SSM
    if any(k in n for k in ("a_log", ".a_weight", ".ssm_a", ".mixer.a")):
        return TensorRole.SSM_A, 0.90, "name suggests SSM A"
    if any(k in n for k in (".d_weight", ".ssm_d", "mixer.d", "ssm.d")):
        return TensorRole.SSM_D, 0.90, "name suggests SSM D"
    if "dt_bias" in n or ".ssm_dt" in n or ".dt_proj" in n:
        return TensorRole.SSM_DT, 0.90, "name suggests SSM dt"
    if "conv1d.weight" in n or ".ssm_conv" in n:
        return TensorRole.SSM_CONV, 0.85, "name suggests SSM conv1d"
    if ".mixer.in_proj" in n:
        return TensorRole.SSM_IN, 0.80, "name suggests SSM in_proj"
    if ".mixer.out_proj" in n:
        return TensorRole.SSM_OUT, 0.80, "name suggests SSM out_proj"
    # MLA
    if "kv_a_proj_with_mqa" in n or "kv_a_layernorm" in n:
        return TensorRole.MLA_KV_COMPRESS, 0.85, "name suggests MLA KV compress"
    if "q_a_proj" in n or "q_a_layernorm" in n:
        return TensorRole.MLA_Q_COMPRESS, 0.85, "name suggests MLA Q compress"
    # RoPE tables (rare)
    if "rope" in n and "freqs" in n:
        return TensorRole.ROPE, 0.85, "name suggests RoPE frequency table"
    return None


def _classify_by_shape(node: TensorNode, dims: ModelDims) -> Optional[Tuple[TensorRole, float, str]]:
    """Shape-driven evidence (plan section 4: 'shape relations')."""
    if not node.shape:
        return None
    shp = list(node.shape)
    h = dims.hidden_size
    inter = dims.intermediate_size
    nv = dims.vocab_size
    nh = dims.num_heads
    nkv = dims.num_kv_heads
    hd = dims.head_dim or (h // nh if nh else 0)
    # LM head: vocab_size projection back to vocab
    if len(shp) == 2 and nv and (shp[0] == nv or shp[1] == nv):
        return TensorRole.LM_HEAD, 0.85, f"shape touches vocab_size={nv}"
    # Embeddings
    if len(shp) == 2 and nv and h and (shp == [nv, h] or shp == [h, nv]):
        return TensorRole.EMBED_TOK, 0.80, "shape matches [vocab, hidden]"
    # Norm / bias: 1D tensor of size hidden_size
    if len(shp) == 1 and h and shp[0] == h:
        return TensorRole.NORM, 0.75, "1D shape equals hidden_size (norm candidate)"
    # Attention Q / K / V / O matrix
    if len(shp) == 2 and h:
        out_dim, in_dim = shp[0], shp[1]
        if in_dim == h:
            # Q: [num_heads * head_dim, hidden]
            if hd and out_dim == nh * hd:
                return TensorRole.ATTN_Q, 0.75, "shape matches [Q, hidden]"
            # O: [hidden, num_heads * head_dim]
            if hd and out_dim == h:
                return TensorRole.ATTN_O, 0.75, "shape matches [hidden, num_heads*head_dim]"
            # K / V: GQA -> out = num_kv_heads * head_dim
            if hd and nkv and out_dim == nkv * hd:
                # K and V both match; pick K as the conservative default.
                # The role is overridable once we see the suffix in name pass.
                return TensorRole.ATTN_K, 0.55, "shape matches KV projection (K by default)"
        # FFN gate/up: [intermediate, hidden] or [hidden, intermediate]
        if inter:
            if out_dim == inter and in_dim == h:
                return TensorRole.FFN_GATE, 0.55, "shape matches [intermediate, hidden] (gate)"
            if out_dim == h and in_dim == inter:
                return TensorRole.FFN_DOWN, 0.55, "shape matches [hidden, intermediate] (down)"
    return None


def _classify_by_layer(node: TensorNode) -> int:
    """Extract layer index from the tensor name."""
    return _extract_layer(node.name)


def _classify_by_expert(node: TensorNode) -> int:
    return _extract_expert(node.name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
# Internal registry of the tensor list so name-pattern rules that need
# global context (e.g. "is there a sibling expert?") can introspect it.
_global_tensors: List[TensorNode] = []


def classify_tensor(node: TensorNode,
                    dims: ModelDims,
                    all_tensors: Optional[Sequence[TensorNode]] = None
                    ) -> TensorNode:
    """Stamp ``node`` with a role and confidence in place.

    The function follows the rank order from the plan:
    1. graph position (we don't have an explicit graph yet — handled by
       the arch adapter via tensor grouping; name+shape handle the rest)
    2. arch metadata (the dims parameter)
    3. shape relations
    4. name pattern
    """
    if all_tensors is not None:
        # Module-level cache so _classify_by_name can see siblings.
        global _global_tensors
        _global_tensors = list(all_tensors)

    node.layer_index = _classify_by_layer(node)
    node.expert_index = _classify_by_expert(node)
    if not node.role_confidence.evidence:
        node.role_confidence = Confidence()

    # 1. Shape evidence first (it overrides weak name matches like "mlp.gate")
    shape = _classify_by_shape(node, dims)
    if shape is not None:
        role, score, reason = shape
        _add(node, role, score, reason)

    # 2. Name evidence — wins unless it conflicts with high-confidence shape
    name_hit = _classify_by_name(node)
    if name_hit is not None:
        role, score, reason = name_hit
        existing = node.role
        if existing != TensorRole.UNKNOWN and existing != role:
            # Conflict: keep the higher-confidence one.
            if score > node.role_confidence.score:
                _add(node, role, score, reason)
            else:
                node.evidence.append(
                    f"name suggests {role.value} but shape suggests "
                    f"{existing.value}; deferring to shape")
        else:
            _add(node, role, score, reason)

    # 3. Tied embeddings: same shape AND same dtype_id as embed_tokens,
    #    AND a stable fingerprint will be added by the writer; here we
    #    mark the suspicion so the planner can de-dup storage later.
    if node.role == TensorRole.LM_HEAD and node.shape and dims.vocab_size:
        if node.shape[0] == dims.vocab_size and node.shape[-1] == dims.hidden_size:
            node.hard_constraints.append(
                "candidate for tied embedding with embed_tokens (verify by hash)")

    return node


def classify_all(graph: ModelIR) -> ModelIR:
    """Stamp every node in ``graph`` and update ``graph.confidence``.

    The function is the single entry point the planner / convert
    pipeline should use.
    """
    dims = graph.dims
    for n in graph.nodes:
        classify_tensor(n, dims, all_tensors=graph.nodes)

    # Tied-embedding heuristic: if LM head and embed_tokens are present
    # and have identical dtype_id and shape, mark TIED_EMBED.
    embed = next((n for n in graph.nodes if n.role == TensorRole.EMBED_TOK), None)
    head = next((n for n in graph.nodes if n.role == TensorRole.LM_HEAD), None)
    if embed is not None and head is not None:
        if (embed.shape == head.shape
                and embed.dtype_id == head.dtype_id
                and embed.numel == head.numel):
            head.role = TensorRole.TIED_EMBED
            head.role_confidence.score = max(head.role_confidence.score, 0.80)
            head.evidence.append("shape/dtype match with embed_tokens (tied)")
            # The actual storage dedup happens at write-time; the writer
            # is responsible for verifying with a hash before eliminating
            # the duplicate.

    # MTP heads: scan for any tensor that lives outside the layer.*. pattern
    # and looks like a prediction head (mtp_proj, shared_head_N, ...).
    mtp_candidates = [n for n in graph.nodes
                      if n.layer_index == -1
                      and ("mtp" in n.name.lower() or "head_proj" in n.name.lower())]
    if mtp_candidates and not graph.mtp_plan.head_names:
        graph.mtp_plan.enabled = True
        graph.mtp_plan.depth = len(mtp_candidates)
        graph.mtp_plan.head_names = [c.name for c in mtp_candidates]
        graph.mtp_plan.notes = "detected by role classifier"

    return graph


__all__ = ["classify_tensor", "classify_all",
           "_extract_layer", "_extract_expert"]