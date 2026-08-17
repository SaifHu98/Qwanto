"""
qwn_arch_registry.py — Architecture detection and ArchAdapter registry
======================================================================

Per section 3 of ``Full Improve Plan.md`` a single-name-based detector is
not enough.  We need:

* a registry of :class:`ArchAdapter` implementations, one per family
* a confidence score (``Confidence``) that combines evidence
* a hard-constraint check that **disables aggressive Q2** when the
  confidence falls below ``0.90``
* a graceful fallback (the "classifier as last resort" mentioned in
  the plan) for unknown architectures

The adapter contract is intentionally small:

.. code-block:: python

    class ArchAdapter:
        def detect(metadata, tensors) -> Confidence
        def build_graph(metadata, tensors) -> ModelIR
        def classify_tensor(node) -> TensorRole      # delegated to qwn_roles
        def validate_shapes(graph) -> ValidationReport
        def kv_layout(graph) -> CacheLayout
        def mtp_layout(graph) -> MTPPlan | None

This module ships with built-in adapters for:

* Llama / Qwen / Mistral / Gemma / Phi-style dense transformers
* MoE families (DeepSeek / GLM-5.2 / OLMoE)
* Mamba / Mamba-2 SSM
* Hybrid Transformer-SSM (Mamba-2 + attention, e.g. Jamba / Zamba)
* MTP / Medusa-like multi-token prediction heads

The fallback :class:`UnknownAdapter` returns a safe ``Q8/FP16`` plan only;
it never picks Q2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from qwn_model_ir import (
    ATTENTION_ROLES,
    Confidence,
    FFN_ROLES,
    MOE_ROLES,
    MTPPlan,
    CacheLayout,
    ModelDims,
    ModelIR,
    SSM_ROLES,
    TensorNode,
    TensorRole,
    ValidationIssue,
    ValidationReport,
)


def _gguf_architecture(metadata: Dict[str, object]) -> str:
    """Normalize Hugging Face and GGUF architecture metadata for detection."""
    raw = metadata.get("architectures")
    if raw is None:
        raw = metadata.get("general.architecture", "")
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return str(raw or "").lower()


def _model_type(metadata: Dict[str, object]) -> str:
    return str(metadata.get("model_type", metadata.get("general.architecture", "")) or "").lower()


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------
class ArchAdapter:
    """Base class.  Subclasses set ``name`` / ``family`` and override hooks."""

    name: str = "unknown"
    family: str = "unknown"           # 'dense' | 'moe' | 'ssm' | 'hybrid'
    priority: int = 50                # higher wins when confidence ties

    # ----- optional pattern hooks ---------------------------------------
    arch_patterns: Tuple[str, ...] = ()
    model_type_patterns: Tuple[str, ...] = ()

    # ----- required hooks -----------------------------------------------
    def detect(self, metadata: Dict[str, object],
               tensors: Sequence[TensorNode]) -> Confidence:
        raise NotImplementedError

    def build_graph(self, metadata: Dict[str, object],
                    tensors: Sequence[TensorNode]) -> ModelIR:
        raise NotImplementedError

    def validate_shapes(self, graph: ModelIR) -> ValidationReport:
        return ValidationReport()

    def kv_layout(self, graph: ModelIR) -> CacheLayout:
        return CacheLayout()

    def mtp_layout(self, graph: ModelIR) -> Optional[MTPPlan]:
        return None


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------
class DenseTransformerAdapter(ArchAdapter):
    """Generic dense transformer with explicit ``known``/``generic`` modes.

    The user critique of an earlier revision was that ``dense`` was too
    coarse: a model that simply has the right *tensor names* would be
    labelled ``dense`` and then accepted for advanced features (MTP,
    MLA, GQA) that may not actually be present.  We split the dense
    path in two:

    * ``known_dense_transformer`` — ``architectures``/``model_type``
      literally matches one of the patterns below (Llama, Qwen2, …).
      Confidence ≥ 0.90.  MTP / MLA / GQA flags may be enabled after
      a separate explicit check.
    * ``generic_dense_transformer`` — only the tensor-name heuristic
      matches.  Confidence is capped at 0.80 and the adapter refuses
      any flag that requires an explicit architecture match.

    Detect via ``architectures`` field or tensor name shape like
    ``model.layers.{L}.self_attn.q_proj.weight``.
    """
    name = "known_dense_transformer"
    family = "dense"
    priority = 60

    arch_patterns = (
        "llama", "qwen", "qwen2", "qwen3", "mistral", "gemma", "phi",
        "phi3", "phi4", "starcoder", "internlm", "baichuan", "yi",
        "falcon", "deepseek_v2-dense",
    )
    model_type_patterns = ("llama", "qwen2", "qwen3", "mistral", "gemma", "phi")

    def detect(self, metadata, tensors):
        conf = Confidence()
        arch = _gguf_architecture(metadata)
        model_type = _model_type(metadata)

        # We do NOT mutate ``self.name`` here: that would override the
        # class-level name on the instance and break MoEAdapter, which
        # inherits from us.  Instead, the chosen mode is exposed via
        # ``Confidence.evidence`` and the registry surfaces it through
        # ``_resolved_name`` below.
        if any(p in arch for p in self.arch_patterns):
            conf.score = 0.95
            conf.evidence.append(f"architectures[0] = {arch!r} matches known dense arch")
            mode = "known_dense_transformer"
        elif any(p in model_type for p in self.model_type_patterns):
            conf.score = 0.90
            conf.evidence.append(f"model_type = {model_type!r} matches known dense arch")
            mode = "known_dense_transformer"
        else:
            dense_hits = 0
            for t in tensors:
                if ".self_attn." in t.name or ".mlp." in t.name:
                    dense_hits += 1
            if dense_hits >= max(4, len(tensors) * 0.20):
                conf.score = 0.70
                conf.evidence.append(
                    f"{dense_hits} tensors match dense attention/MLP naming "
                    "(heuristic only; not verified against metadata)")
                mode = "generic_dense_transformer"
                conf.hard_constraints.append(
                    "generic_dense_transformer: MTP, MLA, fused-QKV, "
                    "and explicit GQA flags are disabled until metadata "
                    "verification succeeds")
            else:
                conf.score = 0.20
                conf.evidence.append("no clear dense evidence")
                conf.hard_constraints.append("confidence too low; refusing Q2")
                mode = "known_dense_transformer"  # placeholder; unknown wins

        conf.evidence.append(f"resolved_dense_mode={mode}")
        return conf

    def build_graph(self, metadata, tensors):
        dims = ModelDims(
            hidden_size=int(metadata.get("hidden_size", 0)),
            intermediate_size=int(metadata.get("intermediate_size", 0)),
            num_layers=int(metadata.get("num_hidden_layers", 0)),
            num_heads=int(metadata.get("num_attention_heads", 0)),
            num_kv_heads=int(metadata.get("num_key_value_heads",
                                          metadata.get("num_attention_heads", 0))),
            head_dim=int(metadata.get("head_dim", 0)),
            vocab_size=int(metadata.get("vocab_size", 0)),
            rope_theta=float(metadata.get("rope_theta", 0.0)),
            rope_scaling=dict(metadata.get("rope_scaling", {}) or {}),
            max_position_embeddings=int(metadata.get("max_position_embeddings", 0)),
        )
        if dims.head_dim == 0 and dims.num_heads:
            dims.head_dim = dims.hidden_size // dims.num_heads

        # If the registry has already resolved this adapter into a
        # known/generic mode (set by detect()), propagate it to the IR
        # arch field.  Otherwise fall back to the class name.
        arch_name = getattr(self, "_resolved_arch", self.name)
        graph = ModelIR(arch=arch_name, family=self.family, dims=dims,
                        adapter_name=self.name)
        graph.nodes = list(tensors)
        return graph

    def validate_shapes(self, graph):
        rep = ValidationReport()
        d = graph.dims
        if d.num_heads and d.head_dim and d.hidden_size:
            if d.num_heads * d.head_dim != d.hidden_size:
                rep.add("error", "", "shape.heads_head_dim",
                        f"{d.num_heads}*{d.head_dim} != {d.hidden_size}")
        for n in graph.nodes:
            if n.role in ATTENTION_ROLES and n.shape:
                # expect shape compatible with [hidden, hidden] or fused variants
                if len(n.shape) >= 2 and d.hidden_size:
                    if n.shape[-2] not in (d.hidden_size, d.num_kv_heads * d.head_dim):
                        rep.add("warn", n.name, "shape.att",
                                f"unexpected attn shape {n.shape}")
        return rep

    def kv_layout(self, graph):
        d = graph.dims
        bpt = 0
        if d.num_layers and d.num_kv_heads and d.head_dim:
            # 2 for K and V, 2 bytes for FP16 KV storage
            bpt = 2 * d.num_layers * d.num_kv_heads * d.head_dim * 2
        return CacheLayout(kind="paged_kv", block_tokens=16, page_bytes=4096,
                           bytes_per_token=bpt, layers=d.num_layers,
                           kv_heads=d.num_kv_heads, head_dim=d.head_dim)

    def mtp_layout(self, graph):
        # Look for explicit multi-token prediction head tensors in the model.
        head_names = [n.name for n in graph.nodes
                      if n.role == TensorRole.MTP_HEAD]
        if not head_names:
            return None
        return MTPPlan(enabled=True, depth=len(head_names), head_names=head_names,
                       expected_speedup_range=[1.3, 1.8],
                       notes="native dense MTP/Medusa heads")


class MoEAdapter(DenseTransformerAdapter):
    """DeepSeek / GLM / OLMoE: dense scaffolding + routed + shared experts."""
    name = "moe"
    family = "moe"
    priority = 70

    arch_patterns = (
        "deepseek", "deepseek_v2", "deepseek_v3", "glm", "olmoe", "mixtral",
        "qwen2_moe", "qwen3_moe", "jamba", "switch",
    )
    model_type_patterns = ("deepseek", "mixtral", "qwen2_moe", "olmoe", "glm")

    def detect(self, metadata, tensors):
        conf = super().detect(metadata, tensors)
        # Look for explicit expert tensors to bump confidence
        expert_hits = sum(1 for t in tensors if ".experts." in t.name
                          or ".expert." in t.name or "mlp.experts" in t.name)
        if expert_hits > 0:
            conf.score = max(conf.score, 0.95)
            conf.evidence.append(f"{expert_hits} expert tensors detected")
            conf.hard_constraints = [
                c for c in conf.hard_constraints if "Q2" not in c]
            # MoE supports Q2A in routed experts because only top_k are active.
            conf.hard_constraints.append(
                "routed experts can use Q2A; shared experts and router must stay >= Q4")
        elif conf.score < 0.90:
            conf.hard_constraints.append("MoE not confirmed; refusing Q2A on experts")
        return conf

    def build_graph(self, metadata, tensors):
        graph = super().build_graph(metadata, tensors)
        graph.arch = self.name
        graph.family = self.family
        graph.adapter_name = self.name
        graph.dims.num_experts = int(metadata.get("num_experts", 0)
                                      or metadata.get("n_routed_experts", 0))
        graph.dims.num_experts_per_tok = int(
            metadata.get("num_experts_per_tok", 0)
            or metadata.get("moe_top_k", 0)
            or metadata.get("top_k", 0))
        return graph

    def kv_layout(self, graph):
        # MLA models have their own cache layout
        if "mla" in (graph.arch or "").lower():
            d = graph.dims
            return CacheLayout(
                kind="mla", block_tokens=16, page_bytes=4096,
                bytes_per_token=0,
                layers=d.num_layers, kv_heads=d.num_kv_heads,
                head_dim=d.head_dim,
                extra={"kv_latent_dim": int(graph.dims.extra.get("kv_lora_rank", 0))})
        return super().kv_layout(graph)


class MambaAdapter(ArchAdapter):
    """Mamba / Mamba-2 pure SSM."""
    name = "mamba"
    family = "ssm"
    priority = 55

    arch_patterns = ("mamba", "mamba2", "falcon_mamba", "zamba")
    model_type_patterns = ("mamba", "mamba2")

    def detect(self, metadata, tensors):
        conf = Confidence()
        arch = _gguf_architecture(metadata)
        mt = _model_type(metadata)
        if any(p in arch for p in self.arch_patterns) or any(p in mt for p in self.model_type_patterns):
            conf.score = 0.95
            conf.evidence.append(f"arch={arch!r} model_type={mt!r}")
            conf.hard_constraints.append("SSM A/D/dt must stay >= FP16 to prevent drift")
        else:
            ssm_hits = sum(1 for t in tensors if ".ssm." in t.name
                           or ".mixer." in t.name)
            if ssm_hits > 0:
                conf.score = 0.75
                conf.evidence.append(f"{ssm_hits} ssm tensors found")
            else:
                conf.score = 0.20
                conf.hard_constraints.append("no SSM evidence")
        return conf

    def build_graph(self, metadata, tensors):
        dims = ModelDims(
            hidden_size=int(metadata.get("hidden_size", 0)),
            intermediate_size=int(metadata.get("intermediate_size", 0)),
            num_layers=int(metadata.get("num_hidden_layers", 0)),
            vocab_size=int(metadata.get("vocab_size", 0)),
            ssm_state_size=int(metadata.get("ssm_state_size",
                                           metadata.get("state_size", 0))),
            ssm_conv_kernel=int(metadata.get("ssm_conv_kernel",
                                            metadata.get("conv_kernel", 0))),
        )
        graph = ModelIR(arch=self.name, family=self.family, dims=dims,
                        adapter_name=self.name)
        graph.nodes = list(tensors)
        return graph

    def validate_shapes(self, graph):
        rep = ValidationReport()
        # Pure SSM must not have KV cache expectations
        for n in graph.nodes:
            if n.role in ATTENTION_ROLES:
                rep.add("warn", n.name, "ssm.unexpected_attn",
                        "SSM model contains attention tensors; check hybrid")
        return rep

    def kv_layout(self, graph):
        # No KV at all; expose a state pool layout instead.
        d = graph.dims
        return CacheLayout(
            kind="ssm_state", block_tokens=1, page_bytes=4096,
            bytes_per_token=(d.ssm_state_size or 0) * d.num_layers * 4,
            layers=d.num_layers,
            extra={"conv_kernel": d.ssm_conv_kernel,
                   "state_size": d.ssm_state_size})


class HybridSSMAdapter(MambaAdapter):
    """Hybrid Transformer-SSM (Jamba, Zamba, etc.)."""
    name = "hybrid_ssm"
    family = "hybrid"
    priority = 65

    arch_patterns = ("jamba", "zamba", "bamba", "falcon-h1", "recurrentgemma")

    def detect(self, metadata, tensors):
        conf = super().detect(metadata, tensors)
        attn_hits = sum(1 for t in tensors if ".self_attn." in t.name)
        ssm_hits = sum(1 for t in tensors if ".ssm." in t.name
                       or ".mixer." in t.name)
        if attn_hits and ssm_hits:
            conf.score = max(conf.score, 0.90)
            conf.evidence.append(f"hybrid: attn={attn_hits}, ssm={ssm_hits}")
        return conf

    def build_graph(self, metadata, tensors):
        graph = super().build_graph(metadata, tensors)
        graph.arch = self.name
        graph.family = self.family
        graph.adapter_name = self.name
        return graph

    def kv_layout(self, graph):
        d = graph.dims
        return CacheLayout(
            kind="hybrid", block_tokens=16, page_bytes=4096,
            bytes_per_token=(2 * d.num_kv_heads * d.head_dim * 2
                             if d.num_kv_heads and d.head_dim else 0),
            layers=d.num_layers,
            kv_heads=d.num_kv_heads, head_dim=d.head_dim,
            extra={"ssm_layers": graph.dims.extra.get("ssm_layers", [])})


class UnknownAdapter(ArchAdapter):
    """Last-resort classifier.  Picks Q8 / FP16 only; refuses Q2.

    See plan section 3, "معمارية مجهولة → Q8/FP16 آمن أو رفض واضح".
    """
    name = "unknown"
    family = "unknown"
    priority = 0

    def detect(self, metadata, tensors):
        conf = Confidence(score=0.20, evidence=["no known arch signature matched"])
        conf.hard_constraints.append(
            "unknown architecture: only Q8 / FP16 / Q4_0 allowed; Q2 refused")
        return conf

    def build_graph(self, metadata, tensors):
        dims = ModelDims()
        # Best-effort guesses from common fields.
        dims.hidden_size = int(metadata.get("hidden_size", 0))
        dims.intermediate_size = int(metadata.get("intermediate_size", 0))
        dims.num_layers = int(metadata.get("num_hidden_layers", 0))
        dims.num_heads = int(metadata.get("num_attention_heads", 0))
        dims.num_kv_heads = int(metadata.get("num_key_value_heads",
                                             metadata.get("num_attention_heads", 0)))
        dims.head_dim = int(metadata.get("head_dim", 0))
        dims.vocab_size = int(metadata.get("vocab_size", 0))
        graph = ModelIR(arch="unknown", family="unknown", dims=dims,
                        adapter_name=self.name)
        graph.nodes = list(tensors)
        graph.confidence = Confidence(score=0.20,
                                      evidence=["unclassified architecture"])
        graph.validation.add(
            "warn", "", "arch.unknown",
            "architecture unknown; conservative plan only (Q8/FP16/Q4_0)")
        return graph


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass
class RegistryEntry:
    adapter: ArchAdapter
    score: float
    evidence: List[str]


class ArchRegistry:
    """Selects the highest-confidence :class:`ArchAdapter` for a model."""

    def __init__(self, adapters: Optional[Sequence[ArchAdapter]] = None) -> None:
        if adapters is None:
            adapters = [
                MoEAdapter(),
                DenseTransformerAdapter(),
                HybridSSMAdapter(),
                MambaAdapter(),
                UnknownAdapter(),                # always last
            ]
        # Order: highest priority first; UnknownAdapter last.
        self._adapters = sorted(adapters, key=lambda a: -a.priority)
        # Stable order: unknowns go to the back regardless of priority.
        self._adapters.sort(key=lambda a: a.name == "unknown")

    @property
    def adapters(self) -> Sequence[ArchAdapter]:
        return tuple(self._adapters)

    def detect(self, metadata: Dict[str, object],
               tensors: Sequence[TensorNode]) -> List[RegistryEntry]:
        """Return every adapter's confidence, sorted by score."""
        results: List[RegistryEntry] = []
        for adapter in self._adapters:
            try:
                c = adapter.detect(metadata, tensors)
            except Exception as exc:  # pragma: no cover - defensive
                c = Confidence(score=0.0, evidence=[f"detector raised {exc!r}"])
            results.append(RegistryEntry(adapter=adapter, score=c.score,
                                         evidence=list(c.evidence)))
        results.sort(key=lambda e: (-e.score, -e.adapter.priority))
        return results

    def select(self, metadata: Dict[str, object],
               tensors: Sequence[TensorNode]) -> Tuple[ArchAdapter, Confidence]:
        ranked = self.detect(metadata, tensors)
        top = ranked[0]
        # Pull the resolved dense mode (known/generic) out of the
        # detector's evidence chain so downstream ``build_graph`` can
        # stamp the correct arch_id.
        resolved_arch = top.adapter.name
        for line in top.evidence:
            if line.startswith("resolved_dense_mode="):
                resolved_arch = line.split("=", 1)[1]
                break
        # Stash on the instance so build_graph() reads it.
        try:
            top.adapter._resolved_arch = resolved_arch
        except Exception:
            pass
        conf = Confidence(
            score=top.score,
            evidence=list(top.evidence),
            hard_constraints=list(_aggregate_constraints(ranked)),
        )
        return top.adapter, conf


def _aggregate_constraints(ranked: Sequence[RegistryEntry]) -> List[str]:
    seen = set()
    out: List[str] = []
    for r in ranked:
        adapter = r.adapter
        # Re-run detect to recover hard_constraints; cached detect returned
        # evidence only for ranking, so we call again on the winner.  For
        # losers we synthesise no extra constraints.
        if r is ranked[0]:
            try:
                full = adapter.detect({}, [])  # cheap re-detect
            except Exception:
                full = Confidence(score=r.score, evidence=r.evidence)
            for c in full.hard_constraints:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
    return out


__all__ = [
    "ArchAdapter", "ArchRegistry", "RegistryEntry",
    "DenseTransformerAdapter", "MoEAdapter", "MambaAdapter",
    "HybridSSMAdapter", "UnknownAdapter",
]
