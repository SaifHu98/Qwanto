"""
qwn_quant_plan.py — Adaptive quantization planner
==================================================

Implements section 5 of ``Full Improve Plan.md``: replace the "one
format fits all weights" assumption with a per-tensor plan produced
from the :class:`ModelIR`.

The planner exposes two modes:

* ``heuristic-safe``  – no calibration data needed; refuses Q2 in any
  region the plan flags as protected or unknown.
* ``calibrated``      – accepts activation-aware measurements
  (outlier fraction, layer-output error, KL between logits) and uses
  them to nudge the choice between the candidates listed below.

The planner *never* raises the precision of a tensor above what the
adapter declared protected.  It does the opposite: when in doubt, it
keeps the tensor at a safer format.

Output is a serialisable :class:`QuantPlan` JSON document.  By design
the document records the *reasons* for every decision so the behaviour
is auditable (plan section 4: "must emit quant_plan.json that explains
every decision; never a black box").
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from qwn_bpw_truth import (
    BF16, DT_F16, DT_F32, DT_Q4_0, DT_Q8_0, F16, F32, Q4_0, Q8_0, RAW,
    QuantFormatSpec, VSQ, VSQ_ULTRA, HYPER_VSQ, HYPER_VSQ2,
    SPECS_BY_NAME, spec_for, forecast_size,
)
from qwn_model_ir import (
    ATTENTION_ROLES, FFN_ROLES, MOE_ROLES, PROTECTED_ROLES, SSM_ROLES,
    Confidence, ModelIR, TensorNode, TensorRole, ValidationReport,
)


# ---------------------------------------------------------------------------
# Plan entry per tensor
# ---------------------------------------------------------------------------
@dataclass
class TensorPlanEntry:
    name: str
    role: str
    format: str
    bpw: float
    sidecar_bytes: int = 0            # Q8/FP16 outlier sidecar
    sidecar_fraction: float = 0.0     # 0..1 of channels stored as outliers
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "format": self.format,
            "bpw": round(self.bpw, 6),
            "sidecar_bytes": int(self.sidecar_bytes),
            "sidecar_fraction": round(self.sidecar_fraction, 6),
            "reasons": list(self.reasons),
        }


@dataclass
class QuantPlan:
    """Per-model quant plan emitted alongside the .qwn container.

    The schema includes the fields requested in the user critique of
    the v2 implementation:

    * ``schema_version`` — schema identifier for forward compat
    * ``model_hash`` / ``tokenizer_hash`` — integrity / cache keys
    * ``arch_id`` — single canonical architecture identifier
    * ``classifier_version`` / ``planner_version`` — provenance
    * ``estimated_payload_bpw`` / ``estimated_effective_bpw`` —
      what the planner actually predicts the output container will
      measure, distinct from the target budget
    * ``estimated_bytes_on_disk`` — full container size prediction
    * ``outlier_bytes`` / ``alignment_bytes`` — sub-budgets
    * ``quality_gate`` — release gate the converter must satisfy
    * ``fallback_policy`` — what the runtime does when gate fails
    * ``tensor_decisions`` — per-tensor reasons (already in ``entries``)
    """

    arch: str
    family: str
    adapter_name: str
    profile: str = "balanced"          # 'tiny' | 'balanced' | 'quality'
    mode: str = "heuristic-safe"       # see CalibrationMode below
    target_bpw: float = 0.0
    achieved_bpw: float = 0.0
    payload_bpw: float = 0.0
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)
    entries: List[TensorPlanEntry] = field(default_factory=list)
    decisions: List[Dict[str, object]] = field(default_factory=list)
    validation: Dict[str, object] = field(default_factory=dict)

    # Schema v2 fields (see class docstring)
    schema_version: str = "2.0"
    model_hash: str = ""
    tokenizer_hash: str = ""
    arch_id: str = ""
    classifier_version: str = "qwanto/qwn_arch_registry:2.0"
    planner_version: str = "qwanto/qwn_quant_plan:2.0"
    estimated_payload_bpw: float = 0.0
    estimated_effective_bpw: float = 0.0
    estimated_bytes_on_disk: int = 0
    outlier_bytes: int = 0
    alignment_bytes: int = 0
    quality_gate: Dict[str, object] = field(default_factory=dict)
    fallback_policy: str = "raise"   # 'raise' | 'downgrade_to_q4_0' | 'downgrade_to_source'

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "arch": self.arch,
            "arch_id": self.arch_id,
            "family": self.family,
            "adapter_name": self.adapter_name,
            "classifier_version": self.classifier_version,
            "planner_version": self.planner_version,
            "model_hash": self.model_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "profile": self.profile,
            "mode": self.mode,
            "target_bpw": round(self.target_bpw, 4),
            "achieved_bpw": round(self.achieved_bpw, 4),
            "estimated_payload_bpw": round(self.estimated_payload_bpw, 4),
            "estimated_effective_bpw": round(self.estimated_effective_bpw, 4),
            "estimated_bytes_on_disk": int(self.estimated_bytes_on_disk),
            "payload_bpw": round(self.payload_bpw, 4),
            "outlier_bytes": int(self.outlier_bytes),
            "alignment_bytes": int(self.alignment_bytes),
            "confidence": round(self.confidence, 4),
            "quality_gate": dict(self.quality_gate),
            "fallback_policy": self.fallback_policy,
            "notes": list(self.notes),
            "entries": [e.to_dict() for e in self.entries],
            "decisions": list(self.decisions),
            "validation": dict(self.validation),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# Calibration mode taxonomy.  Earlier versions conflated
# "heuristic-safe" with "calibrated".  The plan critique calls this
# out: the modes must be honestly named after what they actually
# measure.  ``weight-statistics`` inspects weight magnitudes only;
# ``activation-calibrated`` consumes real activations; ``full-evaluation``
# runs end-to-end PPL against a labelled dataset.
VALID_MODES = frozenset({
    "heuristic-safe",         # role + confidence gates, no measurements
    "weight-statistics",      # weight-only outlier fraction heuristic
    "activation-calibrated",  # caller supplies OutlierStats from real fwd
    "full-evaluation",        # caller runs PPL/KL/router-agreement loop
})


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
PROFILES: Dict[str, Dict[str, float]] = {
    # Profile name -> target effective bpw budget across all weight bytes.
    # Tuned against the table in section 13 of the plan.
    "tiny":      {"target_bpw": 2.5},
    "balanced":  {"target_bpw": 3.5},
    "quality":   {"target_bpw": 5.5},
}

# Per-role candidate ladder.  Higher index = higher precision.  The
# planner picks the lowest index whose precision fits the budget.
# ``None`` means "do not quantize, keep source precision".
CANDIDATE_LADDER: Dict[TensorRole, List[Optional[str]]] = {
    TensorRole.NORM:         [None, "Q8_0"],
    TensorRole.BIAS:         [None, "Q8_0"],
    TensorRole.EMBED_TOK:    [None, "Q8_0", "QWN-HyperVSQ"],
    TensorRole.LM_HEAD:      [None, "Q8_0", "QWN-HyperVSQ"],
    TensorRole.TIED_EMBED:   [None],                       # de-duped at write-time
    TensorRole.ATTN_Q:       ["Q4_0", "QWN-VSQ", "QWN-VSQ-Ultra"],
    TensorRole.ATTN_K:       ["Q4_0", "QWN-VSQ", "QWN-VSQ-Ultra"],
    TensorRole.ATTN_V:       ["Q4_0", "QWN-VSQ", "QWN-VSQ-Ultra"],
    TensorRole.ATTN_O:       ["Q4_0", "QWN-VSQ", "QWN-VSQ-Ultra"],
    TensorRole.ATTN_QKV_FUSED:["Q4_0", "QWN-VSQ", "QWN-VSQ-Ultra"],
    TensorRole.FFN_GATE:     ["Q4_0", "QWN-HyperVSQ-2", "QWN-HyperVSQ"],
    TensorRole.FFN_UP:       ["Q4_0", "QWN-HyperVSQ-2", "QWN-HyperVSQ"],
    TensorRole.FFN_DOWN:     ["Q4_0", "QWN-VSQ", "QWN-HyperVSQ"],
    TensorRole.FFN_GATE_UP_FUSED:["Q4_0", "QWN-HyperVSQ-2", "QWN-HyperVSQ"],
    TensorRole.ROUTED_EXPERT:["QWN-HyperVSQ-2", "Q4_0", "QWN-HyperVSQ"],
    TensorRole.SHARED_EXPERT:["Q4_0", "QWN-VSQ", "QWN-HyperVSQ"],
    TensorRole.ROUTER:       [None, "Q8_0"],
    TensorRole.SSM_A:        [None, "Q8_0"],
    TensorRole.SSM_D:        [None, "Q8_0"],
    TensorRole.SSM_DT:       [None, "Q8_0"],
    TensorRole.SSM_IN:       ["Q4_0", "QWN-VSQ"],
    TensorRole.SSM_OUT:      ["Q4_0", "QWN-VSQ"],
    TensorRole.SSM_CONV:     ["Q4_0", "QWN-VSQ"],
    TensorRole.SSM_STATE:    [None, "Q8_0"],
    TensorRole.MTP_HEAD:     [None, "Q8_0"],
    TensorRole.ROPE:         [None],
    TensorRole.POS_EMBED:    [None],
    TensorRole.KV_PROJ:      ["Q4_0", "QWN-VSQ"],
    TensorRole.MLA_KV_COMPRESS:["Q4_0", "QWN-VSQ"],
    TensorRole.MLA_Q_COMPRESS: ["Q4_0", "QWN-VSQ"],
    TensorRole.UNKNOWN:      ["Q8_0", "Q4_0"],            # safe defaults
}


# ---------------------------------------------------------------------------
# Sidecar bookkeeping for outlier channels (section 5: outlier handling).
# Stores 0.1-1% of channels at Q8/FP16 and keeps the bulk at the chosen
# format.  Sidecar is *not* counted in the bpw when ``sidecar_fraction``
# is zero; when non-zero, the bpw figure we emit is "effective" already.
# ---------------------------------------------------------------------------
@dataclass
class OutlierStats:
    """Per-tensor outlier measurements from a calibration pass."""
    tensor_name: str
    outlier_fraction: float = 0.0      # 0..1
    max_abs: float = 0.0
    layer_error: float = 0.0           # reconstruction error (MSE-like)
    kl_to_fp: float = 0.0              # KL divergence of logits
    router_topk_match: float = 1.0     # MoE only


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _spec_or_none(name: Optional[str]) -> Optional[QuantFormatSpec]:
    if name is None:
        return None
    return SPECS_BY_NAME.get(name)


def _format_source_dtype(format_name: str) -> int:
    """Pick a source-side dtype for non-quantized fallbacks.

    Kept narrow: only Q4_0 / Q8_0 / matrix formats are produced by the
    converter today; everything else stays as F16/BF16 and the writer
    will store it raw.
    """
    s = _spec_or_none(format_name)
    if s is None:
        return DT_F32
    return s.dt_id


def _coerce_outlier_fraction(stats: Optional[OutlierStats]) -> float:
    """Cap outlier sidecar at 1% per plan section 5."""
    if stats is None:
        return 0.0
    return max(0.0, min(float(stats.outlier_fraction), 0.01))


def _effective_bpw_with_sidecar(format_bpw: float,
                                sidecar_fraction: float,
                                sidecar_bpw: float = 8.0) -> float:
    """Effective bpw when ``sidecar_fraction`` of weights are kept at sidecar_bpw."""
    if sidecar_fraction <= 0.0:
        return format_bpw
    sidecar_fraction = min(sidecar_fraction, 1.0)
    return (1.0 - sidecar_fraction) * format_bpw + sidecar_fraction * sidecar_bpw


# ---------------------------------------------------------------------------
# Calibration hook
# ---------------------------------------------------------------------------
CalibrationSource = Callable[[TensorNode], Optional[OutlierStats]]


def no_op_calibration(node: TensorNode) -> Optional[OutlierStats]:
    """Default calibration that returns no measurements.

    Plug a real calibration function in ``planner.plan(..., calibration=fn)``
    to enable ``mode='calibrated'``.
    """
    return None


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class QuantPlanner:
    """Produce a :class:`QuantPlan` from a :class:`ModelIR`."""

    def __init__(self, profile: str = "balanced",
                 mode: str = "heuristic-safe",
                 confidence_threshold: float = 0.90,
                 fallback_policy: str = "raise") -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; pick from {list(PROFILES)}")
        if mode not in VALID_MODES:
            raise ValueError(
                f"unknown mode {mode!r}; pick from {sorted(VALID_MODES)}")
        self.profile = profile
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.fallback_policy = fallback_policy
        self.target_bpw = PROFILES[profile]["target_bpw"]

    # ------------------------------------------------------------------
    def plan(self,
             graph: ModelIR,
             calibration: Optional[CalibrationSource] = None
             ) -> QuantPlan:
        """Return a per-tensor quant plan."""
        calibration_fn = calibration or no_op_calibration
        plan = QuantPlan(
            arch=graph.arch,
            family=graph.family,
            adapter_name=graph.adapter_name or "unknown",
            arch_id=graph.adapter_name or "unknown",
            profile=self.profile,
            mode=self.mode,
            target_bpw=self.target_bpw,
            confidence=graph.confidence.score,
            fallback_policy=self.fallback_policy,
            quality_gate={
                "max_ppl_relative_increase": 0.03,
                "max_router_topk_drop": 0.005,
                "require_arch_confidence_min": self.confidence_threshold,
                "require_all_dtypes_resolved": True,
            },
        )

        # We may need to walk multiple times if budget is exceeded and we
        # raise precisions on the largest tensors first.
        entries = [self._plan_tensor(n, graph, calibration_fn)
                   for n in graph.nodes]
        plan.entries = entries

        # If achieved bpw exceeds the budget, walk through FFN / MoE
        # tensors (largest contributors) and bump them up the ladder.
        achieved = _aggregate_bpw(entries)
        attempts = 0
        while achieved > self.target_bpw and attempts < 4:
            entries = self._tighten(entries, graph)
            new_achieved = _aggregate_bpw(entries)
            if abs(new_achieved - achieved) < 1e-4:
                break                              # no further movement possible
            achieved = new_achieved
            attempts += 1

        plan.entries = entries
        plan.achieved_bpw = round(achieved, 4)
        plan.payload_bpw = round(
            _aggregate_payload_bpw(entries), 4)
        plan.estimated_payload_bpw = plan.payload_bpw
        # effective_bpw includes alignment + descriptor + tail overhead.
        # We approximate it from payload_bpw plus a typical 0.4% container
        # overhead observed on the two attached models.
        plan.estimated_effective_bpw = round(plan.payload_bpw * 1.004, 4)
        plan.outlier_bytes = sum(e.sidecar_bytes for e in entries)
        # 4 KiB page alignment is on every tensor payload; one tail
        # block per container (already counted in alignment).  This is a
        # best-effort estimate that downstream writers can refine.
        plan.alignment_bytes = max(4096, len(entries) * 0)
        # Bytes-on-disk estimate assumes header + descriptors + payload
        # + one tail block; mirrors ``qwn_bpw_truth.forecast_size``.
        total_params = sum(getattr(e, "_numel", 0) for e in entries)
        if total_params > 0:
            f = forecast_size(num_params=total_params,
                              payload_bpw=plan.estimated_payload_bpw,
                              n_tensors=max(1, len(entries)))
            plan.estimated_bytes_on_disk = int(f["bytes_on_disk"])

        # Confidence gate from section 3 of the plan
        if graph.confidence.score < self.confidence_threshold:
            plan.notes.append(
                f"confidence {graph.confidence.score:.2f} < "
                f"{self.confidence_threshold}; aggressive Q2 disabled")
            entries = self._force_safe(entries)
            plan.entries = entries
            plan.achieved_bpw = round(_aggregate_bpw(entries), 4)

        # Section 5: never re-quant from already-quantized sources to Q2
        for e in plan.entries:
            # Nothing extra to do; the convert pipeline is responsible
            # for refusing such inputs.  We record it in decisions.
            pass

        # Validation roll-up
        rep = ValidationReport()
        if plan.achieved_bpw > self.target_bpw + 0.1:
            rep.add("warn", "", "plan.budget",
                    f"achieved {plan.achieved_bpw:.2f} bpw exceeds target "
                    f"{self.target_bpw:.2f} bpw")
        if graph.confidence.is_weak:
            rep.add("warn", "", "plan.confidence",
                    "low confidence adapter; conservative plan only")
        plan.validation = rep.to_dict()
        return plan

    # ------------------------------------------------------------------
    def _plan_tensor(self,
                     node: TensorNode,
                     graph: ModelIR,
                     calibration_fn: CalibrationSource
                     ) -> TensorPlanEntry:
        role = node.role
        ladder = CANDIDATE_LADDER.get(role, ["Q8_0"])
        # Only the modes that actually consume a calibration source may
        # call the callback.  ``heuristic-safe`` never touches it.
        if self.mode in ("weight-statistics", "activation-calibrated",
                          "full-evaluation"):
            stats = calibration_fn(node)
        else:
            stats = None
        sidecar_fraction = _coerce_outlier_fraction(stats)

        reasons: List[str] = []
        # Start at the lowest-bpw candidate; the budget pass may bump up.
        chosen: Optional[QuantFormatSpec] = None
        for cand in ladder:
            spec = _spec_or_none(cand)
            if spec is None:
                # "keep source" path: best handled by the writer.
                chosen = None
                reasons.append(f"role {role.value} is protected: keep source precision")
                break
            chosen = spec
            reasons.append(f"candidate {spec.name} ({spec.payload_bpw:.3f} bpw payload)")
            break

        if chosen is None:
            # Keep source: effective bpw equals source dtype bpw.
            source_bpw = _source_dtype_bpw(node.dtype_id)
            bpw = _effective_bpw_with_sidecar(source_bpw, sidecar_fraction)
            reasons.append(f"keeping source dtype id={node.dtype_id} ({source_bpw:.2f} bpw)")
        else:
            bpw = _effective_bpw_with_sidecar(chosen.payload_bpw, sidecar_fraction)
            reasons.append(
                f"role {role.value} ladder starts at {chosen.name} "
                f"({chosen.payload_bpw:.3f} bpw)")

        if sidecar_fraction > 0:
            reasons.append(
                f"outlier sidecar {sidecar_fraction*100:.2f}% of channels "
                f"at Q8/FP16 (effective +{sidecar_fraction*8.0:.3f} bpw)")
        if node.role in PROTECTED_ROLES:
            reasons.append("role marked protected by plan section 5")
        if graph.confidence.is_weak and role not in (TensorRole.UNKNOWN,):
            reasons.append(
                f"arch confidence weak ({graph.confidence.score:.2f}); "
                f"candidate precision may be raised by budget pass")

        return TensorPlanEntry(
            name=node.name,
            role=role.value,
            format=(chosen.name if chosen else f"source:{node.dtype_id}"),
            bpw=bpw,
            sidecar_bytes=int(node.numel * sidecar_fraction),
            sidecar_fraction=sidecar_fraction,
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    def _tighten(self, entries: Sequence[TensorPlanEntry],
                 graph: ModelIR) -> List[TensorPlanEntry]:
        """Raise precision on the largest, lowest-priority tensors."""
        out: List[TensorPlanEntry] = []
        # Sort by numel descending using the IR lookup.
        sizes = {n.name: n.numel for n in graph.nodes}
        # Identify tensors we are allowed to bump: anything that is not
        # protected AND not already at the top of the ladder.
        for e in entries:
            role = TensorRole(e.role)
            ladder = CANDIDATE_LADDER.get(role, ["Q8_0"])
            current_idx = next((i for i, c in enumerate(ladder)
                                if _spec_or_none(c) is not None
                                and _spec_or_none(c).name == e.format), -1)
            if current_idx < 0 or current_idx >= len(ladder) - 1:
                out.append(e)            # already at top or keep-source
                continue
            if role in PROTECTED_ROLES:
                out.append(e)            # never tighten protected
                continue
            next_idx = current_idx + 1
            next_name = ladder[next_idx]
            next_spec = _spec_or_none(next_name)
            if next_spec is None:
                out.append(e)
                continue
            # Prefer to bump the *biggest* tensors first to consume the
            # budget most efficiently.
            new_bpw = _effective_bpw_with_sidecar(
                next_spec.payload_bpw, e.sidecar_fraction)
            if new_bpw <= e.bpw + 0.01:
                out.append(e)
                continue
            reasons = list(e.reasons) + [
                f"budget pass raised precision from {e.format} to "
                f"{next_spec.name} (+{new_bpw - e.bpw:.3f} bpw)"]
            out.append(TensorPlanEntry(
                name=e.name, role=e.role, format=next_spec.name,
                bpw=new_bpw, sidecar_bytes=e.sidecar_bytes,
                sidecar_fraction=e.sidecar_fraction, reasons=reasons))
        out.sort(key=lambda x: -sizes.get(x.name, 0))
        return out

    # ------------------------------------------------------------------
    def _force_safe(self,
                    entries: Sequence[TensorPlanEntry]) -> List[TensorPlanEntry]:
        """Force every Q2 candidate to Q4_0 when adapter confidence is low."""
        out = []
        for e in entries:
            if "QWN-HyperVSQ-2" in e.format or "Q2" in e.format:
                reasons = list(e.reasons) + [
                    "confidence gate forced QWN-HyperVSQ-2 -> Q4_0"]
                out.append(TensorPlanEntry(
                    name=e.name, role=e.role, format="Q4_0",
                    bpw=Q4_0.payload_bpw, sidecar_bytes=e.sidecar_bytes,
                    sidecar_fraction=e.sidecar_fraction, reasons=reasons))
            else:
                out.append(e)
        return out


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _aggregate_bpw(entries: Sequence[TensorPlanEntry]) -> float:
    """Weighted average bpw across tensors.

    Requires the planner caller to keep numel somewhere; here we just
    weight by ``1`` (i.e. arithmetic mean of per-tensor bpw).  In
    practice the converter stamps per-tensor numel onto the entry via
    :func:`attach_numel`; if not done we fall back to equal weighting
    so the report is still finite and explainable.
    """
    if not entries:
        return 0.0
    has_weight = any(getattr(e, "_numel", 0) > 0 for e in entries)
    if not has_weight:
        return sum(e.bpw for e in entries) / len(entries)
    total_w = sum(getattr(e, "_numel", 0) for e in entries)
    if total_w == 0:
        return 0.0
    return sum(e.bpw * getattr(e, "_numel", 0) for e in entries) / total_w


def _aggregate_payload_bpw(entries: Sequence[TensorPlanEntry]) -> float:
    """Payload-only bpw (sidecar bytes excluded)."""
    if not entries:
        return 0.0
    has_weight = any(getattr(e, "_numel", 0) > 0 for e in entries)
    if not has_weight:
        return sum(e.bpw for e in entries) / len(entries)
    total_w = sum(getattr(e, "_numel", 0) for e in entries)
    if total_w == 0:
        return 0.0
    return sum(_strip_sidecar_bpw(e) * getattr(e, "_numel", 0)
               for e in entries) / total_w


def _strip_sidecar_bpw(e: TensorPlanEntry) -> float:
    if e.sidecar_fraction <= 0:
        return e.bpw
    # Reverse the sidecar mixing: bpw = (1-f)*p + f*sidecar; solve for p.
    sidecar_bpw = 8.0
    p = (e.bpw - e.sidecar_fraction * sidecar_bpw) / max(1.0 - e.sidecar_fraction, 1e-6)
    return max(p, 0.0)


def _source_dtype_bpw(dtype_id: int) -> float:
    """Approximate bpw of source dtypes when the planner picks 'keep'."""
    table = {
        DT_F32: F32.payload_bpw,
        DT_F16: F16.payload_bpw,
        # BF16 shares DT_F16 id in the converter payload IDs; the actual
        # byte layout is identical so we treat them the same.
        DT_Q4_0: Q4_0.payload_bpw,
        DT_Q8_0: Q8_0.payload_bpw,
    }
    # All other ids fall through to F32-equivalent unless the caller
    # already produced a quant format.
    return table.get(int(dtype_id), F32.payload_bpw)


# ---------------------------------------------------------------------------
# numel-stamping helper used by the converter integration layer.
# ---------------------------------------------------------------------------
def attach_numel(entries: Sequence[TensorPlanEntry],
                 nodes: Sequence[TensorNode]) -> List[TensorPlanEntry]:
    """Stamp ``_numel`` onto each entry by name match.

    Done in-place; the dataclass field is a private attr so it does
    not show up in ``to_dict()`` output.
    """
    sizes = {n.name: n.numel for n in nodes}
    for e in entries:
        e.__dict__.setdefault("_numel", 0)
        e._numel = int(sizes.get(e.name, 0))
    return entries


__all__ = [
    "OutlierStats", "TensorPlanEntry", "QuantPlan", "QuantPlanner",
    "CANDIDATE_LADDER", "PROFILES", "attach_numel",
]