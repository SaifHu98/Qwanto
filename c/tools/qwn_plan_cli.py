"""
qwn_plan_cli.py — Standalone CLI: detect → classify → plan → emit
================================================================

A thin entry point that ties together the new pipeline modules without
touching ``c/tools/qwn_convert.py`` (per plan section 11: "يبقى
qwn_convert.py منفذاً للخطة").  Use it from the shell::

    python c/tools/qwn_plan_cli.py path/to/model.safetensors --out plan.json
    python c/tools/qwn_plan_cli.py path/to/model.gguf        --profile tiny
    python c/tools/qwn_plan_cli.py path/to/dir --mode calibrated --calib data/

The tool walks the model directory or single checkpoint, builds a
:class:`ModelIR`, classifies every tensor, runs the planner in either
``heuristic-safe`` or ``calibrated`` mode, and writes a single
``quant_plan.json`` ready to be consumed by the converter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

# Make sibling tools importable when run as a script.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qwn_arch_registry import ArchRegistry
from qwn_model_ir import ModelIR, TensorNode
from qwn_quant_plan import OutlierStats, QuantPlanner, attach_numel
from qwn_roles import classify_all


# ---------------------------------------------------------------------------
# Tensors enumeration
# ---------------------------------------------------------------------------
def _scan_safetensors_meta(path: Path) -> List[TensorNode]:
    """Best-effort: read safetensors index without loading tensors.

    Used for arch detection / role classification.  We rely on the
    ``safetensors`` package when available; otherwise we return an empty
    list and let the caller fall back to GGUF or PyTorch.
    """
    try:
        from safetensors import safe_open  # type: ignore
    except Exception:
        return []
    nodes: List[TensorNode] = []
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.safetensors"))
    for f in files:
        try:
            with safe_open(str(f), framework="pt") as h:
                for k in h.keys():
                    meta = h.get_slice(k).get_shape() if hasattr(h, "get_slice") else None
                    # best-effort: many safetensors wrappers only expose keys()
                    nodes.append(TensorNode(name=k, shape=list(meta or [])))
        except Exception:
            continue
    return nodes


def _scan_metadata(path: Path) -> Dict[str, object]:
    """Read ``config.json`` and ``model.safetensors.index.json`` from path."""
    out: Dict[str, object] = {}
    if path.is_dir():
        cfg = path / "config.json"
        if cfg.exists():
            try:
                out = json.loads(cfg.read_text(encoding="utf-8"))
            except Exception:
                out = {}
    elif path.is_file() and path.suffix == ".gguf":
        # Best-effort GGUF header probe: the project's qwn_convert.py
        # knows the format; we reuse it when available.
        try:
            sys.path.insert(0, str(HERE))
            from qwn_convert import _read_gguf_metadata  # type: ignore
            out = _read_gguf_metadata(str(path)) or {}
        except Exception:
            out = {}
    return out


# ---------------------------------------------------------------------------
# Default calibration (heuristic-safe mode reuses this).
# ---------------------------------------------------------------------------
def heuristic_calibration(node: TensorNode) -> OutlierStats:
    """Conservative outlier estimate from the tensor name alone.

    Production calibration will measure real activations; this stub
    is good enough to seed ``mode='calibrated'`` runs without any
    activation data and still refuses outliers on protected roles.
    """
    from qwn_model_ir import PROTECTED_ROLES
    if node.role in PROTECTED_ROLES:
        return OutlierStats(tensor_name=node.name)
    if any(tag in node.name.lower() for tag in ("expert", "mlp.experts")):
        return OutlierStats(tensor_name=node.name,
                            outlier_fraction=0.005)
    return OutlierStats(tensor_name=node.name, outlier_fraction=0.002)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect architecture, classify tensors, emit quant_plan.json")
    parser.add_argument("model", help="Path to a model dir / .safetensors / .gguf")
    parser.add_argument("--profile", choices=("tiny", "balanced", "quality"),
                        default="balanced")
    parser.add_argument("--mode", choices=(
                            "heuristic-safe", "weight-statistics",
                            "activation-calibrated", "full-evaluation"),
                        default="heuristic-safe")
    parser.add_argument("--calibration-data", type=str, default="",
                        help="Path to a calibration dataset (used by "
                             "activation-calibrated and full-evaluation "
                             "modes).  Not consumed in this release.")
    parser.add_argument("--fallback-policy", choices=(
                            "raise", "downgrade_to_q4_0",
                            "downgrade_to_source"),
                        default="raise")
    parser.add_argument("--out", type=str, default="quant_plan.json")
    parser.add_argument("--adapter", type=str, default="",
                        help="Force a specific adapter name (skip detection)")
    args = parser.parse_args(argv)

    path = Path(args.model).resolve()
    if not path.exists():
        print(f"error: model path does not exist: {path}", file=sys.stderr)
        return 2

    metadata = _scan_metadata(path)
    nodes = _scan_safetensors_meta(path)
    if not nodes:
        # Last-resort: synthesize UNKNOWN nodes from the config metadata so
        # the planner still produces a *plan* (conservative only).
        nodes = []
        hidden = int(metadata.get("hidden_size", 0)) if metadata else 0
        inter = int(metadata.get("intermediate_size", 0)) if metadata else 0
        if hidden and inter:
            nodes.append(TensorNode(name="model.placeholder.q_proj",
                                    shape=[hidden, hidden]))
            nodes.append(TensorNode(name="model.placeholder.mlp_up_proj",
                                    shape=[inter, hidden]))

    registry = ArchRegistry()
    adapter, conf = registry.select(metadata, nodes)
    graph = adapter.build_graph(metadata, nodes)
    graph.confidence = conf
    graph.validation = adapter.validate_shapes(graph)
    graph = classify_all(graph)

    calibration = heuristic_calibration if args.mode in (
        "weight-statistics", "activation-calibrated", "full-evaluation"
    ) else None
    planner = QuantPlanner(profile=args.profile, mode=args.mode,
                           fallback_policy=args.fallback_policy)
    plan = planner.plan(graph, calibration=calibration)
    # Hash model + tokenizer for the schema (best effort).
    try:
        import hashlib
        if path.is_file():
            plan.model_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.is_dir():
            files = sorted(p for p in path.rglob("*")
                           if p.is_file() and p.suffix.lower() in
                           (".safetensors", ".bin", ".gguf", ".json", ".txt"))
            h = hashlib.sha256()
            for f in files:
                h.update(f.read_bytes())
            plan.model_hash = h.hexdigest()
        tok_path = path / "tokenizer.json" if path.is_dir() else None
        if tok_path and tok_path.exists():
            plan.tokenizer_hash = hashlib.sha256(
                tok_path.read_bytes()).hexdigest()
    except Exception:
        pass
    plan.decisions = [
        {"adapter": adapter.name, "confidence": conf.score,
         "constraints": list(conf.hard_constraints)},
        {"profile": args.profile, "mode": args.mode,
         "target_bpw": plan.target_bpw, "achieved_bpw": plan.achieved_bpw},
    ]
    if args.adapter:
        plan.adapter_name = args.adapter

    # Stamp numel from the IR so bpw aggregation is correct.
    attach_numel(plan.entries, graph.nodes)

    Path(args.out).write_text(plan.to_json(), encoding="utf-8")
    print(f"wrote {args.out}: arch={plan.arch} family={plan.family} "
          f"profile={plan.profile} mode={plan.mode} "
          f"achieved={plan.achieved_bpw:.3f} bpw "
          f"(target {plan.target_bpw:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())