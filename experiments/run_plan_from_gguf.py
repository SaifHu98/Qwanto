"""
run_plan_from_gguf.py — Generate a real quant_plan.json from the actual
GGUF tensor list (uses qwn_convert._read_gguf_tensors to enumerate).

This is the bridge between the converter and the planner.  It feeds the
real tensor names + shapes + dtypes into the architecture detector,
role classifier, and planner so the resulting plan reflects the actual
model rather than a synthetic placeholder.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "c" / "tools"))

import qwn_arch_registry as ar
import qwn_convert as qcnv
import qwn_model_ir as ir
import qwn_plan_cli as qpc
import qwn_quant_plan as qp
import qwn_roles as roles


def _enumerate_real_tensors(path: Path):
    """Read the GGUF tensor index; return ``(TensorNode list, arch_meta dict)``."""
    tensors, dims = qcnv._read_gguf_tensors(str(path), quant="q4_0")
    nodes: List[ir.TensorNode] = []
    # Minimal arch metadata for the registry: only the keys the
    # detector actually inspects.
    arch_meta: Dict[str, object] = {
        "general.architecture": "qwen2" if "1.5B" in path.name else "qwen35",
        "hidden_size": dims[0],
        "intermediate_size": dims[1],
        "num_attention_heads": dims[2],
        "num_key_value_heads": dims[3],
        "head_dim": dims[4],
        "num_hidden_layers": dims[5],
        "vocab_size": dims[6],
        "max_position_embeddings": dims[7],
    }
    for t in tensors:
        shp = list(t.get("shape", []))
        numel = 1
        for d in shp:
            numel *= int(d)
        nodes.append(ir.TensorNode(
            name=t["name"],
            shape=shp,
            dtype_id=int(t.get("dtype", 0)),
        ))
    return nodes, arch_meta, dims


def _model_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "models" / "DeepSeek-V4-Pro-Qwen3.5-4B-MTP-BF16.gguf")
    if not src.exists():
        print(f"error: {src} not found")
        return 2
    profile = sys.argv[2] if len(sys.argv) > 2 else "balanced"
    out = (Path(sys.argv[3]) if len(sys.argv) > 3
           else ROOT / "experiments" / "results" /
           f"plan_{src.stem[:8]}_{profile}.json")

    print(f"==> reading real tensor list from {src.name}")
    nodes, meta, dims = _enumerate_real_tensors(src)
    print(f"    {len(nodes)} tensors, dims={dims}")

    registry = ar.ArchRegistry()
    adapter, conf = registry.select(meta, nodes)
    graph = adapter.build_graph(meta, nodes)
    graph.confidence = conf
    graph.validation = adapter.validate_shapes(graph)
    graph = roles.classify_all(graph)

    planner = qp.QuantPlanner(profile=profile, mode="heuristic-safe")
    plan = planner.plan(graph)
    plan.arch_id = adapter.name
    plan.model_hash = _model_sha256(src)
    plan.fallback_policy = "raise"

    out.write_text(plan.to_json(), encoding="utf-8")
    print(f"wrote {out}")
    print(f"    arch={plan.arch} arch_id={plan.arch_id} "
           f"confidence={plan.confidence:.2f}")
    print(f"    target={plan.target_bpw:.2f} "
           f"achieved={plan.achieved_bpw:.2f} "
           f"estimated_effective={plan.estimated_effective_bpw:.2f} "
           f"entries={len(plan.entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())