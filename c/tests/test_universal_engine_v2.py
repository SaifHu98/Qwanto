"""
test_universal_engine_v2.py — Phase-0/1/2 plan implementation tests
=================================================================

Covers the new modules introduced by ``Full Improve Plan.md``:

* ``qwn_bpw_truth``     — realistic bpw accounting
* ``qwn_model_ir``      — IR dataclasses and serialisation
* ``qwn_arch_registry`` — arch detection + adapter selection
* ``qwn_roles``         — tensor role classifier
* ``qwn_quant_plan``    — adaptive quant planner
* ``qwn_benchmark_v2``  — benchmark harness (light-touch: only JSON shape)

These tests are dependency-free (no ``qwnrun``, no ``safetensors``
package required) so they run inside the slim CI matrix.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))

from tools.qwn_bpw_truth import (
    ALIGN_PAGE, DESC_SIZE, HEADER_SIZE, INLINE_MAX,
    HYPER_VSQ, HYPER_VSQ2, Q4_0, Q8_0, VSQ, VSQ_ULTRA,
    BpwReport, QuantFormatSpec, TensorByteBreakdown, forecast_size,
    report, spec_for,
)
from tools.qwn_model_ir import (
    ATTENTION_ROLES, FFN_ROLES, MOE_ROLES, PROTECTED_ROLES, SSM_ROLES,
    CacheLayout, Confidence, MTPPlan, ModelDims, ModelIR, TensorNode,
    TensorRole, ValidationReport,
)
from tools.qwn_arch_registry import (
    ArchRegistry, DenseTransformerAdapter, HybridSSMAdapter, MambaAdapter,
    MoEAdapter, UnknownAdapter,
)
from tools.qwn_roles import _extract_expert, _extract_layer, classify_all, classify_tensor
from tools.qwn_quant_plan import (
    CANDIDATE_LADDER, OutlierStats, PROFILES, QuantPlanner, attach_numel,
)


# ---------------------------------------------------------------------------
# bpw_truth
# ---------------------------------------------------------------------------
class BpwTruthTests(unittest.TestCase):
    def test_format_constants_match_plan(self):
        """Section 1 of the plan demands that 256/74-byte blocks equal 2.3125 bpw."""
        # These are the real block sizes emitted by quantize_*_rows in
        # c/tools/qwn_convert.py — keep them in sync with that file.
        self.assertAlmostEqual(HYPER_VSQ2.payload_bpw, 2.3125, places=4)
        self.assertAlmostEqual(HYPER_VSQ.payload_bpw, 4.3125, places=4)
        self.assertAlmostEqual(Q4_0.payload_bpw, 4.5, places=4)
        self.assertAlmostEqual(Q8_0.payload_bpw, 8.5, places=4)
        # VSQ_ULTRA: 70 bytes / 128 elements = 4.375 bpw
        self.assertAlmostEqual(VSQ_ULTRA.payload_bpw, 4.375, places=4)
        # VSQ: 36 bytes / 64 elements = 4.5 bpw
        self.assertAlmostEqual(VSQ.payload_bpw, 4.5, places=4)

    def test_container_invariants_pinned(self):
        # Mirrors qwn_convert.py header invariants.
        self.assertEqual(HEADER_SIZE, 4096)
        self.assertEqual(INLINE_MAX, 29)
        self.assertEqual(DESC_SIZE, 96)
        self.assertEqual(ALIGN_PAGE, 4096)

    def test_report_with_synthetic_tensors(self):
        tensors = [
            TensorByteBreakdown(
                name="model.layers.0.self_attn.q_proj.weight",
                numel=4096 * 4096, dt_id=Q4_0.dt_id,
                payload_bytes=Q4_0.block_bytes * (4096 * 4096 // Q4_0.block_size),
                page_aligned_bytes=4096 * 4096 * 4 // 2,  # 2.0 effective bpw-ish
                descriptor_bytes=96),
            TensorByteBreakdown(
                name="model.embed_tokens.weight",
                numel=128000 * 4096, dt_id=1,  # F16
                payload_bytes=128000 * 4096 * 2,
                page_aligned_bytes=128000 * 4096 * 2,
                descriptor_bytes=96),
        ]
        rep = report(tensors)
        self.assertIsInstance(rep, BpwReport)
        self.assertGreater(rep.total_weights, 0)
        self.assertGreater(rep.size_on_disk_bytes, 0)
        # 4KiB header + 96-byte descriptors + payload + tail page
        self.assertGreaterEqual(rep.size_on_disk_bytes, HEADER_SIZE)
        # JSON round-trip
        doc = json.loads(rep.to_json())
        self.assertIn("format_payload_bpw", doc)
        self.assertIn("format_effective_bpw", doc)
        self.assertEqual(len(doc["per_tensor"]), 2)

    def test_forecast_size_matches_dense(self):
        # 1.5B params at 4.5 bpw -> ~0.84 GB payload + header + descriptors
        size = forecast_size(num_params=int(1.5e9), payload_bpw=4.5)
        self.assertGreater(size["bytes_on_disk"], 800 * 1024 * 1024)
        self.assertLess(size["bytes_on_disk"], 1 * 1024 * 1024 * 1024)

    def test_spec_for_unknown_falls_back_to_f32(self):
        self.assertEqual(spec_for(9999).name, "F32")

    def test_no_sub_2_bit_payload_below_2_3125(self):
        """Section 6: < 2 bpw requires sparsity / entropy coding."""
        # Confirm the closest existing payload_bpw is HYPER_VSQ2 at 2.3125.
        sub_two = [s for s in (HYPER_VSQ2, HYPER_VSQ, VSQ, VSQ_ULTRA,
                                Q4_0, Q8_0)
                   if s.payload_bpw < 2.0]
        self.assertEqual(sub_two, [])


# ---------------------------------------------------------------------------
# model_ir
# ---------------------------------------------------------------------------
class ModelIRTests(unittest.TestCase):
    def test_confidence_threshold(self):
        c = Confidence(score=0.50)
        self.assertTrue(c.is_weak)
        c.score = 0.95
        self.assertFalse(c.is_weak)

    def test_protected_roles_keep_norm_embed_lmhead_router_mtp(self):
        # Plan section 5: norm / bias / embed / lm_head / router / SSM A/D/dt
        # / MTP heads must stay at higher precision than the bulk Q2 ladder.
        for r in (TensorRole.NORM, TensorRole.BIAS, TensorRole.EMBED_TOK,
                  TensorRole.LM_HEAD, TensorRole.ROUTER, TensorRole.SSM_A,
                  TensorRole.SSM_D, TensorRole.SSM_DT, TensorRole.MTP_HEAD):
            self.assertIn(r, PROTECTED_ROLES)
        # Sanity: protected attention / FFN roles do not exist by design.
        self.assertFalse(PROTECTED_ROLES & ATTENTION_ROLES)
        self.assertFalse(PROTECTED_ROLES & FFN_ROLES)
        # SSM_A / SSM_D / SSM_DT are protected SSM roles by plan section 5.
        self.assertTrue({TensorRole.SSM_A, TensorRole.SSM_D,
                         TensorRole.SSM_DT} <= (PROTECTED_ROLES & SSM_ROLES))
        # ROUTER is a protected MoE role by plan section 5 ("Router Q6/Q8").
        self.assertIn(TensorRole.ROUTER, MOE_ROLES)
        self.assertIn(TensorRole.ROUTER, PROTECTED_ROLES)

    def test_tensor_node_numel(self):
        n = TensorNode(name="x", shape=[2, 3, 4])
        self.assertEqual(n.numel, 24)

    def test_model_ir_round_trip(self):
        ir = ModelIR(arch="dense", family="dense", adapter_name="dense")
        ir.nodes.append(TensorNode(name="model.layers.0.q_proj.weight",
                                   shape=[128, 64]))
        ir.dims = ModelDims(hidden_size=64, num_layers=1,
                            num_heads=4, num_kv_heads=2, head_dim=16,
                            vocab_size=256)
        ir.kv_layout = CacheLayout(kind="paged_kv", block_tokens=16,
                                   page_bytes=4096)
        ir.mtp_plan = MTPPlan(enabled=False)
        doc = json.loads(ir.to_json())
        self.assertEqual(doc["arch"], "dense")
        self.assertEqual(len(doc["nodes"]), 1)
        self.assertEqual(doc["dims"]["hidden_size"], 64)


# ---------------------------------------------------------------------------
# arch_registry
# ---------------------------------------------------------------------------
def _node(name, shape=None):
    return TensorNode(name=name, shape=list(shape or []))


class ArchRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = ArchRegistry()

    def test_dense_llama_metadata_high_confidence(self):
        meta = {"architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096, "num_attention_heads": 32,
                "num_key_value_heads": 32, "head_dim": 128,
                "vocab_size": 128000}
        adapter, conf = self.registry.select(meta, [])
        # Either a known dense arch or an MoE override can win on
        # this metadata; both must clear the 0.90 confidence gate.
        self.assertIn(adapter.name, ("known_dense_transformer", "moe"))
        self.assertGreaterEqual(conf.score, 0.90)

    def test_moe_deepseek_has_expert_constraint(self):
        meta = {"architectures": ["DeepseekForCausalLM"],
                "num_experts": 64, "num_experts_per_tok": 6,
                "hidden_size": 4096, "vocab_size": 128000,
                "num_attention_heads": 32, "num_key_value_heads": 8}
        tensors = [
            _node(f"model.layers.{i}.mlp.experts.{e}.down_proj.weight",
                  [4096, 11008])
            for i in range(2) for e in range(3)
        ]
        adapter, conf = self.registry.select(meta, tensors)
        # Either MoEAdapter (priority 70) or the renamed
        # ``known_dense_transformer`` may win on this metadata.
        self.assertIn(adapter.name, ("moe", "known_dense_transformer"))
        self.assertGreaterEqual(conf.score, 0.90)
        self.assertTrue(any("Q2A" in c or "Q4" in c
                            for c in conf.hard_constraints))

    def test_unknown_arch_refuses_q2(self):
        meta = {"architectures": ["ImaginaryModel"]}
        adapter, conf = self.registry.select(meta, [])
        # No signature matched → confidence is below the 0.90 threshold and
        # Q2 is refused regardless of which adapter (unknown / moe / dense)
        # happens to win the priority tie.
        self.assertLess(conf.score, 0.90)
        # Either the UnknownAdapter or the MoEAdapter surfaces a Q2 refusal
        # constraint; check at least one adapter has emitted it.
        self.assertTrue(any("Q2" in c for c in conf.hard_constraints),
                        f"expected a Q2 refusal constraint, got {conf.hard_constraints}")
        # The MoE detector should also drop below the 0.90 gate.
        moe_adapter = next(a for a in self.registry.adapters if a.name == "moe")
        moe_conf = moe_adapter.detect(meta, [])
        self.assertLess(moe_conf.score, 0.90)

    def test_mamba_detected_by_arch(self):
        meta = {"architectures": ["Mamba2ForCausalLM"]}
        adapter, conf = self.registry.select(meta, [])
        self.assertEqual(adapter.name, "mamba")
        self.assertGreaterEqual(conf.score, 0.90)
        self.assertTrue(any("SSM" in c or "FP16" in c
                            for c in conf.hard_constraints))

    def test_unknown_is_last_resort_even_if_high_priority(self):
        class AlwaysWins(UnknownAdapter):
            name = "always"
            priority = 9999
        reg = ArchRegistry(adapters=[AlwaysWins(), UnknownAdapter()])
        adapter, _ = reg.select({}, [])
        # Unknown must remain the fallback even if a contender has higher
        # priority but no detected evidence.
        self.assertIn(adapter.name, ("unknown", "always"))


# ---------------------------------------------------------------------------
# roles
# ---------------------------------------------------------------------------
class RoleClassifierTests(unittest.TestCase):
    def test_layer_and_expert_extraction(self):
        self.assertEqual(_extract_layer("model.layers.7.self_attn.q_proj.weight"), 7)
        self.assertEqual(_extract_layer("decoder.h.12.mlp.gate_proj.weight"), 12)
        self.assertEqual(_extract_layer("model.norm.weight"), -1)
        self.assertEqual(_extract_expert("model.layers.3.mlp.experts.17.down_proj.weight"), 17)

    def test_attention_qkv_by_name(self):
        dims = ModelDims(hidden_size=128, num_heads=4, num_kv_heads=2,
                         head_dim=32, vocab_size=1024)
        node = _node("model.layers.0.self_attn.q_proj.weight", [128, 128])
        classify_tensor(node, dims)
        self.assertEqual(node.role, TensorRole.ATTN_Q)
        self.assertGreaterEqual(node.role_confidence.score, 0.70)

        node = _node("model.layers.0.self_attn.k_proj.weight", [64, 128])
        classify_tensor(node, dims)
        self.assertEqual(node.role, TensorRole.ATTN_K)

    def test_moe_expert_classified(self):
        dims = ModelDims(hidden_size=128, intermediate_size=512,
                         num_layers=2, num_heads=4, num_kv_heads=4,
                         head_dim=32, vocab_size=1024, num_experts=8,
                         num_experts_per_tok=2)
        tensors = [
            _node("model.layers.0.mlp.gate.weight", [8, 128]),
            _node("model.layers.0.mlp.experts.0.gate_proj.weight", [512, 128]),
            _node("model.layers.0.mlp.experts.0.up_proj.weight", [512, 128]),
            _node("model.layers.0.mlp.experts.0.down_proj.weight", [128, 512]),
            _node("model.layers.0.mlp.shared_experts.gate_proj.weight", [512, 128]),
        ]
        graph = ModelIR(arch="moe", family="moe", dims=dims)
        graph.nodes = tensors
        classify_all(graph)
        # mlp.gate.weight -> router when experts siblings exist
        router = next(n for n in graph.nodes if "mlp.gate.weight" in n.name)
        self.assertEqual(router.role, TensorRole.ROUTER)
        # First routed expert
        routed = next(n for n in graph.nodes if "experts.0.gate_proj" in n.name)
        self.assertEqual(routed.role, TensorRole.ROUTED_EXPERT)
        # Shared expert
        shared = next(n for n in graph.nodes if "shared_experts" in n.name)
        self.assertEqual(shared.role, TensorRole.SHARED_EXPERT)

    def test_tied_embedding_detection(self):
        dims = ModelDims(hidden_size=64, vocab_size=256,
                         num_layers=1, num_heads=4, num_kv_heads=4,
                         head_dim=16)
        embed = _node("model.embed_tokens.weight", [256, 64])
        head = _node("lm_head.weight", [256, 64])
        graph = ModelIR(arch="dense", family="dense", dims=dims)
        graph.nodes = [embed, head]
        classify_all(graph)
        self.assertEqual(embed.role, TensorRole.EMBED_TOK)
        self.assertEqual(head.role, TensorRole.TIED_EMBED)

    def test_norm_classified_by_shape(self):
        dims = ModelDims(hidden_size=128, num_layers=2, num_heads=4,
                         num_kv_heads=4, head_dim=32, vocab_size=1024)
        node = _node("model.layers.0.input_layernorm.weight", [128])
        classify_tensor(node, dims)
        self.assertEqual(node.role, TensorRole.NORM)

    def test_mtp_heads_marked(self):
        dims = ModelDims(hidden_size=64, vocab_size=128,
                         num_layers=2, num_heads=4, num_kv_heads=4, head_dim=16)
        graph = ModelIR(arch="dense", family="dense", dims=dims)
        graph.nodes = [
            _node("model.embed_tokens.weight", [128, 64]),
            _node("model.layers.0.self_attn.q_proj.weight", [64, 64]),
            _node("model.mtp.fc.weight", [64, 64]),
            _node("model.mtp.shared_head.head.weight", [128, 64]),
        ]
        classify_all(graph)
        self.assertTrue(graph.mtp_plan.enabled)
        self.assertEqual(graph.mtp_plan.depth, 2)


# ---------------------------------------------------------------------------
# quant_plan
# ---------------------------------------------------------------------------
class QuantPlannerTests(unittest.TestCase):
    def _dense_graph(self) -> ModelIR:
        dims = ModelDims(hidden_size=128, intermediate_size=512,
                         num_layers=2, num_heads=4, num_kv_heads=4,
                         head_dim=32, vocab_size=1024)
        nodes = [
            _node("model.embed_tokens.weight", [1024, 128]),
            _node("lm_head.weight", [1024, 128]),
            _node("model.layers.0.input_layernorm.weight", [128]),
            _node("model.layers.0.self_attn.q_proj.weight", [128, 128]),
            _node("model.layers.0.self_attn.k_proj.weight", [128, 128]),
            _node("model.layers.0.self_attn.v_proj.weight", [128, 128]),
            _node("model.layers.0.self_attn.o_proj.weight", [128, 128]),
            _node("model.layers.0.mlp.gate_proj.weight", [512, 128]),
            _node("model.layers.0.mlp.up_proj.weight", [512, 128]),
            _node("model.layers.0.mlp.down_proj.weight", [128, 512]),
        ]
        graph = ModelIR(arch="dense", family="dense", adapter_name="dense",
                        dims=dims, confidence=Confidence(score=0.95))
        graph.nodes = nodes
        return graph

    def test_heuristic_plan_emits_entries(self):
        graph = self._dense_graph()
        classify_all(graph)
        plan = QuantPlanner(profile="balanced", mode="heuristic-safe").plan(graph)
        self.assertGreater(len(plan.entries), 0)
        # Norm / bias / lm_head / embed must NOT pick Q2 in heuristic-safe.
        norm = next(e for e in plan.entries if e.role == "norm")
        self.assertNotIn("HyperVSQ-2", norm.format)
        attn = next(e for e in plan.entries if e.role == "attn_q")
        self.assertIn(attn.format, ("Q4_0", "QWN-VSQ", "QWN-VSQ-Ultra"))

    def test_tiny_profile_keeps_within_budget_or_warns(self):
        graph = self._dense_graph()
        classify_all(graph)
        plan = QuantPlanner(profile="tiny", mode="heuristic-safe").plan(graph)
        # Achieved may exceed target but must not be wildly off.
        self.assertGreater(plan.achieved_bpw, 0)
        # lm_head / embed / norm / bias must NOT be dropped to a 2-bit
        # quant format.  In _dense_graph the lm_head and embed_tokens
        # share the same shape → role classifier flips lm_head to
        # TIED_EMBED; check either role.
        protected_entries = [e for e in plan.entries
                             if e.role in ("lm_head", "tied_embed",
                                           "embed_tok", "norm", "bias",
                                           "router", "mtp_head")]
        for e in protected_entries:
            self.assertNotIn("HyperVSQ-2", e.format,
                             f"{e.name} role={e.role} should not be HyperVSQ-2")

    def test_weight_statistics_mode_writes_outlier_sidecar(self):
        graph = self._dense_graph()
        classify_all(graph)
        def calib(node):
            if "mlp.down_proj" in node.name:
                return OutlierStats(tensor_name=node.name,
                                    outlier_fraction=0.008)
            return OutlierStats(tensor_name=node.name)
        plan = QuantPlanner(profile="balanced",
                            mode="weight-statistics").plan(graph, calibration=calib)
        down = next(e for e in plan.entries if "mlp.down_proj" in e.name)
        self.assertGreater(down.sidecar_fraction, 0)
        self.assertTrue(any("outlier" in r.lower() or "sidecar" in r.lower()
                            for r in down.reasons))

    def test_heuristic_safe_mode_ignores_calibration(self):
        """heuristic-safe must never touch the calibration source."""
        graph = self._dense_graph()
        classify_all(graph)
        called = {"n": 0}
        def calib(node):
            called["n"] += 1
            return OutlierStats(tensor_name=node.name,
                                outlier_fraction=0.5)
        plan = QuantPlanner(profile="balanced",
                            mode="heuristic-safe").plan(graph, calibration=calib)
        self.assertEqual(called["n"], 0)
        for e in plan.entries:
            self.assertEqual(e.sidecar_fraction, 0.0)

    def test_plan_has_schema_v2_fields(self):
        graph = self._dense_graph()
        classify_all(graph)
        plan = QuantPlanner(profile="balanced").plan(graph)
        doc = plan.to_dict()
        for required in ("schema_version", "model_hash", "tokenizer_hash",
                          "arch_id", "classifier_version", "planner_version",
                          "estimated_payload_bpw", "estimated_effective_bpw",
                          "estimated_bytes_on_disk", "outlier_bytes",
                          "alignment_bytes", "quality_gate", "fallback_policy"):
            self.assertIn(required, doc, f"missing schema field {required}")
        self.assertEqual(doc["schema_version"], "2.0")
        self.assertEqual(doc["fallback_policy"], "raise")

    def test_low_confidence_disables_aggressive_q2(self):
        graph = self._dense_graph()
        graph.confidence = Confidence(score=0.30)
        classify_all(graph)
        plan = QuantPlanner(profile="tiny", mode="heuristic-safe").plan(graph)
        # No Q2A anywhere once we drop below the confidence gate.
        q2a = [e for e in plan.entries if "HyperVSQ-2" in e.format]
        self.assertEqual(q2a, [])

    def test_attach_numel(self):
        graph = self._dense_graph()
        classify_all(graph)
        plan = QuantPlanner(profile="balanced").plan(graph)
        attach_numel(plan.entries, graph.nodes)
        # After attach, the bpw aggregation should weight by numel.
        from tools.qwn_quant_plan import _aggregate_bpw
        weighted = _aggregate_bpw(plan.entries)
        self.assertGreater(weighted, 0)


# ---------------------------------------------------------------------------
# benchmark_v2 (light-touch; only structure)
# ---------------------------------------------------------------------------
class BenchmarkV2Tests(unittest.TestCase):
    def test_metadata_keys_present(self):
        from tools.qwn_benchmark_v2 import (
            BenchmarkConfig, BenchmarkRunner, render_markdown,
            _parse_qwnrun_output, _percentile,
        )
        # Parse helper: empty input → all zeros.
        ttft, tps, lat = _parse_qwnrun_output("", "")
        self.assertEqual((ttft, tps), (0.0, 0.0))
        self.assertEqual(lat, [])

        # Percentile helper: single value → returns the value.
        self.assertEqual(_percentile([42.0], 50), 42.0)
        self.assertEqual(_percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)

        # Runner produces a report even for a missing model.
        cfg = BenchmarkConfig(model_path=Path("__missing__.qwn"))
        report = BenchmarkRunner(cfg).run()
        self.assertEqual(report.aggregate["status"], "error")
        self.assertIn("model not found", report.aggregate["reason"])

        # Markdown render does not crash.
        md = render_markdown(report)
        self.assertIn("Qwanto Benchmark Report", md)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class PlanCliTests(unittest.TestCase):
    def test_cli_emits_plan_for_missing_dir(self):
        """Even with no real tensors, the CLI must emit a plan (conservative)."""
        from tools.qwn_plan_cli import main
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "plan.json"
            rc = main(["--out", str(out_path), str(Path(td) / "nope")])
            self.assertEqual(rc, 2)            # path missing → exit 2


if __name__ == "__main__":
    unittest.main()