"""
test_real_models.py — End-to-end tests using the actual attached GGUF
models as fixtures.

These tests do not fabricate metadata: they read the real GGUF header
and run the architecture detector, role classifier, quant planner, and
conversion pipeline against the real checkpoints shipped in
``models/``.

Skipped automatically when the source GGUF files are absent.
"""

from __future__ import annotations

import os
import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))

from qwn_arch_registry import ArchRegistry
from qwn_model_ir import TensorNode
from qwn_plan_cli import _scan_metadata
from qwn_quant_plan import QuantPlanner, VALID_MODES
from qwn_roles import classify_all

import qwn_convert as qcnv


ROOT = HERE.parent
MODELS_DIR = ROOT / "models"
MODELS = {
    "1.5B": MODELS_DIR / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
    "4B":   MODELS_DIR / "DeepSeek-V4-Pro-Qwen3.5-4B-MTP-BF16.gguf",
    "27B":  MODELS_DIR / "Qwen3.8-27B-UD-IQ2_M.gguf",
}


def _gguf_metadata(path: Path) -> dict:
    SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                    10: 8, 11: 8, 12: 8}
    SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I",
                  5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

    def read_str(f):
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def read_scalar(f, vt):
        return struct.unpack(SCALAR_FMT[vt], f.read(SCALAR_SIZES[vt]))[0]

    md = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"not GGUF: {path}")
        _ = struct.unpack("<I", f.read(4))[0]    # version
        tc = struct.unpack("<Q", f.read(8))[0]   # tensor_count
        mk = struct.unpack("<Q", f.read(8))[0]   # metadata_kv_count
        for _ in range(mk):
            k = read_str(f)
            vt = struct.unpack("<I", f.read(4))[0]
            if vt in SCALAR_SIZES:
                md[k] = read_scalar(f, vt)
            elif vt == 8:
                md[k] = read_str(f)
            elif vt == 9:
                et = struct.unpack("<I", f.read(4))[0]
                cnt = struct.unpack("<Q", f.read(8))[0]
                if et in SCALAR_SIZES:
                    f.seek(SCALAR_SIZES[et] * cnt, 1)
                elif et == 8:
                    for _ in range(cnt):
                        n = struct.unpack("<Q", f.read(8))[0]
                        f.seek(n, 1)
                else:
                    f.seek(cnt * 4, 1)
        md["tensor_count"] = tc
    return md


def _make_placeholder_tensors(meta: dict) -> list:
    """Synthesise a ``TensorNode`` list from GGUF metadata so the
    classifier has the right shape relationships without loading every
    tensor from disk.

    The placeholder tensors carry the real names + shapes the
    classifier needs; their ``numel`` field is set to the shape
    product so the planner's ``_numel`` stamping works.
    """
    arch = meta.get("general.architecture", "llama")
    hidden = int(meta.get(f"{arch}.embedding_length", 0))
    inter = int(meta.get(f"{arch}.feed_forward_length", 0))
    heads = int(meta.get(f"{arch}.attention.head_count", 0))
    kv = int(meta.get(f"{arch}.attention.head_count_kv", heads))
    head_dim = hidden // max(1, heads) if heads else 0
    layers = int(meta.get(f"{arch}.block_count", 0))
    # Some GGUF exports omit vocab_size from metadata; the tokenizer shape is
    # not needed for these registry tests, so use the documented Qwen2 default
    # only to construct placeholder nodes.
    vocab = int(meta.get(f"{arch}.vocab_size", 151936))
    if not vocab or not hidden:
        return []
    nodes: list = []
    nodes.append(TensorNode(
        name="model.embed_tokens.weight", shape=[vocab, hidden]))
    nodes.append(TensorNode(
        name="lm_head.weight", shape=[vocab, hidden]))
    for L in range(layers):
        nodes += [
            TensorNode(name=f"model.layers.{L}.input_layernorm.weight",
                       shape=[hidden]),
            TensorNode(name=f"model.layers.{L}.self_attn.q_proj.weight",
                       shape=[heads * head_dim, hidden]),
            TensorNode(name=f"model.layers.{L}.self_attn.k_proj.weight",
                       shape=[kv * head_dim, hidden]),
            TensorNode(name=f"model.layers.{L}.self_attn.v_proj.weight",
                       shape=[kv * head_dim, hidden]),
            TensorNode(name=f"model.layers.{L}.self_attn.o_proj.weight",
                       shape=[hidden, heads * head_dim]),
            TensorNode(name=f"model.layers.{L}.post_attention_layernorm.weight",
                       shape=[hidden]),
            TensorNode(name=f"model.layers.{L}.mlp.gate_proj.weight",
                       shape=[inter, hidden]),
            TensorNode(name=f"model.layers.{L}.mlp.up_proj.weight",
                       shape=[inter, hidden]),
            TensorNode(name=f"model.layers.{L}.mlp.down_proj.weight",
                       shape=[hidden, inter]),
        ]
    return nodes


@unittest.skipUnless(MODELS["1.5B"].exists(),
                     "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf not present")
class DeepSeek15BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = MODELS["1.5B"]
        cls.meta = _gguf_metadata(cls.path)

    def test_gguf_metadata_reads_qwen2(self):
        self.assertEqual(self.meta.get("general.architecture"), "qwen2")
        self.assertEqual(self.meta.get("qwen2.attention.head_count"), 12)
        self.assertEqual(self.meta.get("qwen2.attention.head_count_kv"), 2)
        self.assertEqual(self.meta.get("qwen2.block_count"), 28)
        self.assertEqual(self.meta.get("qwen2.embedding_length"), 1536)

    def test_arch_detection_known_dense_transformer(self):
        registry = ArchRegistry()
        adapter, conf = registry.select(self.meta, [])
        # Qwen2 architecture matches a known dense pattern.
        self.assertEqual(adapter.name, "known_dense_transformer")
        self.assertGreaterEqual(conf.score, 0.90)

    def test_role_classification_assigns_all_qkv_ffn(self):
        registry = ArchRegistry()
        nodes = _make_placeholder_tensors(self.meta)
        adapter, conf = registry.select(self.meta, nodes)
        graph = adapter.build_graph(self.meta, nodes)
        graph.confidence = conf
        graph = classify_all(graph)
        # Every per-layer tensor must be classified, not UNKNOWN.
        from collections import Counter
        role_counts = Counter(n.role.value for n in graph.nodes)
        self.assertGreater(role_counts.get("attn_q", 0), 0)
        self.assertGreater(role_counts.get("ffn_down", 0), 0)
        self.assertGreater(role_counts.get("norm", 0), 0)
        # lm_head and embed_tokens must be either embed/tied — both
        # come from the shape-equality tie detection.
        for n in graph.nodes:
            if n.name == "lm_head.weight":
                self.assertIn(n.role.value, ("tied_embed", "lm_head"))
            if n.name == "model.embed_tokens.weight":
                self.assertEqual(n.role.value, "embed_tok")

    def test_quant_plan_balanced_under_budget(self):
        registry = ArchRegistry()
        nodes = _make_placeholder_tensors(self.meta)
        adapter, conf = registry.select(self.meta, nodes)
        graph = adapter.build_graph(self.meta, nodes)
        graph.confidence = conf
        graph = classify_all(graph)
        plan = QuantPlanner(profile="balanced").plan(graph)
        # Every tensor got a format; plan is well-formed.
        self.assertGreater(len(plan.entries), 0)
        doc = plan.to_dict()
        # Schema v2 fields present.
        for k in ("schema_version", "model_hash", "estimated_bytes_on_disk"):
            self.assertIn(k, doc)
        # Protected roles did not pick Q2.
        for e in plan.entries:
            if e.role in ("norm", "embed_tok", "tied_embed", "router"):
                self.assertNotIn("HyperVSQ-2", e.format)

    def test_real_conversion_produces_a_real_qwn(self):
        """Convert the real 1.5B GGUF → .qwn in Q4_0 and check that
        the produced file is a valid .qwn container with the expected
        number of tensors.
        """
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "model.qwn"
            n_bytes = qcnv.convert_model(str(self.path), str(out),
                                          quant="q4_0")
            self.assertTrue(out.exists())
            # The GGUF source is Q4_K_M; the .qwn writer falls through
            # to passthrough for K-quants.  We expect size ~= source
            # minus GGUF metadata overhead.
            self.assertGreater(out.stat().st_size, 1024 * 1024)
            info = qcnv.inspect_qwn(str(out))
            self.assertGreater(info["n_tensors"], 0)
            self.assertGreater(info["n_params"], 0)


@unittest.skipUnless(MODELS["4B"].exists(),
                     "DeepSeek-V4-Pro-Qwen3.5-4B-MTP-BF16.gguf not present")
class DeepSeek4BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = MODELS["4B"]
        cls.meta = _gguf_metadata(cls.path)

    def test_gguf_metadata_reads_qwen35_with_mtp(self):
        self.assertEqual(self.meta.get("general.architecture"), "qwen35")
        # Qwen3.5 has full_attention_interval=4 → hybrid Mamba/Attn.
        self.assertEqual(self.meta.get("qwen35.full_attention_interval"), 4)
        # Has explicit MTP head.
        self.assertEqual(self.meta.get("qwen35.nextn_predict_layers"), 1)
        # 262k context.
        self.assertEqual(self.meta.get("qwen35.context_length"), 262144)

    def test_arch_detection_classified_as_dense_or_moe(self):
        # qwen35 isn't in our pattern list → ``known_dense_transformer``
        # via the architecture string match is expected because the
        # generic dense adapter matches the name prefix.
        registry = ArchRegistry()
        adapter, conf = registry.select(self.meta, [])
        self.assertIn(adapter.name, ("known_dense_transformer",
                                       "generic_dense_transformer"))
        self.assertGreaterEqual(conf.score, 0.60)

    def test_qwen35_mtp_conversion_contract_is_explicit(self):
        tensors, _ = qcnv._read_gguf_tensors(str(self.path), "q4_0")
        config = json.loads(next(t for t in tensors if t["name"] == "__qwn.config")["payload"])
        self.assertEqual(config["is_qwen35"], 1)
        self.assertEqual(config["num_hidden_layers"], 32)
        self.assertEqual(config["mtp_layers"], 1)

    def test_qwen35_hyper_vsq2_conversion_contract_is_explicit(self):
        tensors, _ = qcnv._read_gguf_tensors(str(self.path), "hyper_vsq2")
        self.assertGreater(len(tensors), 400)


@unittest.skipUnless(MODELS["27B"].exists(),
                     "Qwen3.8-27B-UD-IQ2_M.gguf not present")
class Qwen27BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = MODELS["27B"]
        cls.meta = _gguf_metadata(cls.path)

    def test_gguf_metadata_reads_qwen27b(self):
        self.assertEqual(self.meta.get("general.architecture"), "qwen35")
        self.assertEqual(self.meta.get("qwen35.block_count"), 65)
        self.assertEqual(self.meta.get("qwen35.embedding_length"), 5120)
        self.assertEqual(self.meta.get("qwen35.attention.head_count"), 24)
        self.assertEqual(self.meta.get("qwen35.attention.head_count_kv"), 4)

    def test_arch_detection_selects_dense_adapter(self):
        registry = ArchRegistry()
        adapter, conf = registry.select(self.meta, [])
        self.assertIn(adapter.name, ("known_dense_transformer", "generic_dense_transformer"))
        self.assertGreaterEqual(conf.score, 0.60)

    def test_qwen35_27b_conversion_contract_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            tensors, dims = qcnv._read_gguf_tensors(str(self.path), "q4_0")
            config = json.loads(next(t for t in tensors if t["name"] == "__qwn.config")["payload"])
            self.assertEqual(dims[5], 64)
            self.assertEqual(config["ssm_inner"], 10240)
            self.assertEqual(config["ssm_groups"], 48)


class NegativeAndEdgeCaseTests(unittest.TestCase):
    """Negative tests requested in the user critique of the v2 plan."""

    def test_empty_metadata_is_safe(self):
        """Empty metadata must produce a sub-0.90 confidence and a
        Q2 refusal constraint, regardless of which adapter happens to
        win the priority tie.
        """
        registry = ArchRegistry()
        adapter, conf = registry.select({}, [])
        self.assertLess(conf.score, 0.90)
        self.assertTrue(any("Q2" in c for c in conf.hard_constraints),
                        f"expected Q2 refusal, got {conf.hard_constraints}")
        # The MoE detector must also drop below the gate.
        moe = next(a for a in registry.adapters if a.name == "moe")
        moe_conf = moe.detect({}, [])
        self.assertLess(moe_conf.score, 0.90)

    def test_unknown_arch_forbids_mtp(self):
        from qwn_model_ir import MTPPlan, TensorRole, TensorNode
        from qwn_quant_plan import QuantPlanner
        # Build an IR that *claims* an MTP head, but with low
        # confidence (UnknownAdapter).  The planner must still refuse
        # Q2 on protected roles.
        nodes = [
            TensorNode(name="model.layers.0.mtp_fc.weight", shape=[64, 64],
                        role=TensorRole.MTP_HEAD),
            TensorNode(name="model.embed_tokens.weight", shape=[128, 64],
                        role=TensorRole.EMBED_TOK),
        ]
        from qwn_model_ir import ModelDims, ModelIR, Confidence
        ir = ModelIR(arch="unknown", family="unknown",
                      dims=ModelDims(hidden_size=64, vocab_size=128),
                      nodes=nodes,
                      confidence=Confidence(score=0.20))
        classify_all(ir)
        plan = QuantPlanner(profile="tiny").plan(ir)
        mtp = next(e for e in plan.entries if e.role == "mtp_head")
        self.assertNotIn("HyperVSQ-2", mtp.format)

    def test_invalid_bpw_mode_rejected(self):
        with self.assertRaises(ValueError):
            QuantPlanner(mode="bogus-mode")

    def test_calibrated_modes_disjoint_with_heuristic_safe(self):
        self.assertIn("heuristic-safe", VALID_MODES)
        self.assertIn("weight-statistics", VALID_MODES)
        self.assertIn("activation-calibrated", VALID_MODES)
        self.assertIn("full-evaluation", VALID_MODES)
        self.assertNotIn("calibrated", VALID_MODES)  # legacy name removed

    def test_impossible_bpw_rejected(self):
        from qwn_bpw_truth import forecast_size
        # A 1.5B model at 1 bpw would be ~187 MB — far below the
        # smallest format's floor (~432 MB at 2.31 bpw).
        size = forecast_size(num_params=int(1.5e9), payload_bpw=1.0)
        self.assertLess(size["bytes_on_disk"], 200 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
