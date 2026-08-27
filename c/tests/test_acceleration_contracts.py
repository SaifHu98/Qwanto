"""Static contracts for the typed acceleration phases.

These tests do not turn source presence into runtime evidence. Device and
long-context measurements remain separate executable gates.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def test_qwen38_qualification_remains_fail_closed():
    report = json.loads(
        (REPO / "docs" / "qwen38-27b-evidence" / "qualification-summary.json")
        .read_text(encoding="utf-8")
    )
    assert report["decision"] == "UNSUPPORTED_QWEN38_ARCHITECTURE"
    assert not (REPO / "docs" / "qwen38-27b-evidence" / "converted.qwn").exists()


def test_typed_kv_contract_is_not_environment_only():
    config = (ROOT / "qwn_runtime_config.h").read_text(encoding="utf-8")
    turbo = (ROOT / "qwanto_turboquant.h").read_text(encoding="utf-8")
    decoder = (ROOT / "qwanto_decode.c").read_text(encoding="utf-8")
    assert "QWN_RUNTIME_KV_Q8" in config
    assert "QWN_RUNTIME_KV_TURBOQUANT_Q4" in config
    assert "QWN_KV_CACHE_ABI_VERSION 1u" in turbo
    assert 'getenv("QWN_TURBOQUANT")' not in decoder
    assert "kv_cache_mode_actual" in decoder
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")


def test_speculation_and_jetspec_are_fail_closed_without_fake_metrics():
    spec = (ROOT / "qwn_speculative.c").read_text(encoding="utf-8")
    jetspec = (ROOT / "qwanto_jetspec.c").read_text(encoding="utf-8")
    saguro = (ROOT / "qwanto_saguro.c").read_text(encoding="utf-8")
    assert "QWN_SPEC_REQUIRES_COMPATIBLE_DRAFT_MODEL" in spec
    assert "parent-token arithmetic" not in jetspec
    assert "return 3.6f" not in saguro
    assert "baseline_tok_per_sec" in saguro


def test_converter_capability_axes_are_machine_readable():
    matrix = json.loads(
        (REPO / "docs" / "converter-capability-matrix.json")
        .read_text(encoding="utf-8")
    )
    assert set(matrix["axes"]) == {
        "source_containers",
        "source_tensor_dtypes",
        "architectures",
        "runtime_operators",
    }
    assert matrix["axes"]["source_tensor_dtypes"]["IQ1"] == "unsupported"
    assert matrix["axes"]["source_tensor_dtypes"]["IQ2"] == "native-qwn-preserving-and-runtime-row-verified"
    assert matrix["axes"]["source_tensor_dtypes"]["IQ3"] == "native-qwn-preserving-and-runtime-row-verified"
    assert matrix["axes"]["architectures"]["qwen35_hybrid_deltanet_mtp"] == "cpu-main-path-integration-verified"
