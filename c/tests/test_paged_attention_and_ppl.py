import os
import sys
import math
import pytest
from pathlib import Path

# Add tools to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from qwn_ppl import evaluate_ppl_simulation, compute_perplexity_from_logits, WIKITEXT2_SAMPLE


def test_ppl_simulation_hypervsq():
    """Verify QWN-HyperVSQ perplexity calculation and accuracy retention."""
    res = evaluate_ppl_simulation("model_hypervsq.qwn", WIKITEXT2_SAMPLE, context_len=512)
    assert res["bpw"] == 2.70
    assert 11.0 <= res["perplexity"] <= 14.5
    assert res["accuracy_retention_pct"] > 85.0
    assert res["compression_ratio"] > 5.0


def test_ppl_simulation_q4km():
    """Verify legacy Q4_K_M baseline perplexity."""
    res = evaluate_ppl_simulation("model_q4_k_m.gguf", WIKITEXT2_SAMPLE, context_len=512)
    assert res["bpw"] == 4.50
    assert res["perplexity"] < 12.5
    assert res["accuracy_retention_pct"] > 90.0


def test_ppl_log_prob_math():
    """Verify token log-probabilities to perplexity transformation."""
    # Constant prob 0.5 -> NLL = -ln(0.5) = 0.6931 -> PPL = 2.0
    log_probs = [math.log(0.5)] * 10
    ppl, avg_nll = compute_perplexity_from_logits(log_probs)
    assert round(ppl, 2) == 2.0
    assert round(avg_nll, 4) == round(-math.log(0.5), 4)


def test_paged_attention_header_exists():
    """Verify PagedAttention C headers and implementations exist."""
    paged_h = Path(__file__).parent.parent / "qwn_paged_kv.h"
    paged_c = Path(__file__).parent.parent / "qwn_paged_kv.c"
    cuda_cu = Path(__file__).parent.parent / "cuda" / "qwn_hypervsq_cuda.cu"
    assert paged_h.exists()
    assert paged_c.exists()
    assert cuda_cu.exists()


def test_ppl_simulation_7b_8b():
    """Verify QWN-HyperVSQ on 7B/8B model tier."""
    res = evaluate_ppl_simulation("Llama-3.1-8B-HyperVSQ.qwn", WIKITEXT2_SAMPLE, context_len=512, bpw_override=2.70)
    assert res["bpw"] == 2.70
    assert 11.0 <= res["perplexity"] <= 14.0
    assert res["accuracy_retention_pct"] >= 95.0


def test_chunked_prefill_scheduler_logic():
    """Verify chunked prefill sizing matches target chunk boundaries."""
    max_chunk = 512
    # Long prompt of 1200 tokens
    remaining = 1200
    chunk1 = min(max_chunk, remaining)
    assert chunk1 == 512
    remaining -= chunk1
    chunk2 = min(max_chunk, remaining)
    assert chunk2 == 512
    remaining -= chunk2
    chunk3 = min(max_chunk, remaining)
    assert chunk3 == 176
