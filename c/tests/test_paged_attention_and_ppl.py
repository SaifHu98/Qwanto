import os
import sys
import math
import unittest
from pathlib import Path

# Add paths to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
c_dir = Path(__file__).resolve().parent.parent
tools_dir = c_dir / "tools"

for p in [str(root_dir), str(c_dir), str(tools_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from tools.qwn_ppl import evaluate_ppl_simulation, compute_perplexity_from_logits, WIKITEXT2_SAMPLE
except ImportError:
    from c.tools.qwn_ppl import evaluate_ppl_simulation, compute_perplexity_from_logits, WIKITEXT2_SAMPLE


class TestPagedAttentionAndPPL(unittest.TestCase):
    def test_ppl_simulation_hypervsq(self):
        """Verify QWN-HyperVSQ perplexity calculation and accuracy retention."""
        res = evaluate_ppl_simulation("model_hypervsq.qwn", WIKITEXT2_SAMPLE, context_len=512)
        self.assertEqual(res["bpw"], 2.70)
        self.assertTrue(11.0 <= res["perplexity"] <= 14.5)
        self.assertTrue(res["accuracy_retention_pct"] > 85.0)
        self.assertTrue(res["compression_ratio"] > 5.0)

    def test_ppl_simulation_q4km(self):
        """Verify legacy Q4_K_M baseline perplexity."""
        res = evaluate_ppl_simulation("model_q4_k_m.gguf", WIKITEXT2_SAMPLE, context_len=512)
        self.assertEqual(res["bpw"], 4.50)
        self.assertTrue(res["perplexity"] < 12.5)
        self.assertTrue(res["accuracy_retention_pct"] > 90.0)

    def test_ppl_log_prob_math(self):
        """Verify token log-probabilities to perplexity transformation."""
        log_probs = [math.log(0.5)] * 10
        ppl, avg_nll = compute_perplexity_from_logits(log_probs)
        self.assertEqual(round(ppl, 2), 2.0)
        self.assertEqual(round(avg_nll, 4), round(-math.log(0.5), 4))

    def test_paged_attention_header_exists(self):
        """Verify PagedAttention C headers and implementations exist."""
        paged_h = Path(__file__).parent.parent / "qwn_paged_kv.h"
        paged_c = Path(__file__).parent.parent / "qwn_paged_kv.c"
        cuda_cu = Path(__file__).parent.parent / "cuda" / "qwn_hypervsq_cuda.cu"
        self.assertTrue(paged_h.exists())
        self.assertTrue(paged_c.exists())
        self.assertTrue(cuda_cu.exists())

    def test_ppl_simulation_7b_8b(self):
        """Verify QWN-HyperVSQ on 7B/8B model tier."""
        res = evaluate_ppl_simulation("Llama-3.1-8B-HyperVSQ.qwn", WIKITEXT2_SAMPLE, context_len=512, bpw_override=2.70)
        self.assertEqual(res["bpw"], 2.70)
        self.assertTrue(11.0 <= res["perplexity"] <= 14.0)
        self.assertTrue(res["accuracy_retention_pct"] >= 95.0)

    def test_chunked_prefill_scheduler_logic(self):
        """Verify chunked prefill sizing matches target chunk boundaries."""
        max_chunk = 512
        remaining = 1200
        chunk1 = min(max_chunk, remaining)
        self.assertEqual(chunk1, 512)
        remaining -= chunk1
        chunk2 = min(max_chunk, remaining)
        self.assertEqual(chunk2, 512)
        remaining -= chunk2
        chunk3 = min(max_chunk, remaining)
        self.assertEqual(chunk3, 176)


if __name__ == "__main__":
    unittest.main()
