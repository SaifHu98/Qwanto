"""
Unit and Integration Tests for Qwanto Saguaro (SSD) Speculative Decoding Engine
Validates LRU cache, context hashing, adaptive length heuristic, and execution.
"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
ROOT_DIR = C_DIR.parent

sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(C_DIR / "tools"))

from qwn_speculative import SaguaroEngine, SpeculationCache


class TestSpeculativeQuality(unittest.TestCase):
    def test_cache_hashing_and_lru(self):
        """Verify context prefix hashing and LRU cache eviction."""
        cache = SpeculationCache(capacity=3)
        h1 = cache.hash_context([1, 2, 3, 4])
        h2 = cache.hash_context([1, 2, 3, 5])
        h3 = cache.hash_context([1, 2, 3, 4])

        self.assertNotEqual(h1, 0)
        self.assertNotEqual(h1, h2)
        self.assertEqual(h1, h3)

        # Insert 3 elements
        cache.insert([10, 20], [100, 200])
        cache.insert([30, 40], [300, 400])
        cache.insert([50, 60], [500, 600])
        self.assertEqual(len(cache.cache), 3)

        # Lookup first element to refresh LRU
        res = cache.lookup([10, 20], 2)
        self.assertEqual(res, [100, 200])

        # Insert 4th element -> must evict [30, 40]
        cache.insert([70, 80], [700, 800])
        self.assertEqual(len(cache.cache), 3)
        self.assertIsNone(cache.lookup([30, 40], 2))
        self.assertIsNotNone(cache.lookup([10, 20], 2))
        self.assertIsNotNone(cache.lookup([70, 80], 2))

    def test_adaptive_draft_length(self):
        """Verify dynamic draft length adjustment."""
        target_model = ROOT_DIR / "experiments" / "results" / "4B_hyper_vsq2.qwn"
        if not target_model.exists():
            self.skipTest("Target model 4B_hyper_vsq2.qwn not present")

        engine = SaguaroEngine(
            target_model=target_model,
            max_draft_tokens=10,
        )

        engine.cache.total_drafted = 20
        engine.cache.acceptance_rate = 0.95
        self.assertEqual(engine.get_optimal_draft_length(), 8)

        engine.cache.acceptance_rate = 0.75
        self.assertEqual(engine.get_optimal_draft_length(), 5)

        engine.cache.acceptance_rate = 0.50
        self.assertEqual(engine.get_optimal_draft_length(), 3)

    def test_saguaro_engine_generation(self):
        """Verify Saguaro engine execution and stats recording."""
        model_path = ROOT_DIR / "experiments" / "results" / "4B_hyper_vsq2.qwn"
        if not model_path.exists():
            self.skipTest("Target model not present")

        engine = SaguaroEngine(target_model=model_path)
        try:
            res = engine.generate("What is 2 + 2?", max_tokens=8)
            self.assertGreater(res["tokens_generated"], 0)
            self.assertGreater(res["acceptance_rate"], 0.0)
            self.assertIn("optimal_draft_len", res)
        except OSError as e:
            if getattr(e, "winerror", None) == 4551:
                self.skipTest("Windows Application Control blocked binary execution")
            raise


if __name__ == "__main__":
    unittest.main()
