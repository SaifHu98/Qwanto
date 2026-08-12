import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai_server import ResponseCache


class TestResponseCache(unittest.TestCase):
    def test_cache_put_get(self):
        cache = ResponseCache(max_entries=5, ttl_seconds=60)
        prompt = "Explain quantum computing"
        payload = {"choices": [{"message": {"content": "Quantum computing uses qubits..."}}]}

        # Cache miss initially
        self.assertIsNone(cache.get(prompt, 0.0, 1.0, "qwanto-model"))

        # Put entry
        cache.put(prompt, 0.0, 1.0, "qwanto-model", payload)

        # Cache hit
        cached = cache.get(prompt, 0.0, 1.0, "qwanto-model")
        self.assertIsNotNone(cached)
        self.assertEqual(cached["choices"][0]["message"]["content"], "Quantum computing uses qubits...")

        # Temperature > 0 should not cache or return cached
        self.assertIsNone(cache.get(prompt, 0.7, 1.0, "qwanto-model"))


if __name__ == "__main__":
    unittest.main()
