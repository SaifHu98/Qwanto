"""
Unit and Integration Tests for Qwanto Agentic Multi-Step Optimization Engine
Validates ToolResultCache (LRU+TTL), ParallelToolExecutor, and SessionContextManager.
"""

import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
ROOT_DIR = C_DIR.parent

sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(C_DIR / "tools"))

from qwn_agentic import (
    OptimizedAgent,
    ParallelToolExecutor,
    SessionContextManager,
    ToolResultCache,
)


class TestAgenticQuality(unittest.TestCase):
    def test_tool_cache_lru_and_ttl(self):
        """Verify tool result cache LRU eviction and TTL expiration."""
        cache = ToolResultCache(capacity=3, default_ttl=0.2)

        cache.set("web_search", {"q": "python"}, "res1")
        cache.set("web_search", {"q": "clang"}, "res2")
        cache.set("web_search", {"q": "rust"}, "res3")

        self.assertEqual(len(cache.cache), 3)
        self.assertEqual(cache.get("web_search", {"q": "python"}), "res1")

        # Insert 4th -> evicts clang
        cache.set("web_search", {"q": "golang"}, "res4")
        self.assertEqual(len(cache.cache), 3)
        self.assertIsNone(cache.get("web_search", {"q": "clang"}))
        self.assertIsNotNone(cache.get("web_search", {"q": "python"}))

        # Wait for TTL expiration
        time.sleep(0.25)
        self.assertIsNone(cache.get("web_search", {"q": "python"}))

    def test_parallel_tool_executor(self):
        """Verify parallel tool execution speedup."""
        cache = ToolResultCache()
        executor = ParallelToolExecutor(max_workers=8, cache=cache)

        def slow_tool(x):
            time.sleep(0.05)
            return f"result_{x}"

        tasks = [{"tool": "slow_op", "args": i, "fn": slow_tool} for i in range(8)]

        t0 = time.perf_counter()
        results = executor.execute_plan(tasks)
        elapsed = time.perf_counter() - t0

        self.assertEqual(len(results), 8)
        # Sequential would take ~0.40s. Parallel takes ~0.06s.
        self.assertLess(elapsed, 0.25)

        # Second execution must hit cache instantaneously (< 0.01s)
        t1 = time.perf_counter()
        cached_results = executor.execute_plan(tasks)
        cached_elapsed = time.perf_counter() - t1

        self.assertLess(cached_elapsed, 0.02)
        self.assertTrue(all(r["cached"] for r in cached_results))

    def test_session_context_reuse(self):
        """Verify session context manager and prefix freezing."""
        mgr = SessionContextManager()
        ctx = mgr.get_or_create("sess_123")
        self.assertFalse(ctx["is_frozen"])

        mgr.append_turn("sess_123", "Hello", "Hi there!")
        self.assertTrue(ctx["is_frozen"])
        self.assertGreater(ctx["frozen_tokens"], 0)

    def test_optimized_agent_task(self):
        """Verify end-to-end agent task execution."""
        model_path = ROOT_DIR / "experiments" / "results" / "4B_hyper_vsq2.qwn"
        if not model_path.exists():
            self.skipTest("Target model 4B_hyper_vsq2.qwn not present")

        agent = OptimizedAgent(model_path=model_path, max_workers=8)

        tools = [
            {"tool": "fetch_data", "args": {"source": "db1"}},
            {"tool": "fetch_data", "args": {"source": "db2"}},
        ]

        res = agent.process_task("Aggregate telemetry", tools=tools, session_id="sess_abc")
        self.assertEqual(res["tools_count"], 2)
        self.assertIn("elapsed_seconds", res)


if __name__ == "__main__":
    unittest.main()
