"""
Unit and Integration Tests for Qwanto Performance Autopilot System
Validates task classification, matrix selection, and multi-optimization orchestration.
"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
ROOT_DIR = C_DIR.parent

sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(C_DIR / "tools"))

from qwanto_autopilot import QwantoAutoPilot, TaskClassifier, TaskType


class TestAutoPilotQuality(unittest.TestCase):
    def test_task_classification(self):
        """Verify prompt heuristics and intent classification."""
        classifier = TaskClassifier()

        self.assertEqual(classifier.classify("What is the capital of France?"), TaskType.SIMPLE_QA)
        self.assertEqual(
            classifier.classify("def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
            TaskType.CODE_GENERATION,
        )
        self.assertEqual(
            classifier.classify("Explain step-by-step why quantum entanglement does not violate relativity and prove the no-communication theorem."),
            TaskType.REASONING,
        )
        self.assertEqual(
            classifier.classify("Lookup weather in Tokyo", tools=[{"name": "get_weather"}]),
            TaskType.AGENTIC,
        )
        self.assertEqual(
            classifier.classify("Run diagnostics", tools=[{"name": f"tool_{i}"} for i in range(5)]),
            TaskType.TOOL_INTENSIVE,
        )

    def test_autopilot_modes(self):
        """Verify performance profiles across modes."""
        model_path = ROOT_DIR / "experiments" / "results" / "4B_hyper_vsq2.qwn"
        pilot_balanced = QwantoAutoPilot(model_path=model_path, mode="balanced")
        pilot_max_perf = QwantoAutoPilot(model_path=model_path, mode="max-performance")
        pilot_max_qual = QwantoAutoPilot(model_path=model_path, mode="max-quality")

        # Balanced Code Gen
        res_code = pilot_balanced.generate("Write Python quicksort", max_tokens=16)
        self.assertGreaterEqual(res_code.speedup, 5.0)
        self.assertIn("turboquant", res_code.active_optimizations)
        self.assertIn("saguaro_ssd", res_code.active_optimizations)

        # Max Performance
        res_perf = pilot_max_perf.generate("Quick math", max_tokens=16)
        self.assertGreaterEqual(res_perf.speedup, 10.0)
        self.assertEqual(res_perf.thinking_level, "low")

        # Max Quality
        res_qual = pilot_max_qual.generate("Deep essay", max_tokens=16)
        self.assertEqual(res_qual.speedup, 1.0)
        self.assertEqual(res_qual.thinking_level, "high")


if __name__ == "__main__":
    unittest.main()
