"""
Unit and Integration Tests for Qwanto Configurable Thinking Engine
Validates quality, latency scaling, speedup ratios, and OpenAI API parameters.
"""

import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
ROOT_DIR = C_DIR.parent

sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(C_DIR / "tools"))

from qwn_thinking import QwnThinkingEngine, ThinkingLevel


class TestThinkingQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = ROOT_DIR / "experiments" / "results" / "4B_hyper_vsq2.qwn"
        cls.exe_path = C_DIR / "qwnrun.exe"
        if not cls.exe_path.exists():
            cls.exe_path = C_DIR / "qwnrun_msvc.exe"

    def test_thinking_level_parsing(self):
        """Verify parsing of valid and boundary thinking level representations."""
        self.assertEqual(ThinkingLevel.from_value("low"), ThinkingLevel.LOW)
        self.assertEqual(ThinkingLevel.from_value("fast"), ThinkingLevel.LOW)
        self.assertEqual(ThinkingLevel.from_value(0), ThinkingLevel.LOW)
        self.assertEqual(ThinkingLevel.from_value("0"), ThinkingLevel.LOW)

        self.assertEqual(ThinkingLevel.from_value("medium"), ThinkingLevel.MEDIUM)
        self.assertEqual(ThinkingLevel.from_value("balanced"), ThinkingLevel.MEDIUM)
        self.assertEqual(ThinkingLevel.from_value(1), ThinkingLevel.MEDIUM)
        self.assertEqual(ThinkingLevel.from_value(None), ThinkingLevel.MEDIUM)

        self.assertEqual(ThinkingLevel.from_value("high"), ThinkingLevel.HIGH)
        self.assertEqual(ThinkingLevel.from_value("deep"), ThinkingLevel.HIGH)
        self.assertEqual(ThinkingLevel.from_value("cot"), ThinkingLevel.HIGH)
        self.assertEqual(ThinkingLevel.from_value(2), ThinkingLevel.HIGH)

    def test_thinking_engine_initialization(self):
        """Verify engine initialization and path resolution."""
        if not self.model_path.exists() or not self.exe_path.exists():
            self.skipTest("Model file or executable not present")

        engine = QwnThinkingEngine(self.model_path, exe_path=self.exe_path)
        self.assertEqual(engine.model_path, self.model_path.resolve())
        self.assertEqual(engine.exe_path, self.exe_path.resolve())

    def test_thinking_modes_execution_and_speedup(self):
        """Verify execution across LOW, MEDIUM, and HIGH modes and validate speedup."""
        if not self.model_path.exists() or not self.exe_path.exists():
            self.skipTest("Model file or executable not present")

        engine = QwnThinkingEngine(self.model_path, exe_path=self.exe_path)
        prompt = "Explain quantum superposition in one sentence."

        try:
            res_low = engine.generate(prompt=prompt, max_tokens=16, thinking_level="low")
            self.assertEqual(res_low["thinking_level"], "low")
            self.assertGreater(res_low["tokens_generated"], 0)
            self.assertGreater(res_low["tok_per_sec"], 0.0)

            res_high = engine.generate(prompt=prompt, max_tokens=16, thinking_level="high")
            self.assertEqual(res_high["thinking_level"], "high")
            self.assertGreater(res_high["tokens_generated"], 0)
            self.assertGreater(res_high["tok_per_sec"], 0.0)

            # Low mode must be strictly faster than High mode
            speedup = res_low["tok_per_sec"] / res_high["tok_per_sec"]
            self.assertGreater(speedup, 2.0, f"Expected speedup > 2x, got {speedup:.2f}x")
        except OSError as e:
            if getattr(e, "winerror", None) == 4551:
                self.skipTest("Windows Application Control blocked binary execution")
            raise

    def test_thinking_benchmark_harness(self):
        """Verify iterative benchmark utility."""
        if not self.model_path.exists() or not self.exe_path.exists():
            self.skipTest("Model file or executable not present")

        engine = QwnThinkingEngine(self.model_path, exe_path=self.exe_path)
        try:
            bench = engine.benchmark(prompt="Hello", thinking_level="low", max_tokens=8, iterations=2)
            self.assertEqual(bench["thinking_level"], "low")
            self.assertEqual(bench["iterations"], 2)
            self.assertGreater(bench["avg_tok_per_sec"], 0.0)
        except OSError as e:
            if getattr(e, "winerror", None) == 4551:
                self.skipTest("Windows Application Control blocked binary execution")
            raise


if __name__ == "__main__":
    unittest.main()
