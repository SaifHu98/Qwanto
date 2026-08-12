import unittest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai_server import load_presets, save_presets, DEFAULT_PRESETS


class TestPresetsAndTelemetry(unittest.TestCase):
    def test_load_default_presets(self):
        presets = load_presets()
        self.assertIsInstance(presets, list)
        self.assertGreaterEqual(len(presets), 5)
        ids = [p["id"] for p in presets]
        self.assertIn("code_expert", ids)
        self.assertIn("balanced", ids)

    def test_save_and_reload_preset(self):
        initial = load_presets()
        test_preset = {
            "id": "test_unit",
            "name": "Unit Test Preset",
            "system_prompt": "Test prompt",
            "temperature": 0.5,
            "top_p": 0.9,
            "description": "For testing"
        }
        initial.append(test_preset)
        save_presets(initial)

        reloaded = load_presets()
        reloaded_ids = [p["id"] for p in reloaded]
        self.assertIn("test_unit", reloaded_ids)

        # Cleanup
        cleaned = [p for p in reloaded if p["id"] != "test_unit"]
        save_presets(cleaned)


if __name__ == "__main__":
    unittest.main()
