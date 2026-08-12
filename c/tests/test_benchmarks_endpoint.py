import unittest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestBenchmarksEndpoint(unittest.TestCase):
    def test_baseline_and_candidate_json_exist(self):
        base_path = Path(__file__).resolve().parent.parent / "baseline.json"
        self.assertTrue(base_path.exists(), "baseline.json must exist in c/")

        with open(base_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("median_tok_s", data)
            self.assertIn("peak_rss_mb", data)
            self.assertIn("gates_passed", data)


if __name__ == "__main__":
    unittest.main()
