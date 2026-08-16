import unittest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestBenchmarksEndpoint(unittest.TestCase):
    def test_static_benchmark_claims_are_not_shipped(self):
        root = Path(__file__).resolve().parent.parent
        self.assertFalse((root / "baseline.json").exists())
        self.assertFalse((root / "candidate.json").exists())

    def test_evidence_artifact_schema_is_machine_readable_when_present(self):
        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / "benchmark_evidence.json",
            root.parent / "benchmark_evidence.json",
            root.parent / "benchmarks" / "benchmark_evidence.json",
        ]
        present = next((path for path in candidates if path.is_file()), None)
        if present is None:
            self.skipTest("real benchmark evidence artifact is not present")
        with present.open(encoding="utf-8") as stream:
            data = json.load(stream)
        self.assertIn(data.get("evidence_classification"), {
            "MEASURED", "UNAVAILABLE", "INVALID", "TEST_FIXTURE", "EXPERIMENTAL", "PROJECTED"
        })
        self.assertIn("execution_evidence", data)


if __name__ == "__main__":
    unittest.main()
