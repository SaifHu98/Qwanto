import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

import generate_performance_report


class TestPerformanceReport(unittest.TestCase):
    def test_report_keeps_native_and_external_evidence_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            evidence_path = root / "evidence.json"
            bpw_path = root / "bpw.csv"
            empirical_path = root / "empirical.json"
            artifact = root / "4B_hyper_vsq2.qwn"
            artifact.write_bytes(b"native fixture")

            manifest = {
                "models": [{
                    "model_id": "fixture-4B",
                    "format": "qwn",
                    "target_file": "4B_hyper_vsq2.qwn",
                    "target_size_bytes": artifact.stat().st_size,
                    "target_sha256": generate_performance_report._sha256(artifact),
                    "quantization": "HyperVSQ-2",
                }]
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            evidence_path.write_text(json.dumps({
                "benchmark_id": "fixture-benchmark",
                "evidence_classification": "MEASURED",
                "model_metadata": {"sha256": manifest["models"][0]["target_sha256"]},
                "host_environment": {"os": "TEST_FIXTURE", "cpu_model": "fixture"},
                "measured_evidence": {"tok_per_sec": 12.5, "ttft_ms": 0.0},
            }), encoding="utf-8")
            with bpw_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "label", "mode", "ok", "out_bytes_actual", "effective_bpw",
                    "wall_seconds", "throughput_mb_s",
                ])
                writer.writeheader()
                writer.writerow({
                    "label": "4B", "mode": "hyper_vsq2", "ok": "True",
                    "out_bytes_actual": str(artifact.stat().st_size),
                    "effective_bpw": "2.34", "wall_seconds": "1.0",
                    "throughput_mb_s": "2.0",
                })
            empirical_path.write_text(json.dumps({
                "llama_server_benchmarks": {"external": {
                    "model": "fixture.gguf", "decode_tok_s_mean": 99.0,
                    "decode_tok_s_median": 98.0, "ttft_ms_mean": 10.0,
                    "cold_load_seconds": 1.0,
                }}
            }), encoding="utf-8")

            report = generate_performance_report.build_report(
                manifest_path, [evidence_path], bpw_path, empirical_path
            )

        self.assertEqual(report["native_qwn_rows"][0]["tokens_per_second"], 12.5)
        self.assertIsNone(report["native_qwn_rows"][0]["ttft_ms"])
        self.assertEqual(report["native_qwn_rows"][0]["evidence_class"], "MEASURED")
        self.assertEqual(report["external_gguf_rows"][0]["evidence_class"], "EXPERIMENTAL_EXTERNAL")
        self.assertNotIn("99.0", json.dumps(report["native_qwn_rows"]))

    def test_mismatched_conversion_artifact_is_not_reported_as_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "experiments" / "results"
            results.mkdir(parents=True)
            artifact = results / "4B_q4_0.qwn"
            artifact.write_bytes(b"different artifact")
            original_root = generate_performance_report.ROOT
            try:
                generate_performance_report.ROOT = root
                rows, excluded = generate_performance_report._conversion_evidence({
                    ("4B", "q4_0"): {
                        "ok": "True", "out_bytes_actual": "1", "effective_bpw": "4.5",
                        "wall_seconds": "1", "throughput_mb_s": "1",
                    }
                })
            finally:
                generate_performance_report.ROOT = original_root
        self.assertEqual(rows, [])
        self.assertIn("differs from evidence size", excluded[0]["reason"])


if __name__ == "__main__":
    unittest.main()
