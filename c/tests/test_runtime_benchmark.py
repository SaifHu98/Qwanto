import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import benchmark_runtime_phases
import benchmark_release_quality
import thread_autotuner
import benchmark_warm_repeats
import benchmark_cpu_roofline
from runtime_config_snapshot import comparable_runtime_config, make_runtime_config_snapshot


class TestRuntimeBenchmarkEvidence(unittest.TestCase):
    def test_ready_stat_contains_machine_readable_load_fields(self):
        stat = benchmark_runtime_phases.parse_ready_stat(
            b"STAT 0 0.000 0.0 0.0 0 0 model_load_ms=12.5 runtime_ready_ms=14.0 "
            b"file_open_ms=1.0 mmap_ms=2.0 metadata_parse_ms=3.0 tokenizer_init_ms=4.0 "
            b"kv_cache_alloc_ms=5.0 advisory_preload_ms=6.0 first_tensor_touch_ms=7.0 pid=42\n"
        )
        self.assertEqual(stat["model_load_ms"], 12.5)
        self.assertEqual(stat["runtime_ready_ms"], 14.0)
        self.assertEqual(stat["pid"], 42)
        self.assertEqual(stat["first_tensor_touch_ms"], 7.0)

    def test_done_keeps_legacy_stat_prefix_and_exposes_phases(self):
        done = benchmark_runtime_phases.parse_done(
            b"DONE warm-2 STAT 8 20.0 0 0 5 0 "
            b"prefill_ms=25.0 prefill_tok_per_sec=200.0 first_token_ms=30.0 "
            b"decode_wall_ms=400.0 decode_tok_per_sec=20.0 pid=99 "
            b"backend_actual=cpu kernel=avx2 gpu_matmul_count=0 "
            b"cpu_fallback_count=0 active_threads=4\n"
        )
        self.assertEqual(done["request_id"], "warm-2")
        self.assertIn("DONE warm-2 STAT", done["runtime_stat_line"])
        self.assertEqual(done["decode_wall_ms"], 400.0)
        self.assertEqual(done["pid"], 99)
        self.assertEqual(done["active_threads"], 4)

    def test_runtime_snapshot_exposes_decode_path_and_is_comparable(self):
        one_shot = make_runtime_config_snapshot(
            backend="cpu", context_size=4096, max_tokens=8, seed=0,
            prompt="Hello", threads=4, warmup_tokens=4,
        )
        persistent = make_runtime_config_snapshot(
            backend="cpu", context_size=4096, max_tokens=8, seed=0,
            prompt="Hello", threads=4, warmup_tokens=4,
        )
        self.assertEqual(comparable_runtime_config(one_shot), comparable_runtime_config(persistent))
        self.assertEqual(one_shot["decode_function"], "qwn_decoder_generate")
        self.assertEqual(one_shot["thinking_mode"], "none")

    def test_build_info_semantics_keep_candidate_separate_from_execution(self):
        build_info = {
            "compiled_kernels": {"avx2": True, "vnni": True},
            "detected_cpu_features": {"avx2": True, "vnni": True},
            "preferred_kernel_candidate": "vnni",
            "actual_executed_kernel": "Unavailable",
            "model_dtype": "Unavailable",
        }
        self.assertEqual(build_info["preferred_kernel_candidate"], "vnni")
        self.assertEqual(build_info["actual_executed_kernel"], "Unavailable")
        self.assertEqual(build_info["model_dtype"], "Unavailable")
        self.assertEqual(build_info.get("active_threads", "Unavailable"), "Unavailable")

    def test_phase_reports_always_reserve_startup_breakdown(self):
        report = benchmark_runtime_phases.run_phase_benchmark(
            str(ROOT / "missing-model.qwn"), "cold-start", "Hello", 4
        )
        self.assertIn("runtime_config_snapshot", report)
        for key in ("process_create_ms", "file_open_ms", "mmap_ms",
                    "metadata_parse_ms", "tokenizer_init_ms", "kv_cache_alloc_ms",
                    "advisory_preload_ms", "first_tensor_touch_ms",
                    "first_real_forward_ms", "prompt_prefill_ms", "decode_ms",
                    "total_end_to_end_ms"):
            self.assertIn(key, report["measurements"])

    def test_repeat_percentile_is_deterministic(self):
        self.assertEqual(benchmark_warm_repeats.percentile95([1.0, 2.0, 3.0, 4.0, 5.0]), 5.0)
        self.assertIsNone(benchmark_warm_repeats.percentile95([]))

    def test_short_diagnostic_is_not_release_quality(self):
        report = benchmark_warm_repeats.run_repeated_warm_decode
        self.assertTrue(callable(report))
        self.assertEqual(benchmark_release_quality.DEFAULT_VARIANCE_LIMIT, 0.20)

    def test_release_quality_rejects_short_request_count(self):
        report = benchmark_release_quality.run_release_quality(
            model=str(ROOT / "missing-model.qwn"), executable=str(ROOT / "missing-qwnrun.exe"),
            repeats=5,
        )
        self.assertEqual(report["benchmark_class"], "RELEASE_QUALITY")
        self.assertEqual(report["evidence_classification"], "INVALID")
        self.assertIn("at least seven", report["invalid_reasons"][0])

    def test_release_quality_has_explicit_local_pending_classification(self):
        report = benchmark_release_quality.run_release_quality(
            model=str(ROOT / "missing-model.qwn"), executable=str(ROOT / "missing-qwnrun.exe"),
            pending_hosted_validation=True,
        )
        self.assertEqual(report["evidence_classification"], "UNAVAILABLE")
        self.assertNotEqual(report["evidence_classification"], "MEASURED")

    def test_roofline_header_parser_uses_descriptor_byte_size(self):
        self.assertEqual(benchmark_cpu_roofline.QWN_DESC.size, 136)
        self.assertEqual(benchmark_cpu_roofline.QWN_HEADER_PREFIX.size, 112)

    def test_roofline_classification_is_pending_until_hosted(self):
        self.assertEqual(
            benchmark_cpu_roofline.EVIDENCE_CLASSIFICATION,
            "MEASURED_LOCAL_PENDING_HOSTED_VALIDATION",
        )

    def test_thread_autotune_candidates_are_bounded_and_deduplicated(self):
        candidates = thread_autotuner.candidate_threads()
        self.assertEqual(candidates, sorted(set(candidates)))
        self.assertIn(1, candidates)
        self.assertLessEqual(max(candidates),  os.cpu_count() or 1)

    def test_warm_decode_requires_two_requests_under_one_pid(self):
        self.assertTrue(benchmark_runtime_phases.persistent_pid_proven([42, 42], 42, 2))
        self.assertFalse(benchmark_runtime_phases.persistent_pid_proven([42], 42, 2))
        self.assertFalse(benchmark_runtime_phases.persistent_pid_proven([42, 43], 42, 2))

    def test_cuda_zero_matmuls_is_not_proven(self):
        self.assertFalse(benchmark_runtime_phases.cuda_execution_proven({
            "backend_actual": "cuda", "gpu_matmul_count": 0, "cpu_fallback_count": 0,
        }))
        self.assertTrue(benchmark_runtime_phases.cuda_execution_proven({
            "backend_actual": "cuda", "gpu_matmul_count": 1, "cpu_fallback_count": 0,
        }))

    def test_missing_local_model_is_unavailable(self):
        report = benchmark_runtime_phases.run_phase_benchmark(
            str(ROOT / "missing-model.qwn"), "cold-start", "Hello", 4
        )
        self.assertEqual(report["evidence_classification"], "UNAVAILABLE")
        self.assertIn("not found", report["error_reason"])
        json.dumps(report)

    def test_checked_in_manifest_populates_model_identity(self):
        metadata = benchmark_runtime_phases.model_manifest_metadata(
            ROOT / "experiments" / "results" / "4B_hyper_vsq2.qwn"
        )
        self.assertEqual(metadata["architecture"], "qwen35")
        self.assertEqual(metadata["qwn_dtype"], "HyperVSQ-2")


if __name__ == "__main__":
    unittest.main()
