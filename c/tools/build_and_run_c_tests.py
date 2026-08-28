#!/usr/bin/env python3
"""Cross-platform C test builder and runner for CI and developer environments.

Works without requiring make on Windows.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]

QWNRUN_LIB_SRCS = [
    "qwn_runtime_config.c", "qwanto_decode.c", "qwanto_native.c", "qwanto_kernels.c",
    "qwanto_turboquant.c", "qwanto_gpu.c", "qwanto_autopilot.c", "qwanto_thinking.c",
    "qwn_speculative.c", "qwanto_agentic.c", "qwanto_bitdecoding.c", "qwanto_jetspec.c",
    "qwanto_talon.c", "qwanto_sliminfer.c", "qwanto_pquant.c", "qwanto_littlebit.c",
    "qwn_paged_kv.c",
]

TEST_SPECS = [
    ("tests/test_json", ["tests/test_json.c"]),
    ("tests/test_st", ["tests/test_st.c"]),
    ("tests/test_tier", ["tests/test_tier.c"]),
    ("tests/test_grammar", ["tests/test_grammar.c"]),
    ("tests/test_schema_gbnf", ["tests/test_schema_gbnf.c"]),
    ("tests/test_decode_batch", ["tests/test_decode_batch.c"]),
    ("tests/test_idot", ["tests/test_idot.c"]),
    ("tests/test_kv_alloc", ["tests/test_kv_alloc.c"]),
    ("tests/test_i4_acc512", ["tests/test_i4_acc512.c"]),
    ("tests/test_compat_direct", ["tests/test_compat_direct.c"]),
    ("tests/test_scheduler", ["tests/test_scheduler.c"]),
    ("tests/test_qwanto_arch", ["tests/test_qwanto_arch.c", "qwanto_core.c", "qwanto_router.c", "qwanto_attention.c", "aio_compat.c"]),
    ("tests/test_qwanto_native", ["tests/test_qwanto_native.c", "qwanto_native.c", "qwanto_kernels.c", "qwanto_core.c"]),
    ("tests/test_hypervsq2_kernels", ["tests/test_hypervsq2_kernels.c", "qwanto_native.c", "qwanto_kernels.c"]),
    ("tests/test_kv_cache", ["tests/test_kv_cache.c", "qwanto_turboquant.c", "qwanto_native.c", "qwanto_kernels.c"]),
    ("tests/test_turboquant_paper", ["tests/test_turboquant_paper.c", "qwanto_turboquant.c", "qwanto_native.c", "qwanto_kernels.c"]),
    ("tests/test_runtime_config", ["tests/test_runtime_config.c", "qwn_runtime_config.c"]),
    ("tests/test_speculative", ["tests/test_speculative.c"] + QWNRUN_LIB_SRCS),
]


def find_omp_lib() -> str | None:
    if sys.platform != "win32":
        return None
    try:
        res = subprocess.run(["clang", "--print-file-name=libomp.lib"], capture_output=True, text=True, check=False)
        omp_path = res.stdout.strip()
        if os.path.isfile(omp_path) and "\\arm64\\" not in omp_path.lower():
            return omp_path
    except Exception:
        pass
    for root_dir in [r"C:\Program Files\Microsoft Visual Studio", r"C:\Program Files (x86)\Microsoft Visual Studio"]:
        if os.path.isdir(root_dir):
            for root, _, files in os.walk(root_dir):
                if "libomp.lib" in files and (r"\lib\x64" in root or r"\lib\amd64" in root):
                    return os.path.join(root, "libomp.lib")
    return None


def main() -> int:
    cc = os.environ.get("CC", "clang" if sys.platform == "win32" else "gcc")
    ext = ".exe" if sys.platform == "win32" else ""
    omp_lib = find_omp_lib() if sys.platform == "win32" else None

    flags = [
        "-O3", "-mavx2", "-mf16c", "-mfma", "-fopenmp",
        "-D_CRT_SECURE_NO_WARNINGS", "-D_FILE_OFFSET_BITS=64", "-Wno-deprecated-declarations",
        "-I.", "-I" + str(C_DIR),
    ]
    ldflags = []
    if sys.platform == "win32":
        ldflags.append("-lpsapi")
        if omp_lib:
            ldflags.append(omp_lib)
    else:
        ldflags.append("-lm")
        if sys.platform == "linux":
            ldflags.extend(["-pthread", "-ldl"])

    built_bins = []
    for target, srcs in TEST_SPECS:
        target_bin = C_DIR / f"{target}{ext}"
        cmd = [cc] + flags + [str(C_DIR / s) for s in srcs] + ["-o", str(target_bin)] + ldflags
        print(f"[build] Compiling {target}...", flush=True)
        res = subprocess.run(cmd, cwd=C_DIR, check=False)
        if res.returncode != 0:
            print(f"[ERROR] Failed to compile {target}", file=sys.stderr)
            return res.returncode
        built_bins.append(target_bin)

    print("\n[test] Running C test suite...", flush=True)
    failures = []
    for binary in built_bins:
        print(f"--> Running {binary.name}...", flush=True)
        res = subprocess.run([str(binary)], cwd=C_DIR, check=False)
        if res.returncode != 0:
            print(f"[FAIL] {binary.name} exited with code {res.returncode}", file=sys.stderr)
            failures.append(binary.name)
        else:
            print(f"[PASS] {binary.name}\n", flush=True)

    if failures:
        print(f"\n[FAILED] {len(failures)} tests failed: {', '.join(failures)}", file=sys.stderr)
        return 1

    print(f"\n[SUCCESS] All {len(built_bins)} C tests passed cleanly!", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
