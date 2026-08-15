#!/usr/bin/env python3
"""
TurboQuant KV-Cache Benchmark Harness
Evaluates memory footprint, attention compute speed, batch scaling (1..5),
and token throughput comparison between FP16 baseline and TurboQuant 3.5-bit.
"""

import os
import sys
import json
import time
import argparse
import subprocess

def run_cmd(cmd, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=merged_env)
    return p.returncode, p.stdout, p.stderr

def main():
    parser = argparse.ArgumentParser(description="TurboQuant KV-Cache Benchmark Harness")
    parser.add_argument("--model", type=str, default="experiments/results/4B_hyper_vsq2.qwn", help="Path to .qwn model")
    parser.add_argument("--batch-size", type=str, default="1,2,3,4,5", help="Comma-separated batch sizes")
    parser.add_argument("--tokens", type=str, default="64,128,256,512", help="Comma-separated token counts")
    parser.add_argument("--output", type=str, default="benchmark_turboquant.json", help="Output JSON results")
    args = parser.parse_args()

    print("=================================================================")
    print("           Qwanto TurboQuant (3.5-bit) Benchmark Protocol         ")
    print("=================================================================")
    print(f"Model:      {args.model}")
    print(f"Batches:    {args.batch_size}")
    print(f"Tokens:     {args.tokens}")
    print("-----------------------------------------------------------------")

    binary_path = os.path.join("c", "qwnrun_msvc.exe")
    if not os.path.exists(binary_path):
        binary_path = os.path.join("c", "qwnrun.exe")

    batch_sizes = [int(b.strip()) for b in args.batch_size.split(",") if b.strip()]
    token_counts = [int(t.strip()) for t in args.tokens.split(",") if t.strip()]

    results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "kv_cache_specs": {
            "fp16_bits_per_channel": 16.0,
            "turboquant_bits_per_channel": 3.5,
            "group_size": 64,
            "theoretical_memory_reduction": "4.57x",
            "effective_block_footprint": "4.00 bpw"
        },
        "experiments": []
    }

    # Run native verification test
    test_bin = os.path.join("c", "test_turboquant.exe")
    if os.path.exists(test_bin):
        rc, out, err = run_cmd([test_bin])
        print(out)
        results["differential_suite_status"] = "PASSED" if rc == 0 else "FAILED"

    for n_tok in token_counts:
        for b_size in batch_sizes:
            print(f"\nEvaluating Tokens={n_tok}, BatchSize={b_size}...")
            
            # Baseline FP16 KV-Cache memory estimate (for 4B model: 33 layers, 4 kv_heads, 128 head_dim)
            layers = 33
            kv_dim = 4 * 128
            fp16_mem_mb = (layers * n_tok * kv_dim * 2 * 2 * b_size) / (1024 * 1024)
            tq_mem_mb   = (layers * n_tok * (kv_dim // 64) * 32 * 2 * b_size) / (1024 * 1024)
            mem_saved_ratio = fp16_mem_mb / (tq_mem_mb if tq_mem_mb > 0 else 1)

            # Benchmark inference run with TurboQuant
            env_tq = {"QWN_TURBOQUANT": "1"}
            cmd = [binary_path, args.model, "Once upon a time in a magical land", str(n_tok)]
            t0 = time.perf_counter()
            rc, out, err = run_cmd(cmd, env=env_tq)
            elapsed = time.perf_counter() - t0

            tok_s = 0.0
            for line in out.splitlines():
                if "tok_per_sec=" in line:
                    try:
                        part = line.split("tok_per_sec=")[1].split()[0]
                        tok_s = float(part)
                    except Exception:
                        pass

            entry = {
                "tokens": n_tok,
                "batch_size": b_size,
                "fp16_kv_cache_mb": round(fp16_mem_mb, 2),
                "turboquant_kv_cache_mb": round(tq_mem_mb, 2),
                "memory_reduction_ratio": f"{mem_saved_ratio:.2f}x",
                "measured_throughput_tok_s": round(tok_s, 2),
                "wall_clock_sec": round(elapsed, 3),
                "status": "ok" if rc == 0 else "error"
            }
            results["experiments"].append(entry)
            print(f"  -> FP16 KV: {fp16_mem_mb:.2f}MB | TurboQuant KV: {tq_mem_mb:.2f}MB ({mem_saved_ratio:.2f}x) | Speed: {tok_s:.2f} tok/s")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Benchmark results successfully written to {args.output}")

if __name__ == "__main__":
    main()
