import json
import sys
import argparse

def analyze(baseline_path, candidate_path, justification):
    with open(baseline_path, 'r') as f:
        base = json.load(f)
    with open(candidate_path, 'r') as f:
        cand = json.load(f)

    # Validate strict parity in testing conditions
    conditions = ['model_hash', 'quantization', 'context_size', 'cache_state', 'prompt_hash', 'generated_tokens']
    for c in conditions:
        if base.get(c) != cand.get(c):
            print(f"ERROR: Cannot compare. Condition mismatch for {c}: {base.get(c)} != {cand.get(c)}")
            sys.exit(1)

    base_speed = base['median_tok_s']
    cand_speed = cand['median_tok_s']
    base_rss = base['peak_rss_mb']
    cand_rss = cand['peak_rss_mb']

    # Higher is better
    speed_diff = (cand_speed - base_speed) / base_speed
    speed_ratio = cand_speed / base_speed
    
    # Lower is better
    rss_diff = (cand_rss - base_rss) / base_rss
    
    print(f"--- Benchmark Comparison ---")
    print(f"Speed (tok/s): {base_speed:.2f} -> {cand_speed:.2f} ({speed_diff*100:+.2f}%)")
    print(f"Peak RSS (MB): {base_rss:.2f} -> {cand_rss:.2f} ({rss_diff*100:+.2f}%)")

    # Gate 1: No regression > 5% without justification
    if speed_diff < -0.05:
        if not justification:
            print("ERROR: Speed regression > 5% detected without justification.")
            sys.exit(1)
        else:
            print(f"WARNING: Speed regression > 5%. Justification provided: {justification}")
            
    if rss_diff > 0.05:
        if not justification:
            print("ERROR: Memory usage regression > 5% detected without justification.")
            sys.exit(1)
        else:
            print(f"WARNING: Memory regression > 5%. Justification provided: {justification}")

    # Gate 2: Strict format rules
    # "Only write 2x faster when median improvement is >= 1.90x"
    if speed_ratio >= 1.90:
        print("CLAIM PERMITTED: 2x faster")
    else:
        print(f"CLAIM ENFORCED: {speed_diff*100:+.2f}% faster")

    # "Only write 50% lower memory when peak RSS is reduced by at least 45%"
    if rss_diff <= -0.45:
        print("CLAIM PERMITTED: 50% lower memory usage")
    else:
        print(f"CLAIM ENFORCED: {rss_diff*-100:+.2f}% lower memory usage")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline')
    parser.add_argument('candidate')
    parser.add_argument('--justification', default='', help='Justification for regression')
    args = parser.parse_args()
    analyze(args.baseline, args.candidate, args.justification)
