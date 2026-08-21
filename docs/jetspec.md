# JetSpec status

JetSpec is currently `REFERENCE_ONLY` and disabled by default. The C data
structures can build a bounded causal tree from caller-supplied probability
fixtures, remove duplicate token IDs, build an ancestor mask, and verify a
candidate path against target logits. Counters start at zero and are updated
only by those calls; no acceptance rate or speedup is fabricated.

This is not yet product tree speculation. A production implementation must
obtain every proposal from a compatible native QWN draft model or validated
MTP heads, run batched target verification with position IDs and branch cache
mapping, then commit or roll back the KV transaction. Parent-token arithmetic,
synthetic IDs, and initialized speedup values are forbidden.

The feature remains disabled because the current repository has no compatible
draft QWN model and no validated tree-aware decoder/KV transaction path.
Promotion requires distribution correctness, transaction correctness,
end-to-end quality, and repeatable positive net speedup. Until then, the
fixture API is useful only for deterministic structural tests and must not be
exposed as an active runtime capability.
