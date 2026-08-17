# Correct speculative decoding boundary

Qwanto keeps draft/target speculation separate from JetSpec. The native
decoder accepts only validated QWN models; a GGUF source, a same-model copy,
or a synthetic token stream is not a draft model.

## Compatibility contract

Before initialization, the typed engine requires:

- target and draft SHA-256 identities;
- equal vocabulary size and token-ID mapping hash;
- equal BOS/EOS IDs;
- equal context policy;
- a validated chat-template identity in QWN metadata.

The current QWN metadata does not expose the last item. Therefore the current
product status is `IMPLEMENTED_REQUIRES_COMPATIBLE_DRAFT_MODEL`, and the CLI
remains fail-closed when `--speculative` is supplied. The existing 1.5B GGUF
is not used directly.

## Algorithm implemented in the gated engine

The typed `QwnSpecContext` path contains the real draft/target probability
flow: draft proposal probabilities, target verification, acceptance with
`min(1, p/q)`, residual probability correction after rejection, correction
sampling, EOS handling, deterministic PRNG state, and replay-based rollback
for the draft cache. Target and draft histories are advanced only for
committed tokens. Top-p is rejected until a shared sampler contract is
available, because silently applying different truncation rules would change
the distribution.

Counters are zero until a real request executes. They include proposed,
accepted, rejected, bonus/correction tokens, target passes, acceptance rate,
draft/verification/rollback/correction time, and paired baseline/speculative
rates. No default acceptance or speedup is initialized.

The legacy cache/ring symbols remain link-compatible for older tests but do
not generate tokens or report a speedup. They are reference compatibility
only. Promotion requires deterministic distribution tests, KV transaction
invariants, an actual compatible QWN draft, long-generation agreement, and a
positive measured net speedup.
