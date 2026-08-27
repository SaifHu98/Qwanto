# PROJECT_STATE.md

## Purpose

Qwanto Native is the umbrella local-first inference product with a `.qwn`
container, OpenAI-compatible loopback gateway, Qwanto Web console, and Qwanto
Code Tauri desktop agent. The native design tiers model data across VRAM, RAM,
and NVMe mmap.

## Stack and architecture

- Native runtime: C sources under `c/`, built as `qwnrun`; persistent serving
  uses the line protocol consumed by the gateway and Tauri runtime manager.
- Gateway: `c/openai_server.py`, loopback by default, with health, model,
  chat, telemetry, resource, acquisition, and benchmark-evidence endpoints;
  the desktop build freezes it as a supervised loopback sidecar.
- Web: `web/`, a static React/Vite client that talks to a configured local
  gateway and cannot access arbitrary files, terminals, or subprocesses.
- Desktop: `desktop/src-tauri/`, a Tauri v2 host for local runtime, sidecar
  supervision, and approval-gated agent capabilities. Release packaging stages
  target-native `qwnrun` and `qwanto-gateway` resources; model files remain
  user-managed.
- Evidence: `benchmarks/benchmark_reproducible.py` records only real local
  `qwnrun` executions; `benchmarks/generate_performance_report.py` combines
  matching model-manifest evidence without mixing conversion or external GGUF
  measurements into native inference claims.

## Current status

- Working directory: `D:\EcoUni\qwanto` on Windows PowerShell.
- Native/Python validation: `264 passed, 4 skipped, 17 subtests passed` in
  `c/tests/`; includes
  real `/v1/qwanto/models/verify` endpoint testing verifying 4KiB container
  invariants (header, payload alignment, 64-byte padding, tail block) and live
  `qwnrun` smoke test execution.
- Release candidate: annotated tag `v0.1.0-beta.7` and `main` are published at
  commit `5445240`. The first packaging run `33113738318` reached all three
  platforms; Windows failed in the Clang macro quoting step, while Ubuntu and
  macOS were still running at the last check. A workflow quoting fix is ready
  locally and must be pushed before the release can be considered built.
- Web validation: production build passed (`tsc -b && vite build`) and Vitest
  passed (`61 tests`). Qwanto Code desktop UI now uses the Qwanto Studio visual
  system with a fixed bottom input bar that mirrors modern chat surfaces
  (OpenCode, Claude.ai, ChatGPT desktop). The bar exposes a draft
  textarea, attach pill, send/stop button, token usage, and `Enter /
  Shift+Enter` shortcut affordance. The center panel and inspector reserve
  bottom padding so the composer never covers content. Browser chat stays
  chat-only with refreshed tokens. The verification modal reports structural
  invariants plus a live qwnrun `PING → PONG` roundtrip latency sourced from
  `/v1/qwanto/models/verify` and `native_smoke_test`. The desktop surface
  maintains exactly five primary destinations—Project, Chats, Files,
  Changes, Settings.
- Current converter coverage includes verified source decoders for GGUF
  Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K and IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS/NL,
  plus Q8_0 output from F32/F16/BF16 matrices and GGUF Q8_0/K-quant sources.
  `--quant none` preserves Q2_K/Q3_K/Q8_K and the supported IQ payloads as
  native QWN payloads. Native scalar QWN kernels now have differential
  coverage for Q2_K/Q3_K/Q8_K and IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS/NL.
  IQ1_S remains refused. The web converter and model settings expose Q8_0;
  native IQ selection remains an inspected/CLI capability, not a Flash-Next
  activation claim.
- Qwen3.8-27B now has a real converted local artifact at
  `models/Qwen3.8-27B-Q4_0.qwn`. Its CPU main path is integration-verified
  against the local source: full-attention layers, Gated DeltaNet recurrent
  state, causal convolution, gated RMSNorm, and the FFN execute through
  `qwnrun` for one generated token. This is a correctness/integration gate,
  not a benchmark row. MTP execution, MoE dispatch, QSA/ngram/gated-residual
  Flash-Next coverage, and hybrid CUDA coverage remain pending.
- Developer validation tools now exist at `c/tools/qwn_quality_oracle.py` and
  `c/tools/qwn_benchmark.py`. They require a real native artifact and pinned
  fixture/command; they emit `MEASURED_PASS`, `MEASURED`, or an explicit
  incomplete result and never create projected performance claims. `qwnrun`
  serving also has a deterministic `RESET` protocol command for oracle cases.
- Qwanto Code records redacted UI errors in a bounded local
  `.qwanto/diagnostics/errors.jsonl` stream. GitHub issue creation remains an
  explicit, user-approved browser action with a local diagnostic ZIP; no token
  or silent repository write is used.
- Official Qwen3.8 acquisition presets are exposed by the local gateway and
  Qwanto Code model settings: Flash-Next UD-Q4_K_XL from the official Unsloth
  four-shard repository (~111 GB), and Qwen3.8-27B Q4_K_M from the official
  GGML repository (~19 GB). Downloads are resumable per shard and write
  `.qwanto-bundle.json`; because the source listings provide no checksum
  metadata, bundles remain `unverified` and `external_source_only`. They are
  discoverable but intentionally cannot be activated by qwnrun.
- Local CUDA 13.3/MSVC 19.44 rebuilt both `qwn_cuda.dll` and `coli_cuda.dll`.
  The current HyperVSQ-2 ABI synthetic test passed on device 0 with one GPU
  matmul and zero reported fallbacks. This is correctness evidence only; it
  is not a general all-format or performance claim.
- TurboQuant remains intentionally unclaimed as paper-equivalent. The current
  `turboquant-q4` mode is the separately named `QWN-Q4-KV` compatibility
  representation; enabling the cited algorithm requires a distinct format,
  rotation/codebook contract, QJL residual path, and model-level quality
  evidence.
- The existing `benchmark_evidence.json` claim remains unchanged. CPU Phase 3
  local evidence from commit `cb3ca35` uses the 4B HyperVSQ-2 model SHA-256
  `43c128cdbf164e5aee8a192075961a514f87eda1c7c97c5d897d02eda2d29e36`.
  Rebuilt baseline qwnrun SHA-256 is
  `81503e04278d007e45fc85c7b69edd0d1cecf250152128370adfa53eb05a5454`;
  delayed-reduction candidate SHA-256 is
  `966b3e0f75cf2c851a33be431d67741c9b9b390d156d8a28f4d6c4f0d2c56866`.
  At eight workers, baseline/delayed release-quality warm decode medians are
  `17.981481/18.877406 tok/s` for 64 generated tokens and
  `17.845848/18.996390 tok/s` for 128 generated tokens. Exact streamed output
  agreement and 140/140 differential tests pass. The results are
  `MEASURED_LOCAL_PENDING_HOSTED_VALIDATION`; README performance claims remain
  intentionally unchanged.
- CPU Phase A local follow-up uses the production-default delayed VNNI path.
  Final local executable `c/qwnrun_phaseA_final.exe` has SHA-256
  `4d3f3a4b9eca86023b49056439298e7333f0d53f74a70af935c3e9f3fb5e621` and the
  model SHA-256 remains
  `43c128cdbf164e5aee8a192075961a514f87eda1c7c97c5d897d02eda2d29e36`.
  Release-quality clean-commit delayed medians are `18.985890 tok/s` for 64
  tokens and `18.945001 tok/s` for 128 tokens; same-binary disabled controls
  are `17.764969` and `17.737912`. Evidence is under
  `benchmarks/evidence/windows/2026-08-17/phaseA-clean-9a68691/`, records
  `git_worktree_dirty=false`, remains pending hosted validation, and README
  remains unchanged.
- Rust gates are pending on this workstation because `cargo` and `make` are not installed;
  hosted CI remains authoritative for Rust and cross-platform packaging. The local
  native C syntax check and a Windows clang/OpenMP link test passed. CUDA 13.3,
  MSVC 19.44, CMake 4.4.2, and Ninja 1.13.2 are now installed and the local
  HyperVSQ-2 CUDA reference path has compiled and passed synthetic and real-model
  decoder comparisons.
- Release status: `v0.1.0-beta.1`, `v0.1.0-beta.2`, `v0.1.0-beta.3`, the
  explicitly unsigned `v0.1.0-beta.4`, the explicitly unsigned `v0.1.0-beta.5`,
  and the explicitly unsigned `v0.1.0-beta.6` are published GitHub prereleases.
  Beta.6 was tagged on `61fb963` (post-fast-forward merge) and, per the user
  confirmation in this session, the GitHub releases page is publicly visible
  under https://github.com/SaifHu98/Qwanto/releases/tag/v0.1.0-beta.6 with the
  standard unsigned-platform banner and SHA-256 checksum attachment. The
  signed-credentials pipeline remained disabled so SmartScreen and Gatekeeper
  warnings still apply until credentials are configured in the `release-signing`
  GitHub environment.
- Hosted CI run `32017547742` exposed two integration issues in this follow-up:
  the POSIX native build needed `_GNU_SOURCE` before the project header for
  `Dl_info`, and `setup-python` needed an explicit CI dependency manifest for
  pip caching. Both are fixed; hosted run `32017793799` for commit `1a6b493`
  passed all scheduled jobs, including native Linux/Windows, Python, Web,
  documentation, and security checks.
- Hosted Rust correction run `32047197045` passed after commit `363d1d8`.
  The later CPU Phase 3 commits still require a complete hosted validation run
  on their final exact commit.
- Full hosted CI run `32061547684` passed on commit `b9b036e` with Linux and
  Windows native builds, Python, Web, Documentation, Security, and Rust/Tauri
  host jobs green. CUDA correctness/performance remains local because the
  hosted matrix has no real CUDA runner; no CUDA benchmark claim is promoted.
- CPU Phase A documentation was cleaned in commit `ffb46ac`; the canonical
  status is `docs/cpu-phaseA-feature-status-2026-08-17.md`. Its measurements
  remain bound to the recorded `9a68691` executable/source identity. The
  current CUDA follow-up changes native sources after that boundary, so a new
  clean CPU evidence regeneration is required before Phase A can be called
  current.
- CUDA Phase B has a versioned host/DLL ABI and an exact 74-byte HyperVSQ-2
  reference GEMV/GEMM source under `c/cuda/`, with secure runtime-directory
  loading, ABI checks, residency accounting, and fail-closed explicit CUDA.
  On the RTX 5070 Ti Laptop GPU, the ABI DLL compiled for detected `sm_120`,
  synthetic correctness passed, and scalar/VNNI real-model decoder comparisons
  passed with zero required-layer CPU fallbacks. A short persistent run reached
  `backend_actual=cuda` with 9,856 GPU matmuls. CUDA performance remains local
  diagnostic evidence pending hosted validation; the clean seven-request record
  is `benchmarks/evidence/windows/2026-08-17/cuda-phaseB-clean-4d26cdc/` at
  `6b7cf1a`, with median diagnostic decode `20.192933 tok/s`, 26,496 cumulative
  GPU matmuls, zero fallbacks, and 463,370,240 resident bytes. README has not
  changed.

## Active limitations

- Native `.qwn` support is model/architecture dependent; unsupported shapes
  and formats must fail explicitly.
- GGUF, Safetensors, and PyTorch files are source artifacts only. They cannot be
  activated by qwnrun; only validated, architecture-compatible QWN conversions
  can become native runtime models. The Qwen3.8 Q4 CPU main path is now
  integration-verified, while MTP, MoE, and hybrid CUDA remain explicitly
  gated. Native IQ2/IQ3/IQ4 row decoding and source-preserving conversion are
  verified independently; this does not imply complete Flash-Next support.
- Qwen3.8-27B qualification is `CPU_MAIN_PATH_INTEGRATION_VERIFIED` for the
  local Q4_0 conversion only: the source has 65 total layers, 48 Gated
  DeltaNet/SSM layers, 17 full-attention layers, one MTP prediction layer, and
  mixed IQ/K source tensors. The converted artifact is not full architecture
  support because MTP execution, MoE, quality oracle, and benchmark evidence
  are still absent. Native IQ2/IQ3/IQ4 tensor-row execution is verified, but
  it is not yet integrated into the full hybrid model path. Historical fail-closed evidence
  remains under `docs/qwen38-27b-evidence/`.
- The gateway has explicit Hugging Face, allowlisted Direct HTTPS, and local
  file acquisition providers. Downloads use `.part` files and atomic publish;
  checksums, size, disk, redirects, and format compatibility are explicit.
- The Beta.3 Tauri package contains target-native qwnrun and gateway sidecar
  resources but no model weights. Converter/downloader controls remain behind
  Settings and explicit acquisition consent.
- Gateway/client sensor metrics remain unavailable unless the runtime protocol
  or local sensor query supplies them; qwnrun's current phase evidence reports
  measured first-token latency and decode timing.
- CUDA full release-quality performance is not yet established. Explicit
  `--backend cuda` now fails closed when the versioned DLL/device is unavailable;
  on the local RTX host it reports actual CUDA only after a successful model
  matmul. GPU detection or DLL loading alone is never classified as CUDA
  inference.
- Phase 1 typed KV work is wired through qwnrun, the gateway, Tauri start
  options, the decoder, and runtime telemetry. FP16 remains the default. CPU
  Q8 is a scalar reference/attention-correct path; `turboquant-q4` reports the
  distinct `QWN-Q4-KV` compatibility representation rather than claiming the
  cited TurboQuant algorithm. The isolated CUDA Q8 reference passed on the
  local RTX 5070 Ti with max absolute error `1.1920929e-7`, five kernel
  invocations, and resident cleanup to zero. These are local results pending
  hosted validation and are not README performance claims.
- Local validation of the follow-up passed Python `258` tests with 4 skips and
  10 conversion subtests; the optimized Windows Clang qwnrun rebuild and the
  real Qwen3.8 Q4 CPU integration run also passed. `cargo` and `make` remain
  `NOT RUN LOCALLY — HOSTED VALIDATION REQUIRED`; CUDA NVCC synthetic and
  real-model decoder checks are now locally available and passing.
- CPU Phase 3 roofline counters for process reads, memory-controller bandwidth,
  cache misses, cycles, instructions, vector instructions, and OpenMP barrier
  time are `UNAVAILABLE` locally. The corrected selected 8-worker read-only
  stream proxy is `34.421311165 GB/s`; logical executed bytes are
  `481038007.52 bytes/token`, yielding a derived estimate of `71.556323 tok/s`.
  This is not hardware-measured bandwidth and is not a product claim.
- The gateway control-plane contract is versioned at schema `1`; `/health` is
  outside `/v1`, while models/config/telemetry remain under `/v1`.
- Third-party plugin execution remains unavailable by design until a native
  sandbox/supervisor and publisher trust store are configured; manifest
  validation and disabled-by-default app-data storage are implemented.
- Speculative decoding and JetSpec remain disabled while compatible native
  draft-model and tree-aware transaction prerequisites are unavailable; their
  counters do not start with fabricated acceptance or speedup values.
- Phase 2 now uses `qwn_speculative.c` in the qwnrun build, with typed
  compatibility checks, draft/target probability correction, bonus-token
  handling, and fail-closed CLI/gateway behavior. No compatible native draft
  QWN exists in the repository, so the product state remains
  `IMPLEMENTED_REQUIRES_COMPATIBLE_DRAFT_MODEL`; no speculative performance or
  acceptance result is claimed.
- macOS signing/notarization and installed-package smoke tests require the
  corresponding maintainer-owned platform credentials/runners.
- Windows Artifact Signing, macOS notarization, and Linux GPG signing are
  implemented as conditional protected `release-signing` environment gates.
  With credentials absent, Beta.4 is explicitly unsigned; enabling a platform
  gate without complete credentials or verification fails that package job.
- Skills and Plugins are available under Settings > Skills & Plugins. Built-in skills are
  locally invokable with `@skill-name`; native plugin manifests are checksum-
  and capability-validated, stored in app data disabled by default, and never
  executed without a future supervised sandbox.
- The generated QWN performance report keeps native `MEASURED`, conversion-only,
  and external GGUF evidence separate; it omits mismatched conversion artifacts
  instead of displaying stale sizes.

## Important decisions

- Keep the web UI browser-safe and local-endpoint-only; privileged filesystem
  and process operations stay in the desktop host or gateway.
- Keep benchmark classifications `MEASURED`, `UNAVAILABLE`, `INVALID`,
  `TEST_FIXTURE`, `EXPERIMENTAL`, and `PROJECTED`; never substitute values.
- Release workflow supports manual or temporary-tag package validation and
  scheduled validation plus existing `v*` tags; the Beta4 publish job runs
  after successful package jobs whether signing is absent or conditionally
  enabled and verified. Unsigned notes and checksum coverage are mandatory.
- Qwanto Code stores chat attachments and redacted feedback bundles only under
  the selected workspace; unsupported file/image model input is shown as
  unavailable instead of being sent to the runtime.
- GitHub connection is intentionally unavailable until a native OS-keychain
  credential backend is added; the Settings surface provides only safe public
  repository links, an approval-gated reporter skill, and never accepts a
  token in browser storage.
- Beta.3 remains unchanged. Existing Beta.4 is unsigned and remains tied to its
  already-published tag; follow-up work requires a new hosted validation cycle
  before any future package publication. Native runtime configuration now flows
  through CLI, gateway, desktop, and decoder; explicit CUDA is fail-closed, while
  auto mode remains on CPU until a real CUDA matmul completes.
- Windows qwnrun builds use real LLVM OpenMP and package `libomp140.x86_64.dll`
  beside the executable. The runtime reports actual ISA, OpenMP, CUDA counters,
  and a load-time CUDA DLL SHA-256 without hashing the hot path.
- Phase 2 CPU evidence work adds a release-quality persistent benchmark (one
  warmup plus seven measured requests), stderr draining for the serve harness,
  explicit build-info candidate-versus-executed semantics, bounded thread
  autotune evidence, activation-sum ablation counters, and a typed Auto/Manual
  worker policy in Qwanto Code. Activation-sum precompute is retained only
  when same-config measured evidence beats recomputation; CUDA remains out of
  scope and README performance claims remain unchanged.
- Preserve the tiered-memory architecture and upstream Colibri attribution.

- Phase A local decisions: delayed reduction is enabled by default with
  `QWN_HYPERVSQ2_DISABLE_DELAYED_REDUCTION=1` retained only as an explicit
  developer ablation override; row blocking is rejected for end-to-end
  performance; current shift/mask unpacking remains selected; SIMD SwiGLU is
  rejected as not material; OS-default affinity remains selected. CUDA was not
  started. Full hosted validation on the final exact commit is still required.

- Phase 2 clean evidence is now regenerated from commit `e23c2a8` with
  executable SHA-256 `3cca5eb31638ccaf8dad90992d46bd3828b6e2b9d09304bbf560a87e02e9f24b`.
  The persistent release-quality CPU record is `MEASURED` at 8 active VNNI
  workers with median warm decode `17.877580 tok/s`, p95 decode latency
  `3639.493 ms`, seven same-PID requests, and zero GPU matmuls/CPU fallbacks.
  Thread scaling measured requested=active workers 1/2/4/8/16/32; the 8-worker
  row was highest for this workload. Activation-sum precompute won the
  same-configuration ablation and is retained. README performance claims,
  CUDA, tags, and releases remain unchanged.

## 2026-08-17 — Desktop agent UX and evidence matrix

- **Change:** Kept the Qwanto Code primary navigation at Project, Chats, Files,
  Changes, and Settings; added a compact official-logo top bar, no-model state,
  file search, lazy gateway startup timing, and an inspector that starts
  collapsed and auto-reveals for diffs, approvals, output, or selected files.
- **Change:** Split Skills & Plugins, GitHub, and Feedback into focused internal
  Settings sections with keyboard-oriented vertical navigation; removed the
  duplicated GitHub card from Privacy & Internet.
- **Change:** Added benchmark matrix schema/generator and regenerated evidence
  and performance reports from the real local qwnrun executable/model hashes.
  CUDA remains unavailable because no GPU matmul was observed.
- **Files:** `web/src/App.tsx`, `web/src/components/DesktopAgentView.tsx`,
  `web/src/components/DesktopSettingsView.tsx`, `web/src/index.css`,
  `desktop/src-tauri/src/lib.rs`, `benchmarks/`, `benchmark_evidence.json`,
  `docs/performance*`, `README.md`, and focused tests.
- **Validation:** Full Python `219 passed, 3 skipped`; web `56 passed` and
  production build; brand/release/skill/documentation checks passed; local
  Cargo/Make/nvcc remain unavailable and require hosted CI.
- **Decision:** Do not create or modify a tag/release. Existing unsigned
  Beta.4 remains unchanged; this follow-up must pass hosted CI before any
  future publication.

## 2026-08-17 — Hosted Rust Clippy correction

- **Change:** Removed four needless picker `Ok`/`?` wrappers, replaced the
  attachment base64 capacity calculation with `usize::div_ceil`, and removed
  needless path/reference forms reported by hosted Clippy.
- **Evidence:** Hosted run `32021371447` passed native C, Python, Web, docs,
  and security, then failed only in Rust/Tauri Clippy with seven actionable
  annotations. No release or tag was created.
- **Validation:** `git diff --check` passed locally; Cargo remains unavailable
  here, so the corrected Rust gate is being verified by hosted CI.
- **Follow-up:** The first correction left one generic path-borrow Clippy
  diagnostic; `Path::as_path()` now makes both workspace containment checks
  explicit.

## 2026-08-17 — Windows native CI environment hardening

- **Evidence:** Hosted run `32022377288` passed the Ubuntu native job and the
  documentation/security gates but failed the Windows native job with only a
  generic process-exit annotation; the public log did not expose the failing
  command. The native source tree was unchanged from the preceding Windows
  success.
- **Change:** Made CI and future package builds discover x64 LLVM OpenMP
  import/runtime files from the active Clang installation and Visual Studio
  roots, reject ARM64 candidates, print the selected paths, and include the
  searched roots in strict failure messages.
- **Validation:** Workflow YAML parsed, release-policy validation passed, and
  `git diff --check` passed locally. Hosted Windows native validation remains
  required because this workstation has no native `make`/Cargo toolchain.
- **Decision:** No tag or release was created; the existing unsigned Beta.4
  remains unchanged.

- **Follow-up:** The initial diagnostic patch searched broad Visual Studio
  roots; the current working tree narrows that to active tool roots and
  explicit x64 tool/redist paths so hosted validation remains bounded.

- **Follow-up:** Broad Visual Studio roots were removed from the recursive
  search entirely; versioned x64 globs are now the only Visual Studio paths.

- **Follow-up:** The redist glob now stops at the versioned `VC\Redist\MSVC`
  root, allowing the actual `debug_nonredist\x64` layout to be discovered
  without scanning the complete Visual Studio installation.

- **Follow-up:** Windows CI/package builds resolve the Clang executable from
  PATH or standard LLVM locations and use that exact path for compiler and
  sccache invocation.

- **Follow-up:** The resolver also covers versioned Visual Studio
  `VC\Tools\Llvm\x64\bin\clang.exe` installations, the likely hosted runner
  location when Clang is not on PATH.

- **Current CI follow-up:** The hosted Windows native job still fails before
  compilation; public logs are authentication-gated. The resolver now uses
  `vswhere` when present, validates executable paths, and searches bounded x64
  LLVM/MSVC OpenMP locations. A retained PowerShell brace was removed after
  local AST parsing identified it.

- **Correction:** Hosted run `32025034411` showed the `vswhere` resolver still
  failed before compilation. The Windows CI and release build blocks are now
  restored to the exact known-good `f9b47ec` Clang/OpenMP invocation, with x64
  filters preserved. Hosted validation is pending after this correction.

## 2026-08-17 — Split runtime phase evidence

- **Change:** Added machine-readable cold-start, persistent prefill, persistent
  warm-decode, and thread-scaling harnesses. Warm decode requires two measured
  requests under one PID after a warmup request before classification.
- **Runtime audit:** qwnrun now exposes compiler/version, optimization flags,
  executable SHA-256, OpenMP compile/runtime state, requested and hot-path
  active threads, CPU ISA, model dtype, backend, CUDA counters, and PID. The
  HyperVSQ-2 74-byte CPU matmul path records actual participating OpenMP
  workers and dispatches AVX2/VNNI only when compiled code and runtime support
  both exist.
- **Validation:** Python `225 passed, 4 skipped`; web `56 passed` and
  production build; native C syntax and local Clang/OpenMP link passed. Fresh
  CPU phase evidence is `MEASURED`; explicit CUDA is `UNAVAILABLE` because
  `qwn_cuda.dll`/device support is absent. `cargo`, `make`, and `nvcc` are not
  installed on this workstation.
- **Decision:** README performance claims were intentionally not changed. No
  tag or release was created; existing Beta.4 remains unchanged.

## 2026-08-21 — Beta.6 desktop UX redesign (Qwanto Code)

- **Change:** Replaced the Beta.5 cyberpunk-heavy visual system with a calmer,
  professional Qwanto Code design system: tokenised dark palette
  (`--bg-0..3`, `--primary` `#7c9eff`, `--purple` `#a78bfa`, `--green` `#5fd9a8`),
  consistent `--radius-sm|md|lg|xl` and `--space-1..8`, focus rings,
  motion-respectful animations, and `prefers-reduced-motion` honored.
- **Change:** Introduced a fixed bottom input bar in Qwanto Code only. The new
  `desktop-composer` is sticky at the bottom of the main column with a
  send/stop button, attach pill, draft textarea, and a meta footer showing
  `Enter / Shift+Enter` shortcut and live token usage. The center panel and
  inspector reserve 150–180px bottom padding so the composer never covers
  content. The composer is disabled with explicit placeholders when the
  gateway is not connected or no validated model is active.
- **Change:** Refactored message rendering: avatar + role label + content
  inside a dedicated `desktop-message-content` grid, with a `chat-skill-preview`
  block surfaced above the timeline whenever `@skill-name` is being invoked.
  Auto-scroll on new messages. Mobile breakpoints collapse the sidebar and
  stack the workspace under the inspector.
- **Change:** Refreshed the Settings surface and verification modal with the
  same design tokens. The verification modal now uses a status bar
  (`passed` / `failed` variants), an evidence grid (format, quantization,
  tensors, smoke latency), an invariants checklist, and a live handshake row
  reporting `PING → PONG` with measured latency. SHA-256 stays honest: shown
  when `validate_qwn` returns one, otherwise the structural-bound reason.
- **Verification wiring:** Real native verification goes through the existing
  `/v1/qwanto/models/verify` endpoint. The handler resolves the path against
  `_is_safe_path()` + configured model dirs + project root, returns
  `incompatible_format` for non-`.qwn` artifacts, runs `validate_qwn` for
  header / tail / payload / padding invariants, then runs `native_smoke_test`
  to spawn `qwnrun <model> --serve`, send `PING\n`, and parse `PONG`. The
  test gateway run reports `latency_ms` from a `time.perf_counter()` window
  around the live process. No client-side fabrication — when the gateway is
  unreachable or the executable is missing the modal reflects `failed` with
  the upstream reason.
- **Files:** `web/src/index.css`, `web/src/components/DesktopAgentView.tsx`,
  `web/src/components/DesktopSettingsView.tsx` (verification modal only),
  `web/src/__tests__/desktop_ui.test.tsx`,
  `web/src/__tests__/desktop_visual.test.tsx`,
  `web/src/__tests__/responsive_ui.test.ts`. Browser chat visual tokens are
  refreshed for consistency; the browser surface keeps its own shell and
  stays chat-only.
- **Validation:** Web `61 passed` and `tsc -b && vite build` clean
  (81.43 kB CSS / 280.87 kB JS, gzip 15.20 kB / 84.24 kB). Full Python suite
  `253 passed, 4 skipped` in 93.54s. Native C/OpenMP rebuild via
  `python c/tools/build_and_run_c_tests.py` passed all 17 binaries
  (including the HyperVSQ-2 differential 140/140, speculative 433/433,
  runtime config, KV-cache, decode batch, scheduler, and protocol suites).
  Gateway integration test `test_gateway_integration.py` (which exercises
  `/v1/qwanto/models/verify` end-to-end with a real subprocess) passed
  3/3 in 4.43s.
- **Decision:** Do not tag or publish Beta.6 from this workstation. Push the
  change, wait for a full green hosted CI run (Linux native, Windows native,
  Python, web build, docs, security, Rust/Tauri), then publish
  `v0.1.0-beta.6` as an explicit UNSIGNED prerelease with the standard
  checksum coverage. Existing Beta.4 / Beta.5 installers remain available
  and unchanged.

## 2026-08-22 — v0.1.0-beta.6 closeout

- **Tag:** `v0.1.0-beta.6` annotated, peeled to commit `61fb963a82bdbc0b8121ea4503885aa4fa4d0acc`.
- **Branch state at tag:** `main` fast-forwarded to `61fb963` from `cb432f2`
  via `git merge --ff-only feature/beta6-redesign`. The three commits now on
  `main` are:
  - `08e7486` feat(ui): modern desktop redesign + fixed bottom composer.
  - `2fb5b5d` docs(quantization): supported/unsupported matrix + 4-phase roadmap.
  - `61fb963` docs(perf): honest README/manifest/format audit + Q4_0 row.
- **Release signature:** UNSIGNED on Windows, macOS, and Linux. No signing
  credentials present in the `release-signing` GitHub environment, so the
  publish job ran with the explicit unsigned-prerelease banner and the
  standard SHA-256 checksum file. SmartScreen and Gatekeeper warnings apply.
- **Verified locally before tag push:**
  - `python -m pytest c/tests/ -q` → 253 passed, 4 skipped.
  - `npm test` → 61 passed; `npm run build` clean (81.43 kB CSS, 280.87 kB JS, gzip 15.20 / 84.24).
  - `python c/tools/build_and_run_c_tests.py` → 17/17 binaries clean (HyperVSQ-2 140/140, speculative 433/433, runtime config, KV, scheduler, protocol).
  - `python -m pytest c/tests/test_gateway_integration.py -q` → 3/3 (real `/v1/qwanto/models/verify` PING -> PONG subprocess roundtrip).
- **What changed in this release, attested by the tagged commit:**
  - Modern professional Qwanto Code UI with fixed bottom input bar (OpenCode / Claude.ai style composer) at desktop surface; browser chat stayed in-card.
  - New `docs/qwn-supported-quantizations.md` with the full container-dtype table mapped to `c/qwanto_native.h` lines 53-62 and a complete "what we will not claim" closing section that points at the assertion file for each refusal.
  - New `docs/dtype-support-roadmap.md` as a 4-phase forward plan with prerequisites, reference-oracle requirements, acceptance gates, and out-of-scope enumeration per phase. No code shipped for any phase.
  - `docs/model-manifest.json` upgraded to `2.1.0` with the new `DeepSeek-R1-Distill-Qwen-1.5B-Q4_0` QWN row (two MEASURED executions: 1.718 tok/s @ 64 tokens, 2.474 tok/s @ 128 tokens with `--threads 8`) plus the existing 4B HyperVSQ-2 canonical row. The Qwen-3.5-4B-MTP and Qwen-3.8-27B-IQ2_M source GGUF entries are recorded with `architecture` / `family` fields and the exact blocker reason in each `inference_evidence.reason`. No fabrication of round-trip numbers.
  - `benchmarks/generate_performance_report.py` made data-driven: the format-status table is now driven by the actual MEASURED rows in the evidence set rather than a stale hardcoded UNVAILABLE for Q4_0, with Q8_0 / BF16 / VSQ / VSQ_ULTRA / HYPER_VSQ rows added so the table covers every implemented dtype.
  - `benchmarks/generate_performance_report.py` tests tightened: schema requires `model_architecture` and `qwn_dtype` even when those values are tagged Unavailable before promotion. CUDA classifier now rejects rows where `--backend cuda` reports zero `gpu_matmul_count`, so a CUDA-classified run that never executed any GPU matmul cannot be promoted to MEASURED.
  - README, `docs/performance.md`, `docs/qwn-format.md`, and `docs/performance-report.{md,json}` all updated to reflect the new Q4_0 MEASURED row, the corrected dtype table, the CUDA fail-closed status on the 1.5B model, and the new reproduce commands for both MEASURED fixtures.
- **Decision (release closeout):** the release workflow attached the
  standard UNSIGNED banner and the SHA-256 coverage; hosted CI for the branch
  was green before `main` was fast-forwarded, and the tag was created from
  the final `61fb963` commit. No README performance claim was edited to
  advertise Beta.6 beyond the documented MEASURED rows.
- **Decision (next steps):** no further work is scheduled on this session.
  Future Qwen-3.5-MTP and Qwen-3.8-27B work must follow
  `docs/dtype-support-roadmap.md` Phases 3 and 4 respectively, gated on
  reference oracles. Speculative decoding with a non-native QWN draft remains
  refused. TurboQuant, JetSpec, SlimInfer, BitDecoding remain reference only.
## 2026-08-22 — Multi-backend dispatch audit (no hot-path changes)

- **Audit:** Inspected `c/glm.c`, `c/qwanto_decode.c`, `c/cuda/qwn_cuda_abi.h`, and
  `c/cuda/qwn_hypervsq_cuda.cu` to understand the existing CPU/CUDA dispatch
  surface. The product **already** ships a multi-backend design (`--backend
  auto/cpu/cuda` plus per-op CUDA knobs `COLI_CUDA`, `COLI_CUDA_ATTN`,
  `COLI_CUDA_DENSE`, `COLI_CUDA_EXPERT_GB`), but routing is whole-session: a
  model is either CPU-only or CUDA-only for the whole generation pass. There
  is **no** per-forward-pass intra-op CPU+GPU scheduling.
- **Measurement:** Captured three additional MEASURED rows for 4B HyperVSQ-2
  on RTX 5070 Ti Laptop (`qwn_cuda.dll` SHA `7bf5e3e595ed47fba17b79ccf74c334c42b53ba7ea5766738d721e372edd4dcb`,
  `qwnrun.exe` SHA `fc9086962f9ba0fa77d758a972ccfce6a0fc04b1cce27fdbc75df04109b22881`,
  git commit `8d6d0ebe84de42d4905e47792fccd53913349923`): CPU 128 tok = 8.784
  tok/s, CUDA 128 tok = 8.216 tok/s (gpu_matmul_count=11264, cpu_fallback_count=0),
  CUDA 64 tok = 4.414 tok/s (gpu_matmul_count=7168, cpu_fallback_count=0). CUDA
  ABI 1 ships only the q8-quantized HyperVSQ-2 reference path on RTX 5070 Ti
  Laptop and is 5.9% slower than the local CPU VNNI path at 128 tokens. The
  1.5B Q4_0 model still cannot use CUDA today: the CUDA attempt exits `rc=-1`
  with `layer 0 attn matmul failed`. See
  `benchmarks/evidence/windows/2026-08-22/1.5b_q4_0_cuda_attempt.json`.
- **Decision (no code shipped):** A "true CPU+GPU in the same forward pass"
  implementation requires per-op routing tables, CUDA streams + OpenMP
  barrier orchestration, multi-backend memory budgeting, per-shape
  empirical policy, a hybrid-aware reference oracle for differential tests,
  and a coherent KV cache layout across both backends. None of those exist;
  building them honestly is weeks of engineering, not in this session's
  scope. The dispatch code in `c/glm.c` and `c/qwanto_decode.c` is left
  untouched.
- **Decision (committed):** `docs/parallel-execution-roadmap.md` records the
  five-phase plan: P0 capture-only telemetry flag, P1 per-layer measured
  routing, P2 shape-aware active policy, P3 full hybrid with CUDA streams,
  and P4 broaden CUDA dtype coverage (overlaps with
  `docs/dtype-support-roadmap.md` Phases 1+2). Phase P0 is the only one that
  fits in a small honest commit window; even P0 has not been implemented
  yet. No code in `c/qwanto_decode.c`, `c/glm.c`, `c/cuda/qwn_cuda_abi.h`,
  or `c/qwnrun.c` was touched in this work.
- **Validation:** Web 61/61 + production build clean. Python 253/253 passed,
  4 skipped. Native 17/17 binaries clean. Gateway integration 3/3.
