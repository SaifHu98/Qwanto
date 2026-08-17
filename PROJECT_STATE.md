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
- Native/Python validation: `243 passed, 4 skipped` in `c/tests/`; safe
  model-acquisition tests use only
  local HTTP fixtures and cover resume, checksum, cancellation, disk, and
  atomic QWN behavior.
- Web validation: production build passed and Vitest passed (`56 tests`). The
  desktop surface has exactly five primary destinations—Project, Chats, Files,
  Changes, Settings—while the browser surface remains chat-only. Settings now
  uses an internal one-section-at-a-time navigation with viewport contracts for
  1280px, 1440px, 1920px, and short laptop heights. Skills & Plugins, GitHub,
  and Feedback are separate lazy settings sections; model/conversion/download
  work remains focused under Models.
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
- Release status: `v0.1.0-beta.1`, `v0.1.0-beta.2`, `v0.1.0-beta.3`, and the
  explicitly unsigned `v0.1.0-beta.4` are published GitHub prereleases. Beta.4
  package/publish run `31974921398` passed and its assets contain only installers
  plus SHA-256 coverage. The current follow-up changes are after the existing
  Beta.4 tag and must not be silently described as part of that release.
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
  can become native runtime models. Qwen3.5 hybrid/MTP and 27B conversion paths
  fail explicitly until their tensor and reference-oracle validation exists.
- Qwen3.8-27B qualification is currently `UNSUPPORTED_QWEN38_ARCHITECTURE`:
  the local GGUF has 65 layers, 48 Gated DeltaNet/SSM layers, 17 full-attention
  layers, four MTP tensors, and mixed IQ dtypes not supported by the converter.
  The qualification tool records every source tensor and refuses conversion
  before output; generated evidence is under `docs/qwen38-27b-evidence/` and
  binds to clean source commit `a198402`; no QWN support or benchmark claim
  exists for this source.
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
- Local validation of the follow-up passed Python `243/243` executed tests with
  4 skips, focused ABI/evidence tests `21 passed`, Web build and `56` Vitest
  tests, and C/OpenMP syntax/link checks. `cargo` and `make` remain
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
