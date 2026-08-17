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
- Native/Python validation: `219 passed, 3 skipped` in `c/tests/`; the decoder
  test passed separately (`2 passed`). Focused gateway tests passed
  (`45 passed`). Safe model-acquisition tests use only
  local HTTP fixtures and cover resume, checksum, cancellation, disk, and
  atomic QWN behavior.
- Web validation: production build passed and Vitest passed (`56 tests`). The
  desktop surface has exactly five primary destinations—Project, Chats, Files,
  Changes, Settings—while the browser surface remains chat-only. Settings now
  uses an internal one-section-at-a-time navigation with viewport contracts for
  1280px, 1440px, 1920px, and short laptop heights. Skills & Plugins, GitHub,
  and Feedback are separate lazy settings sections; model/conversion/download
  work remains focused under Models.
- A current local Windows `qwnrun.exe` build produced a real `MEASURED`
  `benchmark_evidence.json` and `benchmarks/benchmark_matrix.json` record for
  the checked-in 4B `.qwn` fixture: CPU decode `8.154199 tok/s`; TTFT and CUDA
  execution counters are explicitly unavailable for this run.
- Rust gates are pending on this workstation because `cargo` and `make` are not installed;
  hosted CI remains authoritative for Rust and cross-platform packaging. The local
  native C syntax check and a Windows clang/OpenMP link test passed; the local CUDA
  device/toolkit is unavailable.
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

## Active limitations

- Native `.qwn` support is model/architecture dependent; unsupported shapes
  and formats must fail explicitly.
- GGUF, Safetensors, and PyTorch files are source artifacts only. They cannot be
  activated by qwnrun; only validated, architecture-compatible QWN conversions
  can become native runtime models. Qwen3.5 hybrid/MTP and 27B conversion paths
  fail explicitly until their tensor and reference-oracle validation exists.
- The gateway has explicit Hugging Face, allowlisted Direct HTTPS, and local
  file acquisition providers. Downloads use `.part` files and atomic publish;
  checksums, size, disk, redirects, and format compatibility are explicit.
- The Beta.3 Tauri package contains target-native qwnrun and gateway sidecar
  resources but no model weights. Converter/downloader controls remain behind
  Settings and explicit acquisition consent.
- TTFT and sensor metrics remain unavailable unless the runtime protocol or
  local sensor query supplies them.
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
- Preserve the tiered-memory architecture and upstream Colibri attribution.

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
