# PROJECT_STATE.md

## Purpose

Qwanto is a local-first native inference runtime with a `.qwn` container,
OpenAI-compatible loopback gateway, shared React web UI, and Tauri desktop
host. The native design tiers model data across VRAM, RAM, and NVMe mmap.

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
- Evidence: `benchmarks/benchmark_reproducible.py` is the release benchmark
  boundary. It records only real local `qwnrun` executions and explicit
  classifications for all other outcomes.

## Current status

- Working directory: `D:\EcoUni\qwanto` on Windows PowerShell.
- Native/Python validation: `202 passed, 14 skipped` in `c/tests/`; the decoder
  test passed separately (`2 passed`). Safe model-acquisition tests use only
  local HTTP fixtures and cover resume, checksum, cancellation, disk, and
  atomic QWN behavior.
- Web validation: production build passed and Vitest passed (`47 tests`). The
  desktop surface has exactly five primary destinations—Project, Chats, Files,
  Changes, Settings—while the browser surface remains chat-only.
- A current local Windows `qwnrun.exe` build produced a real `MEASURED`
  `benchmark_evidence.json` record for the checked-in 4B `.qwn` fixture.
- Rust gates are pending on this workstation because `cargo` is not installed;
  hosted CI remains authoritative for Rust and cross-platform packaging. CI
  run `31966186386` passed native, web, security, Rust check/test/clippy, and
  changed-area gates.
- Release status: `v0.1.0-beta.1`, `v0.1.0-beta.2`, and `v0.1.0-beta.3` are
  published GitHub prereleases and must remain unchanged. Beta.3 hosted CI run
  `31966186386` and package/publish run `31966709143` passed. The public
  release contains five target-native installers, an installer SHA-256
  manifest, and factual unsigned/no-model release notes.

## Active limitations

- Native `.qwn` support is model/architecture dependent; unsupported shapes
  and formats must fail explicitly.
- GGUF models use the external local-runtime boundary and are not native QWN
  benchmark claims.
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
- macOS signing/notarization and installed-package smoke tests require the
  corresponding maintainer-owned platform credentials/runners.

## Important decisions

- Keep the web UI browser-safe and local-endpoint-only; privileged filesystem
  and process operations stay in the desktop host or gateway.
- Keep benchmark classifications `MEASURED`, `UNAVAILABLE`, `INVALID`,
  `TEST_FIXTURE`, `EXPERIMENTAL`, and `PROJECTED`; never substitute values.
- Release workflow supports manual or temporary-tag package validation and
  scheduled validation plus existing `v*` tags; tagged runs upload unsigned
  installers to a GitHub prerelease after the package matrix is green, with
  checksums and release notes.
- Preserve the tiered-memory architecture and upstream Colibri attribution.
