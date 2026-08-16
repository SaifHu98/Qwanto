# PROJECT_STATE.md

## Purpose

Qwanto is a local-first native inference runtime with a `.qwn` container,
OpenAI-compatible loopback gateway, shared React web UI, and Tauri desktop
host. The native design tiers model data across VRAM, RAM, and NVMe mmap.

## Stack and architecture

- Native runtime: C sources under `c/`, built as `qwnrun`; persistent serving
  uses the line protocol consumed by the gateway and Tauri runtime manager.
- Gateway: `c/openai_server.py`, loopback by default, with health, model,
  chat, telemetry, resource, and benchmark-evidence endpoints.
- Web: `web/`, a static React/Vite client that talks to a configured local
  gateway and cannot access arbitrary files, terminals, or subprocesses.
- Desktop: `desktop/src-tauri/`, a Tauri v2 host for local runtime and agent
  capabilities. Release packaging stages a target-native `qwnrun` resource;
  model files remain user-managed.
- Evidence: `benchmarks/benchmark_reproducible.py` is the release benchmark
  boundary. It records only real local `qwnrun` executions and explicit
  classifications for all other outcomes.

## Current status

- Working directory: `D:\EcoUni\qwanto` on Windows PowerShell.
- Native/Python validation: `200 passed, 14 skipped` in `c/tests/`; the decoder
  test passed separately (`2 passed`). Safe model-acquisition tests use only
  local HTTP fixtures and cover resume, checksum, cancellation, disk, and
  atomic QWN behavior.
- Web validation: production build passed and Vitest passed (`35 tests`).
- A current local Windows `qwnrun.exe` build produced a real `MEASURED`
  `benchmark_evidence.json` record for the checked-in 4B `.qwn` fixture.
- Rust gates are pending on this workstation because `cargo` is not installed;
  CI remains authoritative for Rust and cross-platform packaging.
- Release status: Beta / not release-ready until fresh CI and a green package
  workflow provide the remaining evidence. `workflow_dispatch` validates
  packages; an existing `v*` tag publishes a prerelease only after all matrix
  jobs pass. No tag or release was created by this work.

## Active limitations

- Native `.qwn` support is model/architecture dependent; unsupported shapes
  and formats must fail explicitly.
- GGUF models use the external local-runtime boundary and are not native QWN
  benchmark claims.
- The gateway has explicit Hugging Face, allowlisted Direct HTTPS, and local
  file acquisition providers. Downloads use `.part` files and atomic publish;
  checksums, size, disk, redirects, and format compatibility are explicit.
- The Tauri Beta package contains qwnrun only. Converter/downloader controls
  are disabled in the installed shell until a gateway sidecar is packaged and
  supervised.
- TTFT and sensor metrics remain unavailable unless the runtime protocol or
  local sensor query supplies them.
- macOS signing/notarization and installed-package smoke tests require the
  corresponding maintainer-owned platform credentials/runners.

## Important decisions

- Keep the web UI browser-safe and local-endpoint-only; privileged filesystem
  and process operations stay in the desktop host or gateway.
- Keep benchmark classifications `MEASURED`, `UNAVAILABLE`, `INVALID`,
  `TEST_FIXTURE`, `EXPERIMENTAL`, and `PROJECTED`; never substitute values.
- Release workflow supports manual or temporary-tag package validation and existing `v*` tags;
  tagged runs upload unsigned installers to a GitHub prerelease after the
  package matrix is green, without creating tags.
- Preserve the tiered-memory architecture and upstream Colibri attribution.
