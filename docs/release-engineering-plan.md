# Qwanto release-engineering plan

Status: implementation plan for the current Beta release track.

## Current architecture and relationships

- `c/qwnrun.c` is the native local inference process. One-shot execution emits
  diagnostics on stderr; `--serve` keeps one process alive and accepts the
  `PING`, `CONFIG`, `FORWARD`, `SUBMIT`, and `CANCEL` protocol on stdin.
- `c/openai_server.py` is the local OpenAI-compatible gateway. It owns the HTTP
  API, model selection, local static web serving, authentication, queueing, and
  the native process lifecycle. Its default bind address is loopback.
- `web/` is the only UI. A browser calls the configured gateway over HTTP; it
  cannot invoke Tauri commands or access arbitrary local files and processes.
- `desktop/` packages the same `web/dist` UI in Tauri and exposes the native
  runtime and agent tools through Rust commands. The desktop process must find a
  target-native `qwnrun` resource, while models remain user-managed files and
  are never bundled.
- `desktop/src-tauri/src/permission_policy.rs` is the agent authorization
  boundary. Read-only tools may run in Plan mode; writes, edits, command
  execution, staging, and commits require an approval token and a canonical
  workspace. Tool execution uses argv directly and must not introduce a shell
  escape hatch.
- `benchmarks/benchmark_reproducible.py` is an evidence producer, not a
  performance fixture. Only a successful real `qwnrun` process with a valid
  positive token count may receive `MEASURED` classification.

## Evidence-backed gaps to close

1. Tauri has `targets: all` but no target-aware `qwnrun` resource declaration,
   release metadata, or tag-triggered packaging workflow. Packaging must fail
   clearly when a required binary is absent, and must not contain model files.
2. The desktop runtime lookup needs to match the packaged resource layout on
   Windows, macOS, and Linux and report a useful missing-runtime error.
3. The web UI needs an explicit local endpoint/onboarding state and a visible
   browser-versus-desktop boundary. It must not imply browser filesystem,
   terminal, or agent permissions that only Tauri provides.
4. Benchmark execution needs deterministic classifications for missing
   executable/model (`UNAVAILABLE`), invalid fixture (`INVALID`), nonzero exit,
   zero-token output, malformed protocol/output, test-only fixtures
   (`TEST_FIXTURE`), experimental runs (`EXPERIMENTAL`), and projections
   (`PROJECTED`). It must retain command argv, hashes, raw-output hashes, and
   measured values only when the evidence is valid.
5. Existing documentation and UI/test fixtures contain stale or hard-coded
   machine and throughput claims. They must be labeled as historical evidence,
   test fixtures, experimental, or projected, or removed from production
   claims. Missing large-model fixtures may skip only at the test boundary.
6. CI needs an explicit release-build validation path while preserving the
   existing Linux Tauri dependencies, NumPy installation, native decoder tests,
   and persistent serve-protocol tests.

## Implementation order

1. Add focused benchmark parsing/classification tests and make the harness
   produce schema-valid evidence without fabricated metrics.
2. Add target-native Tauri resource configuration and a release workflow for
   `v*` tags. The workflow builds the native binary per OS, builds the shared
   web UI, runs the required checks, and packages Windows NSIS/MSI, macOS DMG,
   and Linux AppImage/deb artifacts. It does not create tags, publish releases,
   download models, or mask failures.
3. Harden desktop resource lookup and expose the existing agent/runtime
   boundary clearly in the shared web UI. Keep browser mode HTTP-only and
   localhost-first.
4. Replace unsupported benchmark/documentation claims with evidence links and
   explicit status categories. Add the required architecture, format, UI,
   desktop-agent, packaging, security, readiness, and manifest documents.
5. Run the mandated validation sequence in order, then inspect the complete
   diff for whitespace, scope, missing resources, and unverified claims.

## Release acceptance criteria

- Native C unit/decoder tests and persistent `--serve` tests execute; large
  real-model tests skip only when their named fixture is absent.
- Rust check, test, and Clippy with `-D warnings` pass.
- The web build and unit tests pass.
- A release build contains the target-native `qwnrun` resource and no model
  file; the packaged runtime can resolve the resource path.
- The gateway defaults to loopback and external backends/downloads remain
  explicit opt-ins. Agent writes, commands, staging, and commits remain
  approval-gated and workspace-contained.
- Every published benchmark number has a real evidence artifact with matching
  executable/model hashes and command arguments. Unknown telemetry is `null`
  with an explanation, never a guessed value.
- `RELEASE_READINESS.md` says Beta-ready only when the CI and documentation
  evidence is present; otherwise it names the remaining blocker.

## Non-goals

- No model download, model bundling, cloud runtime, external LLM, or remote
  agent service is added.
- No second frontend, broad Rust rewrite, native hot-path refactor, or unrelated
  feature work is included.
- No tag, GitHub Release, or deployment is created automatically by this work.
