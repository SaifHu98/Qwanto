# Model acquisition and conversion design

Status: design baseline for the Beta desktop release.

## Existing behavior

Qwanto has two separate local workflows today:

- The Python gateway (`c/openai_server.py`) exposes `/v1/qwanto/models`,
  `/v1/qwanto/download`, `/v1/qwanto/convert`, and polling endpoints. The
  existing download manager writes directly to a destination chosen by the
  request and reports a coarse progress object. The converter loads
  `c/tools/qwn_convert.py` in a background thread.
- The Tauri application starts the packaged `qwnrun` executable and exposes
  model discovery, runtime control, agent permissions, and telemetry through
  Rust commands. It does not package Python, the gateway, or a converter/
  downloader sidecar.

The shared web dashboard is therefore not allowed to imply that conversion or
remote acquisition is available in an installed Tauri build. Those controls
remain gateway capabilities only until a reviewed gateway sidecar is packaged
and safely supervised.

## Verified source-format matrix

The matrix is derived from `c/tools/qwn_convert.py` and its tests, not from
file-extension claims in the UI.

| Source | Code path | Beta status |
| --- | --- | --- |
| GGUF | GGUF header/tensor reader; validated tokenizer metadata; selected F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, and Q6_K paths | Supported; tested with synthetic and real-fixture coverage when fixtures exist |
| Safetensors | F32, F16, and BF16 shards; optional `config.json` and `tokenizer.json` sidecars | Supported; tested with synthetic local fixtures |
| PyTorch checkpoint | `torch.load(..., weights_only=True)` for `.pt`, `.pth`, and PyTorch `.bin` | Implemented conditionally on PyTorch; tested only when the local dependency/fixture is supplied |
| QWN | Native runtime input and post-conversion output | Runtime format; not a conversion source |
| ONNX, Keras/H5, arbitrary binary | No native reader or verified tensor ABI | Unsupported and must fail fast |

An extension is only a hint. The converter must reject ONNX/Keras and unknown
formats before creating a published output. A `.bin` file is accepted only if
it is a valid PyTorch checkpoint through the verified loader; an arbitrary
binary file is not a supported model format.

## Acquisition providers

Acquisition is explicit, local-first, and provider-scoped:

1. Hugging Face public artifacts use an adapter that constructs an HTTPS
   resolve URL from a repository, revision, and filename. The catalog contains
   metadata only; browsing or downloading does not silently fetch model data.
2. Direct HTTPS uses a user-supplied URL and requires HTTPS. HTTP is permitted
   only for an explicitly enabled loopback test server.
3. Local file import accepts an existing local path and copies it into the
   model-library boundary after canonical containment checks.

Provider manifests carry the provider, source identity, filename, expected
size when known, SHA-256 when published, format, license/gated state, and
whether the artifact is verified. Tokens are never persisted in a manifest or
written to a log. Gated Hugging Face artifacts require an explicit license
confirmation in the request.

All remote targets are checked for HTTPS, approved host policy, safe redirects,
and a destination filename that cannot escape the model library. Downloads
stream to `<name>.part`, resume only with a validated byte range, enforce a
maximum size and free-disk preflight, and publish with an atomic rename only
after size/checksum verification. Missing checksums are labeled unverified;
they are not silently promoted to verified. Cancellation removes the partial
artifact and leaves no usable model behind. Status exposes real bytes,
throughput, ETA when computable, pause/resume state, retry count, and the
failure reason.

### Official Qwen3.8 presets

The gateway exposes `/v1/qwanto/model-presets` and
`/v1/qwanto/download/preset` for two explicit Hugging Face choices:

- `qwen38-flash-next-ud-q4-k-xl`: the official-agent Flash-Next UD-Q4_K_XL
  bundle from `unsloth/Qwen3.8-Flash-Next-GGUF`, four GGUF shards, about 111 GB.
- `qwen38-27b-q4-k-m`: the lightweight fallback Q4_K_M artifact from
  `ggml-org/Qwen3.8-27B-GGUF`, one GGUF file, about 19 GB.

The bundle downloader writes each shard atomically and records completion in
`.qwanto-bundle.json`, allowing completed shards to be retained across an
interrupted download. The official listings do not provide checksums in the
catalog metadata, so the resulting source bundle is labeled `unverified` and
`external_source_only`. It is not presented as a native QWN model and cannot
be activated by `qwnrun` until the matching architecture runtime is complete.

## Desktop packaging design

Beta installers package the native `qwnrun` target only. They do not contain
model weights, Python, a hidden inference service, or a claim of signing. The
Tauri UI therefore reports converter/downloader capabilities from the connected
host and presents them as unavailable when the Python gateway capability is not
present. A future gateway sidecar must be separately packaged, started with a
fixed executable path and environment, bound to loopback, health-checked, and
stopped with the desktop process before these controls can be enabled.

The package workflow builds Windows NSIS and MSI, macOS DMG, and Linux AppImage
and DEB. Manual dispatch is for package validation and stores workflow
artifacts. A version tag publishes the already-built installers as GitHub
Release assets only after every matrix job succeeds. The first release is a
pre-release Beta and remains unsigned unless real signing credentials and
verification are configured.

## Safe model lifecycle

The lifecycle is deliberately staged:

`request -> provider manifest -> download/import -> inspect -> verify -> compatibility check -> convert (if supported) -> temporary .qwn.part -> QWN validation -> atomic rename -> native smoke test -> explicit approval/load -> live telemetry`

Inspection and compatibility checks are metadata-only until the user starts a
conversion or runtime load. Conversion manifests record the source identity,
source SHA-256 when available, converter version, quantization mode, output
SHA-256, validation result, and verification classification. Existing files are
never overwritten without an explicit confirmation.

Hardware-aware size and runtime estimates may use locally reported CPU, memory,
GPU, and disk facts, but every estimate is labeled as an estimate. No synthetic
throughput, TTFT, GPU, disk, or download values are displayed. Unavailable
sensors carry an explicit reason. Native telemetry uses qwnrun DONE frames for
tokens and throughput, submit-to-first-DATA for TTFT, the active model/backend,
PID, and only real host sensor values.

## Acceptance criteria

- No public-network call occurs during CI; acquisition tests use a local HTTP
  server and explicit loopback test mode.
- HTTPS, host allowlists, redirect validation, path containment, gated-license
  confirmation, max-size/free-space checks, checksum mismatch, range resume,
  cancellation cleanup, pause/resume, retry, and atomic publication are tested.
- GGUF, Safetensors, and conditional PyTorch support are shown distinctly;
  ONNX, Keras/H5, and arbitrary binary input fail clearly.
- A converted artifact is never published before QWN header, descriptor,
  alignment, tensor bounds, and checksum validation, and a native smoke test is
  required before approval/load.
- Desktop Model Library states are explicit for Providers, Discover, Download
  queue, Local models, Converter, Runtime, and Benchmark. Missing gateway
  capabilities show an actionable unavailable/error state rather than a working
  looking control.
- CI retains NumPy, Tauri Linux dependencies, native decoder/protocol tests,
  real-fixture conditional skips, and strict benchmark evidence classifications.
