# Qwanto Native

Qwanto is a Beta local-first AI runtime for running supported dense transformer
models through a native C decoder, a loopback OpenAI-compatible gateway, a
shared React dashboard, and an optional Tauri desktop host.

The repository is engineering-complete for its tested paths, but it is not a
promise that every model, GPU backend, sensor, or platform is production-ready.
Read the status and evidence before relying on a path.

## Architecture

```mermaid
flowchart LR
  Browser[web/ React UI] -->|HTTP| Gateway[c/openai_server.py]
  Desktop[desktop/ Tauri shell] -->|same shared UI| Browser
  Desktop -->|native commands and approvals| Rust[desktop/src-tauri]
  Gateway -->|persistent stdin/stdout| Qwnrun[c/qwnrun --serve]
  Qwnrun -->|mmap / prefetch| QWN[(user-managed .qwn model)]
```

The browser UI has no filesystem or terminal authority. The desktop host may
start the packaged target-native `qwnrun` resource and expose approval-gated
agent tools. Models are never bundled.

## Current support matrix

| Surface | Status | Validation |
| --- | --- | --- |
| Native decoder and persistent serve protocol | Beta-supported | C/Python tests and CI |
| Loopback gateway and OpenAI-compatible API | Beta-supported | `c/tests/` |
| Shared web dashboard | Beta-supported | `npm run build`, `npm test` |
| Windows NSIS/MSI, macOS DMG, Linux AppImage/DEB | Package workflow; unsigned unless real signing is configured | `.github/workflows/release.yml` |
| GGUF, Safetensors, PyTorch `.pt`/`.pth`/PyTorch `.bin` | Converter-supported source formats; fixture coverage is conditional | [model acquisition design](docs/model-acquisition-design.md) |
| ONNX, Keras/H5, arbitrary `.bin` | Unsupported; converter fails fast | [model acquisition design](docs/model-acquisition-design.md) |

## Quick start

Install Python test dependencies and native tools appropriate to your platform.

```sh
python -m pip install pytest numpy
make -C c qwnrun
python c/coli web --model path/to/model.qwn
```

The gateway defaults to `http://127.0.0.1:8000`. Open the gateway-served UI
when `web/dist` exists, or run the development UI separately:

```sh
cd web
npm ci
npm run dev
```

For the desktop shell, stage the target-native runtime first:

```sh
mkdir -p desktop/src-tauri/resources
cp c/qwnrun desktop/src-tauri/resources/qwnrun
cd desktop
cargo tauri dev
```

Use `qwnrun.exe` as the source on Windows; the staged resource name remains
`qwnrun`. See [desktop/README.md](desktop/README.md) and
[docs/packaging.md](docs/packaging.md) for package builds.

## Model and container rules

Native inference consumes `.qwn` containers with a 4 KiB header, validated
descriptors, 64-byte payload alignment, and the project’s VRAM → RAM → NVMe
memory model. Inspect [docs/qwn-format.md](docs/qwn-format.md) before creating
or distributing a container. Large model files are local fixtures and are not
source-controlled release assets.

Model acquisition is explicit and provider-scoped. Hugging Face public HTTPS,
allowlisted direct HTTPS, and local-file import are supported by the local
gateway. Downloads use a `.part` file, range resume, checksum/size checks, disk
preflight, and atomic publication. The packaged Tauri Beta contains qwnrun only;
its converter and downloader are honestly disabled until a gateway sidecar is
packaged and supervised.

## Benchmark evidence

Run a real local benchmark; do not copy values from another host:

```sh
python benchmarks/benchmark_reproducible.py \
  --model experiments/results/4B_hyper_vsq2.qwn \
  --executable c/qwnrun \
  --max-tokens 64 \
  --output benchmark_evidence.json
```

| Category | Meaning | UI/reporting rule |
| --- | --- | --- |
| `MEASURED` | Successful real qwnrun with positive tokens and monotonic timing | May display measured values |
| `UNAVAILABLE` | Missing executable/model/sensor or timed out | Display unavailable |
| `INVALID` | Nonzero exit, malformed output, or zero tokens | Never display throughput |
| `TEST_FIXTURE` | Explicit parser/UI test data | Never a production claim |
| `EXPERIMENTAL` | Real but outside the native comparison boundary | Label the boundary |
| `PROJECTED` | Estimate or planning value | Never call measured |

See [docs/benchmark-methodology.md](docs/benchmark-methodology.md) and
[docs/model-manifest.json](docs/model-manifest.json). The repository does not
ship a universal hardware profile or fallback tok/s value.

## Security and local-only behavior

- The gateway binds to loopback by default and preserves security headers.
- API keys are optional for loopback and are compared in constant time when
  configured; non-loopback operation should use a strong key and narrow CORS.
- External runtime downloads and remote backends are explicit opt-ins.
- Desktop agent paths stay inside the canonical workspace. Mutations,
  commands, staging, and commits require approval tokens.
- Browser chat cannot launch a process or read arbitrary local files.

Read [docs/security-model.md](docs/security-model.md) and
[docs/local-only.md](docs/local-only.md) for the detailed boundary.

## Validation

```sh
python -m pytest c/tests/ -q
make -C c test-c
cargo check --manifest-path desktop/src-tauri/Cargo.toml
cargo test --manifest-path desktop/src-tauri/Cargo.toml
cargo clippy --manifest-path desktop/src-tauri/Cargo.toml -- -D warnings
cd web && npm run build && npm test
```

CI preserves the Linux Tauri packages (`libwebkit2gtk-4.1-dev`,
`libsoup-3.0-dev`, and `pkg-config`) and installs NumPy for conversion tests.
The release workflow packages only after the native resource and web build are
available.

## Documentation map

- [Architecture](docs/architecture.md)
- [QWN format](docs/qwn-format.md)
- [Web UI boundary](docs/web-ui.md)
- [Desktop agent](docs/desktop-agent.md)
- [Security model](docs/security-model.md)
- [Packaging](docs/packaging.md)
- [Model acquisition design](docs/model-acquisition-design.md)
- [Benchmark methodology](docs/benchmark-methodology.md)
- [Release engineering plan](docs/release-engineering-plan.md)
- [Release readiness](RELEASE_READINESS.md)

Qwanto preserves attribution to the upstream Colibrì multi-tier memory work.
The project is licensed under Apache 2.0; see [LICENSE](LICENSE).
