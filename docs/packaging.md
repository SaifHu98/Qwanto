# Qwanto packaging

The release workflow is `.github/workflows/release.yml`. `workflow_dispatch`
builds packages for validation and uploads workflow artifacts. A `v*` tag runs
the same matrix and, only after all package jobs pass, creates or updates a
GitHub prerelease with the installer assets. The workflow never creates a tag.

## Package contents

Each target stages its own compiled `qwnrun` binary at
`desktop/src-tauri/resources/qwnrun` before Tauri bundling. The resource is
resolved by the desktop runtime from the application resource directory. No
`.qwn`, `.gguf`, `.safetensors`, or other model file is bundled.

| Runner | Native build | Tauri output |
| --- | --- | --- |
| Windows x64 | Clang native `qwnrun.exe` | NSIS installer and MSI |
| macOS runner | `make -C c qwnrun` | DMG |
| Ubuntu 22.04 | `make -C c qwnrun` | AppImage and Debian package |

## Local build

```sh
make -C c qwnrun
mkdir -p desktop/src-tauri/resources
cp c/qwnrun desktop/src-tauri/resources/qwnrun
cd web && npm ci && npm run build
cd ../desktop
cargo tauri build --bundles appimage,deb
```

Use the target-native copy command on Windows. A package build without the
staged resource is intentionally a failure, because a desktop binary that
cannot start its declared runtime is not release-ready.

## Validation

The workflow runs web tests/build, Rust check/test/Clippy, compiles the native
engine, builds Tauri, checks that the resource exists, rejects model files in
the bundle, and runs `git diff --check`. `workflow_dispatch` is the package
validation gate. For a Beta release, wait for that gate to pass, then create a
version tag such as `v0.1.0-beta.1` and push it; the tag-triggered workflow
publishes a prerelease only after every Windows, macOS, and Linux package job
is green.

Packages contain qwnrun only. They do not contain model weights, Python, or a
gateway sidecar, and the UI labels converter/download capabilities unavailable
inside the installed shell. Packages are not described as signed unless real
signing and verification have been configured.
