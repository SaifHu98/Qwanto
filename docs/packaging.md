# Qwanto packaging

The release workflow is `.github/workflows/release.yml`. It runs only after a
maintainer pushes a `v*` tag; it does not create tags or GitHub releases.

## Package contents

Each target stages its own compiled `qwnrun` binary at
`desktop/src-tauri/resources/qwnrun` before Tauri bundling. The resource is
resolved by the desktop runtime from the application resource directory. No
`.qwn`, `.gguf`, `.safetensors`, or other model file is bundled.

| Runner | Native build | Tauri output |
| --- | --- | --- |
| Windows x64 | Clang native `qwnrun.exe` | NSIS installer and MSI |
| macOS runner | `make -C c qwnrun` | DMG; experimental until observed on the target host |
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

The workflow runs web tests/build, compiles the native engine, builds Tauri,
checks that the resource exists, rejects model files in the bundle, and runs
`git diff --check`. Supported artifact files are uploaded as workflow
artifacts. Publishing a release remains a deliberate maintainer action.
