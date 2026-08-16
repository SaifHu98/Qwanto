# Qwanto packaging

The release workflow is `.github/workflows/release.yml`. `workflow_dispatch`
builds packages for validation and uploads workflow artifacts. A temporary
`package-validation-*` tag provides the same non-publishing validation path
when dispatch is unavailable. A `v*` tag runs
the same matrix and, only after all package jobs pass, creates or updates a
GitHub prerelease with the installer assets. The release publisher never
creates a tag; maintainers create and push the annotated version tag after
hosted checks pass.

## Package contents

Each target stages its own compiled `qwnrun` binary at
`desktop/src-tauri/resources/qwnrun` and freezes the Python gateway into
`desktop/src-tauri/resources/qwanto-gateway` before Tauri bundling. Both are
resolved from the application resource directory. No `.qwn`, `.gguf`,
`.safetensors`, or other model file is bundled.

| Runner | Native build | Tauri output |
| --- | --- | --- |
| Windows x64 | Clang native `qwnrun.exe` + frozen gateway | NSIS installer and MSI |
| macOS runner | `make -C c qwnrun` + frozen gateway | DMG |
| Ubuntu 22.04 | `make -C c qwnrun` + frozen gateway | AppImage and Debian package |

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
the bundle, and runs `git diff --check`. `workflow_dispatch` and
`package-validation-*` are package validation gates. For a Beta release, wait
for hosted CI and package checks to pass, then create a version tag such as
`v0.1.0-beta.4` and push it; the tag-triggered workflow publishes a prerelease
only after every Windows, macOS, and Linux package job is green.

## Branding

`assets/brand/qwanto-icon.png` is the approved 512x512 source mark. The Tauri
PNG/ICO/ICNS files and `web/public/qwanto-icon.png` are checked against the
source hash by `c/tools/check_brand_assets.py`; the same check runs in CI and
the package workflow. Do not introduce a second lettermark or favicon.

## Signing policy

Packaging and signing are separate gates. The `release-signing` environment
uses `SIGNING_ENABLED=true` plus the platform-specific protected variables
below to opt into real verification. When the global or platform credentials
are absent, packaging succeeds and the release is explicitly unsigned; no
self-signed certificate is created and no unsigned artifact is sent to
SignTool verification.

| Platform | Enable variable | Protected credentials and verification |
| --- | --- | --- |
| Windows | `SIGNING_ENABLED=true` and `QWANTO_WINDOWS_SIGNING_ENABLED=true` | Azure Artifact Signing OIDC identity, account/profile variables; all staged sidecars and EXE/MSI files receive timestamped Authenticode signatures and `SignTool verify /pa /all /tw` must pass. |
| macOS | `SIGNING_ENABLED=true` and `QWANTO_MACOS_NOTARIZATION_ENABLED=true` | Developer ID certificate, keychain password, Apple notarization credentials; nested Mach-O files, app, and DMG are signed, submitted, stapled, then checked with `codesign` and `spctl`. |
| Linux | `SIGNING_ENABLED=true` and `QWANTO_LINUX_SIGNING_ENABLED=true` | Protected GPG private key and key ID; AppImage, DEB, and the Linux SHA256SUMS file receive detached ASCII-armored signatures and `gpg --verify` runs in CI. |

The Windows implementation uses the official [Azure Artifact Signing
Action](https://github.com/Azure/artifact-signing-action). Credentials belong
only in the protected `release-signing` GitHub Environment. If a signing
variable is set to `true` but a required credential or verification step fails,
the package job fails. Otherwise the release summary reports:

```text
Windows signing: UNSIGNED
macOS notarization: UNSIGNED / NOT NOTARIZED
Linux artifact signature: UNSIGNED
```

The unsigned beta note warns about Windows SmartScreen and macOS Gatekeeper
and directs users to verify the SHA-256 checksum before installing.

Packages contain the native runtime and gateway sidecar, but no model weights.
The sidecar is bound to loopback, starts with hidden child-process flags on
Windows, and is supervised by the desktop host. Packages are not described as
signed unless real signing and verification have been configured.
