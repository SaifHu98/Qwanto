# Qwanto Native — Qwanto Code desktop surface

Qwanto Code is the desktop agent surface of Qwanto Native. It is a Tauri v2
application around the shared React UI in `../web`. It adds native runtime and
approval-gated agent commands through the Rust host; it does not create a
second frontend.

The packaged application contains the target-native `qwnrun` executable and
gateway sidecar as resources. It never contains a model. Users select local
`.qwn` containers; the Rust host starts the sidecar on loopback using a dynamic
port, waits for its structured ready handshake, and starts `qwnrun` only after
a validated model is selected.
The browser build cannot invoke these Tauri commands or access arbitrary local
files and processes.

## Development

Build the native resource first, then run the shared UI in the desktop shell:

```sh
make -C c qwnrun
mkdir -p desktop/src-tauri/resources
cp c/qwnrun desktop/src-tauri/resources/qwnrun
cd web && npm ci && npm run build
cd ../desktop
cargo install tauri-cli --version "^2.0.0" --locked
cargo tauri dev
```

On Windows, build `c/qwnrun.exe` with the supported native toolchain and copy
it to `desktop/src-tauri/resources/qwnrun`. The release workflow performs this
target-specific staging and freezes `c/openai_server.py` into the matching
gateway sidecar automatically.

## Agent safety boundary

The desktop agent has Plan and Agent modes. The canonical workspace is set by
the user. Reads and directory inspection are read-only; writes, edits, command
execution, staging, and commits require a short-lived approval token bound to
the session, arguments, workspace, and execution mode. Commands use explicit
argv and do not run through a shell. Secrets are redacted from tool output and
persisted sessions.

## Validation

```sh
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo check --manifest-path src-tauri/Cargo.toml
cargo test --manifest-path src-tauri/Cargo.toml
cargo clippy --manifest-path src-tauri/Cargo.toml -- -D warnings
```

Release packaging requires `desktop/src-tauri/resources/qwnrun` and
`desktop/src-tauri/resources/qwanto-gateway`, and produces Windows NSIS/MSI,
macOS DMG, or Linux AppImage/deb packages depending on the runner. Models and
benchmark artifacts are intentionally excluded.
