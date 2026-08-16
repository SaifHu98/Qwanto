# Desktop runtime and coding agent

The Tauri host is a native wrapper around the shared web UI. It is the only
surface allowed to start the packaged `qwnrun` resource or execute local agent
tools.

## Runtime lifecycle

`QwantoRuntimeManager` locates `qwnrun` beside the executable, in the Tauri
resource directory, or in the documented development paths. It validates a
`.qwn` model path, starts `qwnrun --serve`, and parses READY, PONG, CONFIG,
DATA, DONE, and ERROR frames. One process remains alive for multiple prompts.

Release builds stage a target-native binary as
`desktop/src-tauri/resources/qwnrun`. Models are not resources; they stay on
user-selected storage.

The Beta package deliberately does not bundle Python or the gateway. The
shared Model Library therefore disables remote acquisition and conversion in
the installed shell and explains the boundary. The local gateway web console
may expose the provider adapters described in
[model-acquisition-design.md](model-acquisition-design.md). Enabling those
controls in the desktop shell requires a future loopback-only, health-checked
gateway sidecar with a fixed packaged executable path.

## Agent modes and tools

The Rust policy supports Plan and Agent modes. Read-only inspection is
available for planning. File writes, edits, command execution, staging, and
commits require an approval token tied to the exact session, tool, arguments,
workspace, mode, and expiry.

Commands are passed as an argv vector to `std::process::Command`; no shell
interpolation is used. Paths and working directories are canonicalized against
the configured workspace. Output is bounded and secret-redacted before being
returned or persisted.

The browser cannot access these commands. A browser can call the HTTP gateway,
but that is not desktop-agent permission.
