# Desktop runtime and coding agent

The Tauri host is a native wrapper around the shared web UI. It is the only
surface allowed to start the packaged `qwnrun` resource or execute local agent
tools.

## Runtime lifecycle

`QwantoRuntimeManager` locates `qwnrun` beside the executable, in the Tauri
resource directory, or in the documented development paths. It validates a
`.qwn` model path, starts `qwnrun --serve`, and parses READY, PONG, CONFIG,
DATA, DONE, and ERROR frames. One process remains alive for multiple prompts.

Release builds stage target-native binaries as
`desktop/src-tauri/resources/qwnrun` and
`desktop/src-tauri/resources/qwanto-gateway`. Models are not resources; they
stay on user-selected storage.

The Rust host starts the gateway sidecar on `127.0.0.1` with port `0`, a
dedicated model root, and a temporary ready file. The sidecar prints one
`QWANTO_GATEWAY_READY {json}` line containing the bound port and URL, and the
host also verifies the loopback URL before passing it to the web UI. A missing,
invalid, or crashed sidecar is shown as a failed desktop status; it is never
silently replaced by a remote service.

On Windows, the gateway, `qwnrun`, converter, downloader, and integrated agent
commands keep their standard streams piped into the host and use
`CREATE_NO_WINDOW`. Shutdown terminates the supervised process tree, including
the hidden `taskkill` cleanup helper. Gateway failures stay in the app with
Open logs and Restart gateway actions; no console or Windows Terminal window is
used for internal work.

The shared Model Library keeps conversion and download controls in Settings.
They require explicit user consent, disk/source/checksum validation, and use
the sidecar’s local acquisition paths described in
[model-acquisition-design.md](model-acquisition-design.md).

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

## Local memory and resume

Project memory is stored at `.qwanto/project-memory.json` in the selected
workspace. It contains only user-reviewable summary, architecture notes,
conventions, decisions, and task checkpoints. The desktop session store keeps
redacted local checkpoints so an interrupted conversation can be resumed from
the Project view. Both stores are local-only and can be edited, exported,
cleared, or disabled; neither is uploaded as an inference fallback.
