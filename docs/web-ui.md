# Web UI and gateway boundary

The React application in `web/` is the only Qwanto frontend. It can be served
by Vite, by the local gateway from `web/dist`, or inside the Tauri window.

## Connection flow

1. The default endpoint is `http://127.0.0.1:8000/v1` during browser/Vite
   development. Gateway-served builds use the current origin and `/v1`.
2. “Probe server” calls `/v1/models` and `/health`. A connected badge means
   those HTTP calls succeeded; it does not claim that a model is loaded.
3. Chat, model management, diagnostics, telemetry, and benchmark views use
   the configured endpoint. The API key is held in memory and sent only to
   that endpoint.
4. Values not supplied by the gateway are rendered as `Unavailable`.

## Browser versus desktop

Browser mode is intentionally limited to HTTP. It cannot invoke Tauri IPC,
read arbitrary local paths, launch a process, or approve desktop agent tools.
The desktop window still uses this same UI, but its native Rust commands are a
separate capability boundary documented in [desktop-agent.md](desktop-agent.md).

The UI removes the ambiguous web-search affordance from the local chat
composer. External search/fetch capabilities, when present in the Python
agent runtime, are not silently implied by browser chat.

## Honest status display

The dashboard and benchmark view do not contain host-specific fallback values.
Throughput, TTFT, VRAM, NVMe bandwidth, temperatures, and hardware names are
shown only from a live gateway response or a benchmark evidence artifact.
Benchmark evidence carries its classification (`MEASURED`, `UNAVAILABLE`,
`INVALID`, `TEST_FIXTURE`, `EXPERIMENTAL`, or `PROJECTED`) into the UI.
