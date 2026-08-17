# QWN-native gateway architecture

The gateway decouples the HTTP presentation layer from the native runtime
while retaining the OpenAI-compatible API contract. It deliberately exposes
one model-execution boundary: a supervised `qwnrun` process loading a
validated `.qwn` container.

## Core Components

### `BackendCapability`
A typed configuration object allowing backends to report feature support dynamically:
- `streaming`
- `tool_calls`
- `structured_output`
- `reasoning`
- `cancellation`
- `model_discovery`

### `Backend` (Abstract Base Class)
The universal interface defining the contract:
- `chat_completions()`
- `completions()`
- `models()`
- `health_check()`
- `unload()`

## Runtime boundary

`NativeBackend` supervises `qwnrun` and reports its measured runtime
telemetry. Legacy adapter types may remain in the Python library for API
compatibility and unit-test coverage, but the gateway does not instantiate,
download, or forward to them. GGUF and other source formats must pass through
the local converter before activation.

## Safety Controls
- **Validated model boundary**: QWN metadata, descriptor bounds, payload bounds,
  and hardware fit are checked before activation.
- **Supervised process**: `qwnrun` uses piped protocol streams and is stopped
  with the gateway; no external model process is spawned.
- **Loopback policy**: the API binds to loopback by default and retains its
  authentication and HTTP defense-header controls.
