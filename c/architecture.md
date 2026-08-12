# Universal Backend Architecture

The Colibri Universal Backend Layer decouples the monolithic engine runtime from the HTTP presentation layer, enabling multiple model-execution backends while retaining identical API contracts.

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

## Included Adapters

1. **`NativeBackend`**: 
   - Wraps the high-performance Colibri C engine (`Engine`).
   - Retains direct process pipeline reading, zero-copy overhead, and memory pinning.
2. **`OpenAICompatibleBackend`**:
   - HTTP-based connection-pooling adapter.
   - Forwards JSON payloads transparently while bubbling up exact upstream OpenAI error schemas.
3. **`OllamaBackend` & `LlamaCppBackend`**:
   - Subclasses of the OpenAI adapter enforcing specific capabilities (e.g., overriding `tool_calls=False` for Ollama default).

## Safety Controls
- **Recursive Routing Prevention**: The gateway blocks downstream backend URLs that resolve to the gateway's own bound IP and port via DNS and loopback validation.
- **Connection Polling**: `http.client` timeouts are enforced to prevent hanging the gateway if an external backend freezes.
