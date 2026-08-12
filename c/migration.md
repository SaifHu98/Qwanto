# Qwanto API Migration Guide

## Backend Selection

The `openai_server.py` gateway now supports formal backend selection via the `--backend` flag. 
By default (`--backend auto`), the server will examine the `--model` path and guess the appropriate backend.

### Available Backends:
- **`native`**: The high-performance C engine (default for directory paths).
- **`llama-cpp`**: Local llama.cpp server (default for `.gguf` files).
- **`ollama`**: Local Ollama instance (default for `hf.co/` prefixed models).
- **`openai`**: A remote or proxy OpenAI-compatible API endpoint.

### New Flags
- `--backend [name]`: Force a specific backend instead of auto-discovery.
- `--backend-url [url]`: Explicit URL for the remote or local API. Overrides the default port choices (`11434` for Ollama, `8080` for llama.cpp).

### Example Uses
```bash
# Force the Native engine
python openai_server.py --model ./glm --backend native

# Point to an external vLLM server disguised as OpenAI
python openai_server.py --model llama-3 --backend openai --backend-url http://10.0.0.5:8000/v1
```

## API Behavior Changes
- Unsupported operations (e.g., Tool calling via Ollama backend) will now explicitly return an OpenAI-shaped error payload:
  ```json
  {"error": {"message": "Invalid parameter 'tool_choice'", "type": "invalid_request_error"}}
  ```
- Connecting to a backend URL that resolves recursively back to the gateway's IP and port will immediately fail with a `recursive_routing` error rather than stalling the server.
