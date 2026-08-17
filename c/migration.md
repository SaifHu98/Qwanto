# Qwanto API Migration Guide

## Runtime selection

The gateway accepts `--backend auto|native|qwn` for compatibility with
existing launch scripts. All three values select the same QWN-native runtime
boundary; `auto` does not inspect a source model and does not select an
external service. A model must be an existing, validated `.qwn` container.

GGUF, Safetensors, and PyTorch files are conversion inputs only. Convert them
locally, validate the generated QWN metadata and tensor descriptors, then
activate the resulting `.qwn` artifact. Source files cannot be served or
executed by the gateway.

## API behavior

The loopback gateway continues to expose the OpenAI-compatible local API.
Requests that attempt to load or execute a non-QWN source return the
structured `qwn_required` error instead of forwarding to llama.cpp, Ollama,
OpenAI, or another external backend. Optional web search and GitHub actions
remain explicit agent tools, never inference backends.
