# Qwanto documentation

Qwanto Native is a local-first native runtime with Qwanto Web as its safe
browser console and Qwanto Code as its Tauri desktop agent surface. Use this
index to choose the right boundary before running a command or importing a
model.

## Start here

- [Repository overview](../README.md)
- [Architecture](architecture.md)
- [Local-only behavior](local-only.md)
- [Security model](security-model.md)
- [Troubleshooting](troubleshooting.md)

## Desktop coding agent

- [Desktop runtime and agent](desktop-agent.md)
- [Skills and plugins](skills-and-plugins.md)
- [Web UI boundary](web-ui.md)
- [Packaging](packaging.md)
- [Release engineering plan](release-engineering-plan.md)
- [Native quality and benchmark gates](quality-and-benchmarking.md)
- [Release readiness](../RELEASE_READINESS.md)

The browser is a chat-only client. Project inspection, file access, terminal
execution, diffs, approvals, and agent tools are desktop-only capabilities.

## Models and gateway

- [API and gateway contract](api.md)
- [Model acquisition design](model-acquisition-design.md)
- [Conversion and acquisition guide](conversion.md)
- [QWN container format](qwn-format.md)
- [Benchmark methodology](benchmark-methodology.md)
- [QWN performance and quantization](performance.md)
- [Model manifest](model-manifest.json)

Models are user-managed and are never shipped in source archives or installers.
The desktop sidecar binds to loopback, selects a dynamic port, and reports its
ready URL through a structured handshake.
