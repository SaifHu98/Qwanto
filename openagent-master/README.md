# ⚡ Qwanto Native — Offline Local AI Desktop Coding Agent

**Qwanto Native** is a production-grade, 100% offline, privacy-first desktop coding agent powered exclusively by the **Qwanto C Runtime Engine** (`qwnrun`). It provides developers with an intelligent coding assistant that runs locally on consumer and workstation hardware without transmitting a single byte over the public internet.

---

## 🔒 100% Air-Gapped Privacy & Security Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               React 19 Desktop Interface                               │
│  - Workspace Explorer · Conversation Timeline · Dual-Mode Switch (Plan / Agent)       │
│  - Interactive Tool Approval Gates · Diff Viewer · Live Hardware Command Center        │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Typed IPC Preload Bridge (window.qwanto)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              Electron Main Process (Node.js)                           │
│  ┌───────────────────────────┐ ┌───────────────────────────┐ ┌────────────────────────┐│
│  │   QwantoRuntimeAdapter    │ │     PermissionPolicy      │ │      ToolExecutor      ││
│  │ - Supervises qwnrun proc  │ │ - Out-of-bounds guards    │ │ - read/write/edit_file ││
│  │ - Manages .qwn models     │ │ - Risky command prompts   │ │ - glob, grep, list_dir ││
│  │ - Streams generation tok/s│ │ - Secret redaction filter │ │ - bash, git operations ││
│  └───────────────────────────┘ └───────────────────────────┘ └────────────────────────┘│
│  ┌───────────────────────────┐ ┌───────────────────────────┐ ┌────────────────────────┐│
│  │     HardwareProbe &       │ │       SessionStore        │ │     LocalApiServer     ││
│  │    TelemetryCollector     │ │ - Local SQLite/JSON store │ │ - 127.0.0.1:8000 only  ││
│  │ - NVIDIA NVML / CPU loads │ │ - Plan & step persistence │ │ - OpenAI compatibility ││
│  └───────────────────────────┘ └───────────────────────────┘ └────────────────────────┘│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Local Stdio Subprocess (Zero Network)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Qwanto Native C Runtime (qwnrun.exe)                            │
│  - AMD Ryzen 9 (AVX-VNNI) + NVIDIA RTX 5070 Ti (BitDecoding Tensor Cores) + NVMe mmap │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚫 Removed Cloud & Telemetry Dependencies

To guarantee absolute data privacy and zero runtime network egress, the following upstream dependencies and components were completely removed or replaced:

1. **Removed Cloud Model Providers**:
   - `Anthropic Claude Agent SDK` (`@anthropic-ai/claude-agent-sdk`)
   - `OpenRouter API` (`https://openrouter.ai/api`)
   - `Ollama HTTP Client`
2. **Removed Remote Auto-Updater**:
   - `electron-updater` GitHub Releases pollers and background downloaders.
3. **Removed External CDNs and Telemetry**:
   - Remote script injectors, external CSS fonts, error tracking beacons, and cloud OAuth flows.
4. **Replaced With 100% Local Modules**:
   - `QwantoRuntimeAdapter`: Supervises local `qwnrun.exe` over stdio.
   - `PermissionPolicy`: Strict local filesystem boundaries with secret redaction.
   - `HardwareProbe`: Local CIM/NVML hardware interrogation.
   - `SessionStore`: Fully local JSON session persistence in `userData`.

---

## 🛡️ Central Permission Model & Execution Modes

### 1. 🛡️ Plan Mode
- **Zero Mutations Permitted**: The agent cannot modify files or execute shell commands without prior explicit approval.
- The agent analyzes the codebase, generates an execution plan, and awaits user review.

### 2. ⚡ Agent Mode
- **Autonomous Execution with Boundary Guards**:
  - `read_file`, `list_directory`, `glob`, `grep`, `git status`: Auto-permitted inside workspace.
  - `write_file`, `edit_file`: Permitted within workspace root; audited in local logs.
  - `Dangerous Actions` (`rm -rf`, `git push --force`, out-of-workspace writes): Require user confirmation card with diff preview.

---

## 📊 Live Hardware Telemetry

The top command center monitors real hardware resources:
- **NVIDIA GeForce RTX 5070 Ti (12GB GDDR6)**: VRAM allocation bar, GPU compute saturation, temperature, BitDecoding Tensor Cores.
- **AMD Ryzen 9 9955HX (16 Cores, 32 Threads)**: 32-thread visual load meter, 5.40 GHz boost, AVX-VNNI / AVX-512 vector acceleration.
- **Samsung PM9A1a NVMe SSD**: Zero-copy mmap streaming bandwidth meter.
- **Live Throughput**: Real-time generation tok/s and TTFT latency.

---

## 📦 Model Support & Ingestion

Qwanto Native directly executes `.qwn` and `.gguf` containers:
- **TWLA 1.58-Bit**: Post-training ternary weights for ultra-low memory footprint.
- **HyperVSQ-2**: Vector-quantized weights for maximum accuracy and throughput.
- **LittleBit-2**: Sub-1-bit latent factorization (<0.6 GB RAM).
- **pQuant**: Decoupled dominant 1-bit + high-precision branch quantization.

---

## 🚀 Quick Start

### 1. Build and Run Electron App
```bash
# Build desktop app
npm run build

# Start Qwanto Native
npm start
```

### 2. Run Test Suite
```bash
# Run unit & security tests
npm test
```

### 3. Optional Localhost OpenAI-Compatible Server
- Toggle **Local API Server** in the Command Center or start via port `8000`.
- Connect any compatible client to `http://127.0.0.1:8000/v1`.

---

## 📄 License
Licensed under the Apache License 2.0.
