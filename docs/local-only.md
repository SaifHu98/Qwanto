# 🔒 Qwanto Local-Only Profile Specification

## 1. Overview

By default, Qwanto operates in a **production local-only profile**. This mode ensures complete data sovereignty and air-gapped execution for sensitive workloads.

---

## 2. Permitted vs. Forbidden Operations

| Operation Category | Default (Local-Only Profile) | With `--allow-external-runtime` |
|---|:---:|:---:|
| Native `.qwn` Model Inference (`qwnrun`) | ✅ **Permitted** (Local C SIMD) | ✅ **Permitted** |
| Local Memory Tiering (VRAM / RAM / NVMe mmap) | ✅ **Permitted** (Zero-Copy) | ✅ **Permitted** |
| Localhost HTTP Gateway (`127.0.0.1:8000`) | ✅ **Permitted** | ✅ **Permitted** |
| Local Web Dashboard (`web/`) & Desktop Shell | ✅ **Permitted** | ✅ **Permitted** |
| Automatic Download of `llama-server` Binaries | ❌ **Forbidden** (Fails Safely) | ✅ Allowed with explicit user authorization |
| External Cloud Model Forwarding (`api.openai.com`) | ❌ **Forbidden** (Rejected) | ✅ Allowed with explicit user credentials |
| Remote Ollama Pull Requests (`ollama pull`) | ❌ **Forbidden** | ✅ Allowed if Ollama is running locally |
| Telemetry Export / Usage Analytics | ❌ **Forbidden** (Zero Egress) | ❌ **Forbidden** (Never Collected) |

---

## 3. Runtime Contract & Error Responses

When an external runtime feature is requested while running under the default local-only profile, Qwanto returns a structured error rather than attempting automatic network downloads:

```json
{
  "error": {
    "message": "[local-only] Automatic download of external runtimes is disabled in default local-only profile. Use native .qwn models with qwnrun or supply local llama-server on PATH.",
    "type": "permission_error",
    "code": "external_runtime_disabled"
  }
}
```

---

## 4. Enabling External Runtime (Opt-In)

If external runtime compatibility (such as downloading GGUF helper binaries) is explicitly desired:

### CLI Flag:
```bash
python c/openai_server.py --model models/model.gguf --allow-external-runtime
```

### Environment Variable:
```bash
export QWANTO_ALLOW_EXTERNAL_RUNTIME=1
```

---

## 5. Network Interface Binding

- Default binding is strictly `127.0.0.1`.
- To bind across a private local area network, explicitly specify:
  ```bash
  python c/openai_server.py --model models/model.qwn --host 192.168.1.100 --api-key <YOUR_SECRET_KEY>
  ```
  *(Note: A strong `QWANTO_API_KEY` is strongly required when binding to non-loopback interfaces.)*
