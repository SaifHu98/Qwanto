# Qwanto Native local-first runtime boundary

## 1. Overview

By default, Qwanto operates in a **production local-only profile**. This mode ensures complete data sovereignty and air-gapped execution for sensitive workloads.

---

## 2. Permitted vs. Forbidden Operations

| Operation Category | Qwanto Native policy |
|---|:---:|
| Native `.qwn` Model Inference (`qwnrun`) | ✅ **Permitted** after container and hardware-fit validation |
| Local Memory Tiering (RAM / NVMe mmap; CUDA only with measured kernel work) | ✅ **Permitted** |
| Localhost HTTP Gateway (`127.0.0.1`) | ✅ **Permitted** |
| Local Web Console and Qwanto Code | ✅ **Permitted** |
| GGUF, Safetensors, and PyTorch inference | ❌ **Forbidden**; conversion inputs only |
| Automatic external runtime download or fallback | ❌ **Forbidden** |
| Cloud model forwarding and Ollama inference | ❌ **Forbidden** |
| Optional web search and GitHub actions | ✅ Explicit external tool approval required |
| Telemetry export / usage analytics | ❌ **Forbidden** by default |

---

## 3. Runtime Contract & Error Responses

When a source artifact is sent to the runtime, Qwanto returns a structured error rather than attempting a fallback:

```json
{
  "error": {
    "message": "Only a validated .qwn model can serve requests; source artifacts are conversion inputs only.",
    "type": "permission_error",
    "code": "qwn_required"
  }
}
```

---

## Network Interface Binding

- Default binding is strictly `127.0.0.1`.
- To bind across a private local area network, explicitly specify:
  ```bash
  python c/openai_server.py --model models/model.qwn --host 192.168.1.100 --api-key <YOUR_SECRET_KEY>
  ```
  *(Note: A strong `QWANTO_API_KEY` is strongly required when binding to non-loopback interfaces.)*
