# Security Policy — Qwanto

## 🔒 Local-Only Security Model

Qwanto is engineered as a **privacy-first, local-only AI execution runtime**. By default, all model inference, data processing, prompt caching, and hardware telemetry remain strictly contained on the host machine.

### Core Security Guarantees:
1. **Zero External Network Egress**: In default execution mode, Qwanto initiates no outbound network connections, analytics beacons, update checks, remote telemetry exports, or cloud model fallbacks.
2. **Strict Localhost Binding**: The HTTP gateway (`c/openai_server.py`) binds exclusively to loopback interfaces (`127.0.0.1` / `::1`) by default. Binding to `0.0.0.0` or external network interfaces requires explicit user configuration.
3. **No Automatic Binary Downloads**: Automatic downloads of external runtime binaries (e.g. `llama-server`) are disabled by default. External runtime capabilities are strictly opt-in via `--allow-external-runtime`.
4. **Filesystem Boundary Isolation**: All model loading and file ingestion operations are guarded by `_is_safe_path()` checks to prevent directory traversal (`../`) attacks.
5. **Constant-Time Authentication**: When `QWANTO_API_KEY` is configured, HTTP Bearer tokens are validated using constant-time string comparison (`secrets.compare_digest`) to prevent timing side-channel attacks.
6. **HTTP Defense-in-Depth Headers**: The gateway automatically attaches strict defense headers to all HTTP responses:
   - `X-Content-Type-Options: nosniff`
   - `X-Frame-Options: DENY`
   - `Referrer-Policy: no-referrer`
   - `Content-Security-Policy: default-src 'self'`
7. **Safe Subprocess Execution**: All engine subprocesses are invoked using structured argument vectors (`subprocess.Popen([arg1, arg2, ...])`) rather than shell string interpolation, eliminating shell command injection vectors.

---

## 🛡️ Supported Versions

Only the latest release on the `main` branch receives active security updates.

| Version | Supported | Notes |
|---|:---:|---|
| `main` (Latest) | ✅ Yes | Actively maintained with security patches |
| Legacy branches | ❌ No | Please upgrade to the latest `main` commit |

---

## 🚨 Reporting a Vulnerability

If you discover a security vulnerability or unexpected network egress in Qwanto:

1. **Do not create a public GitHub issue.**
2. Send a detailed report via private email to:
   - **Maintainer**: `saifhu98@github.com`
3. Please include:
   - Description of the vulnerability or network leak.
   - Steps to reproduce (proof-of-concept script or command).
   - Potential impact and affected components.
4. **Response SLA**: We acknowledge security reports within **24 hours** and aim to provide a remediation patch within **72 hours**.
