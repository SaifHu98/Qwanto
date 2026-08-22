# AGENTS.md — Qwanto Project Rules

## Coding Rules
- Production-grade, idiomatic code; minimal changes; preserve existing APIs.
- Reuse existing patterns, modules, and helpers before introducing new abstractions.
- Never hardcode user-facing text — detect and reuse any i18n/locale system (currently none present in the repo; if localization becomes in-scope, introduce the smallest appropriate mechanism rather than scattering hardcoded strings).
- Validate input, enforce authorization boundaries, prevent path traversal / injection / XSS.
- Atomic DB/API transactions where relevant; bounded caches.
- Follow existing style per language (PEP 8 for Python, conventional C99/C11 for native, ESLint defaults for TS/React).

## Architecture Constraints
- **Do not break the tiered-memory model**: VRAM → RAM → NVMe mmap with layer-ahead prefetching is the core architectural principle.
- **Zero-overhead hot path** in native decoder: no `snprintf`, no `expf`/`powf`/`cosf`/`sinf`, no scalar bit-conversion, no hash lookups in per-token forward pass. All descriptors resolved at load time.
- **`.qwn` container invariants**: 4KiB header, 4KiB-aligned tensor payloads, 64-byte padding, tail-block offset in last 8 bytes.
- **OpenAI compatibility**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, SSE streaming, `QWANTO_API_KEY` bearer auth.
- **No second UI**: web dashboard (`web/`) is the single source of truth; Tauri (`desktop/`) only packages it. Browser surface stays chat-only; Qwanto Code (desktop) is the only surface that owns the agent workspace.
- **No second localization system**.
- **Fixed bottom composer**: in Qwanto Code, the chat input bar is sticky at the bottom of the main column (`desktop-composer`). The center panel and inspector must reserve bottom padding so the composer never covers content. This pattern is desktop-only; the browser chat surface keeps its in-card composer.

## Commands
- Python tests: `python -m pytest c/tests/ -q`
- Native tests: `make -C c test-c`
- Build dashboard: `cd web && npm install && npm run build`
- Convert model: `python c/coli pack <input> <output.qwn> --quant <q4_0|q8_0|none|...>`
- Serve: `python c/coli web --model <path>`
- Build native: `make -C c qwnrun` (CUDA optional: rebuild for CUDA backend)

## Testing
- Run `python -m pytest c/tests/ -q` after Python changes.
- Run `make -C c test-c` after C changes.
- Run `cd web && npm run build` after TS/React changes (also runs `tsc -b`).
- Run `cd web && npm test` for web unit tests.

## Security
- HTTP defense headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`) must be preserved.
- `_is_safe_path()` boundary check must guard every filesystem operation that takes a user-supplied path.
- Bearer auth (`QWANTO_API_KEY`) is opt-in; do not disable existing auth checks.
- `eval`, `exec`, shell interpolation with user data are forbidden; use `subprocess.run([...], ...)` with explicit argv.

## Deployment Restrictions
- No bundled inference engine in Tauri desktop build (models are user-managed).
- `llama-server` is fetched on demand on Windows from official llama.cpp releases.
- Do not commit model files, secrets, or build artifacts (see `.gitignore`).

## Agent Rules
- Read `PROJECT_STATE.md` and `AGENTS.md` before significant work.
- Update `PROJECT_STATE.md` and append to `AGENT_LOG.md` after meaningful changes.
- Preserve upstream Colibri attribution when touching the multi-tier memory subsystem.
- Do not refactor unrelated code; do not introduce new dependencies without justification.
