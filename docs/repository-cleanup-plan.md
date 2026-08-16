# 🧹 Repository Cleanup & Migration Plan: `openagent-master/` Inventory

## 1. Executive Summary & Inventory

The `openagent-master/` directory is an isolated, legacy Electron-based reference directory located at the root of the repository. Qwanto Native has transitioned completely to a lightweight **Tauri 2.11 + React 18 + Native C Engine** architecture (`desktop/src-tauri/` + `web/` + `c/`).

### Inventory Metrics:
- **Directory Path**: `d:/EcoUni/qwanto/openagent-master`
- **Total Tracked Files**: 202 files
- **Total Disk Size**: ~2.10 MB ($2,099,237$ bytes)
- **Active Build Dependencies**: **0** (neither `web/` nor `desktop/src-tauri/` nor `c/` imports or depends on `openagent-master`).
- **External / Cloud Provider References**: Contains legacy references to cloud LLM providers (Anthropic, OpenRouter, cloud Ollama) and Electron packaging scripts that violate Qwanto's 100% local-only execution invariant.

---

## 2. Component Breakdown

| Subdirectory / File Group | File Count | Purpose / Status | Migration / Action |
|---|:---:|---|---|
| `openagent-master/electron/` | 18 | Legacy Electron main process, preload, and IPC bridges. | Replaced by Tauri Rust host (`desktop/src-tauri/`). Safe for removal. |
| `openagent-master/src/` | 142 | Legacy React components and state management. | Core UI patterns extracted to native React dashboard (`web/src/`). Safe for removal. |
| `openagent-master/package.json` | 1 | Node.js Electron dependency manifest. | Superseded by `web/package.json` and `desktop/package.json`. Safe for removal. |
| `openagent-master/scripts/` | 8 | Build and packaging scripts for Electron. | Superseded by `desktop/src-tauri/Cargo.toml` and `docs/packaging.md`. Safe for removal. |
| `openagent-master/docs/` | 4 | Legacy upstream OpenAgent documentation. | Superseded by `README.md` and `docs/`. Safe for removal. |

---

## 3. License & Attribution Implications

- **Upstream License**: MIT License (samhu1/openagent).
- **Attribution Invariant**: Useful architectural patterns adopted during design are documented in `PROJECT_STATE.md`.
- **Qwanto Platform License**: The entire active repository is licensed under the **Apache License 2.0**.

---

## 4. Phased Removal & Migration Plan

### Step 1: Pre-Removal Verification (Current State)
- Confirm that no build script, CI workflow, test runner, or frontend import links to `openagent-master/`.
- Verify that `desktop/` and `web/` build and pass all unit tests autonomously.

### Step 2: Explicit User Approval Gate
- In accordance with safety rules, `openagent-master/` will **not** be deleted automatically.
- A dedicated user approval task will trigger the deletion command:
  ```bash
  git rm -r openagent-master
  ```

### Step 3: Post-Removal Validation
- Re-run full test suites:
  ```bash
  python -m pytest c/tests/ -q
  cd web && npm test
  ```
- Confirm repository root is clean, minimal, and 100% Qwanto-native.
