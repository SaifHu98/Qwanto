# Release readiness

## Status: Beta 2 gates in progress

`v0.1.0-beta.1` is an existing release and remains unchanged. This working
tree targets `v0.1.0-beta.2`; a fresh local and hosted validation run is
required before publishing it.

## Gates

| Gate | Status | Evidence required |
| --- | --- | --- |
| Native C and persistent protocol tests | Pending fresh CI | `c/tests/`, native CI job |
| Python gateway/security/conversion tests | Pending fresh CI | `python -m pytest c/tests/ -q` |
| Rust check/test/Clippy | Pending fresh CI | Tauri CI jobs |
| Web build and tests | Pending fresh CI | web CI job |
| Windows NSIS/MSI | Pending tag workflow run | uploaded package artifacts |
| Linux AppImage/deb | Pending tag workflow run | uploaded package artifacts |
| macOS DMG | Pending tag workflow run and target review | uploaded package artifact |
| No model files in packages | Workflow assertion added | package verification step |
| Documentation links | Pending fresh CI | `python c/tools/check_doc_links.py` |
| Native benchmark evidence | Host-dependent | `benchmark_evidence.json` with `MEASURED` classification |

Do not create the beta.2 tag or publish until the fresh CI run is green and the
target package artifacts have been inspected. Missing model fixtures may remain
skipped only when absent; a present fixture must run.
