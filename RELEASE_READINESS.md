# Release readiness

## Status: Not Release Ready

This working tree contains the Beta release-engineering changes, but the new
documentation, benchmark, web, and packaging paths require a fresh CI run.
The previous CI run predates these changes and is not evidence for this tree.

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
| Native benchmark evidence | Host-dependent | `benchmark_evidence.json` with `MEASURED` classification |

Do not create a tag or publish a release until the fresh CI run is green and
the target package artifacts have been inspected. Missing model fixtures may
remain skipped only when absent; a present fixture must run.
