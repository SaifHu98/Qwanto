"""Check the release workflow's unsigned and conditional-signing contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def main() -> int:
    text = WORKFLOW.read_text(encoding="utf-8")
    required = (
        "SIGNING_ENABLED == 'true'",
        "signtool.Source verify /pa /all /tw",
        "grep -Fq \"$basename\" \"$checksum_file\"",
        "--prerelease",
        "This is an unsigned beta release. Windows SmartScreen and macOS Gatekeeper may display warnings. Verify the SHA-256 checksum before installing.",
        "libwebkit2gtk-4.1-dev",
        "libsoup-3.0-dev",
        "pkg-config",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise SystemExit(f"Release workflow contract missing: {', '.join(missing)}")
    if "production-release-gate" in text or "qwanto-icon.png release-assets" in text:
        raise SystemExit("Release workflow still contains a blocking signing gate or extra icon upload")
    signtool_block = text[text.index("Verify Windows Authenticode signatures"):text.index("Create Linux detached signatures")]
    if "SIGNING_ENABLED == 'true'" not in signtool_block:
        raise SystemExit("SignTool verification is not conditionally guarded by SIGNING_ENABLED")
    print("Release workflow unsigned/conditional-signing contract verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
