"""Verify that packaged branding files are present and share the approved source hash."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets" / "brand" / "qwanto-icon.png"
MIRRORS = [
    ROOT / "desktop" / "src-tauri" / "icons" / "icon.png",
    ROOT / "web" / "public" / "qwanto-icon.png",
]
REQUIRED = MIRRORS + [
    ROOT / "desktop" / "src-tauri" / "icons" / "32x32.png",
    ROOT / "desktop" / "src-tauri" / "icons" / "128x128.png",
    ROOT / "desktop" / "src-tauri" / "icons" / "128x128@2x.png",
    ROOT / "desktop" / "src-tauri" / "icons" / "icon.ico",
    ROOT / "desktop" / "src-tauri" / "icons" / "icon.icns",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def main() -> int:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    if missing:
        raise SystemExit("Missing brand assets: " + ", ".join(missing))
    source_hash = sha256(SOURCE)
    mismatched = [str(path.relative_to(ROOT)) for path in MIRRORS if sha256(path) != source_hash]
    if mismatched:
        raise SystemExit("Brand mirrors differ from assets/brand/qwanto-icon.png: " + ", ".join(mismatched))
    if png_size(SOURCE) != (512, 512):
        raise SystemExit(f"Approved source must remain 512x512, got {png_size(SOURCE)}")
    print("Brand assets verified:", source_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
