"""Verify the checked-in built-in skill package shape."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / "skills"


def main() -> int:
    packages = sorted(path for path in SKILLS.iterdir() if path.is_dir())
    if not packages:
        raise SystemExit("No built-in skill packages found")
    for package in packages:
        manifest_path = package / "skill.json"
        instructions_path = package / "SKILL.md"
        if not manifest_path.is_file() or not instructions_path.is_file():
            raise SystemExit(f"Incomplete skill package: {package}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {"schema_version", "id", "name", "version", "capabilities", "entrypoint", "builtin"}
        missing = required.difference(manifest)
        if missing:
            raise SystemExit(f"{manifest_path} is missing: {', '.join(sorted(missing))}")
        if manifest["schema_version"] != 1 or manifest["builtin"] is not True:
            raise SystemExit(f"Invalid built-in skill metadata: {manifest_path}")
        if manifest["id"] != package.name or manifest["entrypoint"] != "SKILL.md":
            raise SystemExit(f"Skill id/entrypoint mismatch: {manifest_path}")
        if not manifest["capabilities"] or any(not isinstance(value, str) for value in manifest["capabilities"]):
            raise SystemExit(f"Invalid capabilities: {manifest_path}")
    print(f"Built-in skill packages verified: {len(packages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
