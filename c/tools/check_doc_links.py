"""Check repository-local Markdown links without contacting external hosts."""

from __future__ import annotations

import re
import sys
from pathlib import Path


LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
DOCUMENTS = ("README.md", "RELEASE_READINESS.md", "desktop/README.md", "web/README.md")


def markdown_files(root: Path) -> list[Path]:
    files = [root / relative for relative in DOCUMENTS if (root / relative).is_file()]
    files.extend(sorted((root / "docs").glob("*.md")))
    return files


def broken_links(root: Path) -> list[str]:
    failures: list[str] = []
    for document in markdown_files(root):
        for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
            for raw_target in LINK_RE.findall(line):
                target = raw_target.strip().split("#", 1)[0].split("?", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (document.parent / target).resolve()
                if resolved.is_dir():
                    resolved = resolved / "README.md"
                if not resolved.is_file():
                    failures.append(f"{document.relative_to(root)}:{line_number}: {raw_target}")
    return failures


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    failures = broken_links(root)
    if failures:
        print("Broken repository-local Markdown links:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Documentation links OK ({len(markdown_files(root))} Markdown files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
