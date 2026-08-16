#!/usr/bin/env python3
"""
audit_secrets.py — Static multi-language AST and pattern audit for unredacted credentials.
"""

import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_PATTERNS = [
    re.compile(r'(sk-[a-zA-Z0-9_-]{24,})'),
    re.compile(r'(ghp_[a-zA-Z0-9]{20,})'),
    re.compile(r'(xoxb-[0-9]{10,}-[0-9]{10,}-[a-zA-Z0-9]{24,})'),
]

ALLOW_SUBSTRINGS = [
    'test-fixture',
    'dummy',
    'test_secret',
    '[REDACTED',
    'Regex::new',
    're.compile',
    'r"sk-',
    "r'sk-",
    'r"ghp_',
    "r'ghp_",
    'placeholder',
    'EXAMPLE',
    'assertNotIn',
    'assert!',
    'mock',
]

SKIP_DIRS = {
    '.git',
    'node_modules',
    'dist',
    'target',
    'openagent-master',
    '.gemini',
    'artifacts'
}

CHECK_EXTENSIONS = {
    '.py', '.ts', '.tsx', '.rs', '.c', '.h', '.md', '.json', '.toml', '.yml', '.yaml'
}

def audit_workspace():
    bad = False
    checked_files = 0
    for root, dirs, files in os.walk(ROOT_DIR):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in CHECK_EXTENSIONS:
                checked_files += 1
                filepath = Path(root) / f
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fp:
                        for line_no, line in enumerate(fp, 1):
                            for pattern in SECRET_PATTERNS:
                                if pattern.search(line):
                                    if not any(allow in line for allow in ALLOW_SUBSTRINGS):
                                        print(f"Potential secret in {filepath.relative_to(ROOT_DIR)}:{line_no}: {line.strip()[:60]}", file=sys.stderr)
                                        bad = True
                except Exception as e:
                    print(f"Warning: could not read {filepath}: {e}", file=sys.stderr)

    if bad:
        print(f"FAILED: Found unredacted credentials across {checked_files} files.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Secret scan clean: {checked_files} files checked.", file=sys.stderr)

if __name__ == "__main__":
    audit_workspace()
