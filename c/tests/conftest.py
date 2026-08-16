"""
conftest.py — pytest session-scoped path setup for c/tests/.

Ensures the c/ source directory is discoverable as a package root so
that tests can import modules like:

    from doctor import ...
    from resource_plan import ...
    from openai_server import ...
    import backends
    from tools.qwn_convert import ...

This runs before test collection on all platforms, including the
ubuntu-latest GitHub Actions runner where pytest's rootdir detection
may differ from Windows.
"""
import sys
from pathlib import Path

# Add the c/ directory (parent of tests/) to sys.path so that all
# top-level Python modules in c/ are importable by name.
C_DIR = Path(__file__).resolve().parent.parent
if str(C_DIR) not in sys.path:
    sys.path.insert(0, str(C_DIR))

# Also expose c/tools/ for convenience (some tests use both
# `from tools.qwn_convert import ...` and `import qwn_convert ...`).
TOOLS_DIR = C_DIR / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
