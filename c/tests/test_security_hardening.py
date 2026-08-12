import unittest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai_server import _is_safe_path, PROJECT_ROOT


class TestSecurityHardening(unittest.TestCase):
    def test_path_traversal_prevention(self):
        # Safe path within project root
        safe_file = PROJECT_ROOT / "README.md"
        self.assertTrue(_is_safe_path(safe_file))

        # Path traversal attempting to leave project root
        traversal = PROJECT_ROOT / ".." / ".." / "Windows" / "System32"
        self.assertFalse(_is_safe_path(traversal))

        # Relative path traversal string
        unsafe_str = "../../etc/passwd"
        self.assertFalse(_is_safe_path(unsafe_str))


if __name__ == "__main__":
    unittest.main()
