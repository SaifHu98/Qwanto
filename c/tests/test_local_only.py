import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
sys.path.insert(0, str(C_DIR))

import openai_server


class TestLocalOnlyProfile(unittest.TestCase):
    def test_default_host_is_localhost(self):
        """Verify that the HTTP gateway defaults strictly to 127.0.0.1."""
        self.assertEqual(openai_server.serve.__defaults__[0], "127.0.0.1")

    def test_ensure_llama_server_blocks_download_by_default(self):
        """Verify that _ensure_llama_server does not trigger network download when allow_download is False."""
        with patch.dict(os.environ, {}, clear=True), \
             patch("shutil.which", return_value=None), \
             patch.object(Path, "exists", return_value=False), \
             patch("urllib.request.urlopen") as mock_urlopen:
            
            exe = openai_server._ensure_llama_server(allow_download=False)
            self.assertIsNone(exe)
            mock_urlopen.assert_not_called()

    def test_safe_path_boundary_enforcement(self):
        """Verify that _is_safe_path rejects directory traversal attempts."""
        project_root = openai_server.PROJECT_ROOT
        
        # Valid paths within project
        valid_path = project_root / "c" / "qwnrun.c"
        self.assertTrue(openai_server._is_safe_path(valid_path))

        # Traversal attacks
        escaped_path = project_root / ".." / ".." / "Windows" / "System32"
        self.assertFalse(openai_server._is_safe_path(escaped_path))

    def test_license_is_apache_2(self):
        """Verify that LICENSE file is Apache 2.0."""
        license_file = project_root = openai_server.PROJECT_ROOT / "LICENSE"
        self.assertTrue(license_file.exists())
        with open(license_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Apache License", content)
        self.assertIn("Version 2.0", content)


if __name__ == "__main__":
    unittest.main()
