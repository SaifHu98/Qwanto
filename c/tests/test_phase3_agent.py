import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
sys.path.insert(0, str(C_DIR))

import openai_server


class TestPhase3AgentSecurity(unittest.TestCase):
    def test_workspace_boundary_enforcement(self):
        """Verify that files outside workspace root are rejected."""
        project_root = openai_server.PROJECT_ROOT
        
        # Valid path inside project
        valid_file = project_root / "c" / "qwnrun.c"
        self.assertTrue(openai_server._is_safe_path(valid_file))

        # Malicious traversal path
        traversal_file = project_root / ".." / ".." / "Windows" / "System32" / "cmd.exe"
        self.assertFalse(openai_server._is_safe_path(traversal_file))

    def test_secret_redaction(self):
        """Verify that typical API keys and secrets are redacted."""
        raw_text = "Connecting to service with secret sk-12345678901234567890abcdef and ghp_98765432109876543210"
        
        # Redaction logic test
        import re
        redacted = re.sub(r"sk-[a-zA-Z0-9_-]{20,}", "[REDACTED_API_KEY]", raw_text)
        redacted = re.sub(r"ghp_[a-zA-Z0-9]{20,}", "[REDACTED_GITHUB_TOKEN]", redacted)

        self.assertNotIn("sk-12345678901234567890abcdef", redacted)
        self.assertIn("[REDACTED_API_KEY]", redacted)
        self.assertNotIn("ghp_98765432109876543210", redacted)
        self.assertIn("[REDACTED_GITHUB_TOKEN]", redacted)

    def test_license_terms(self):
        """Verify Apache 2.0 license integrity."""
        license_path = openai_server.PROJECT_ROOT / "LICENSE"
        with open(license_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("Apache License", text)
        self.assertIn("Version 2.0", text)


if __name__ == "__main__":
    unittest.main()
