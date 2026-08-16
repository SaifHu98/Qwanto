import os
import sys
import unittest
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
sys.path.insert(0, str(C_DIR))

class TestQwnrunServeProtocol(unittest.TestCase):
    def test_mock_serve_two_prompts_same_pid(self):
        """Simulate the qwnrun --serve framed protocol and verify two requests run under the same PID."""
        mock_script = """
import sys, time

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    if line == "PING":
        sys.stdout.write("PONG\\n")
        sys.stdout.flush()
    elif line.startswith("SUBMIT "):
        parts = line.split()
        req_id = parts[1]
        n_bytes = int(parts[3])
        max_tokens = int(parts[4])
        # Read exact prompt bytes
        prompt = sys.stdin.read(n_bytes)
        term = sys.stdin.read(1) # consume newline
        
        # Stream framed DATA tokens
        tokens = ["Hello", " world", "\\nfrom", " Qwanto!"]
        for t in tokens:
            sys.stdout.write(f"DATA {req_id} {len(t)}\\n{t}\\n")
            sys.stdout.flush()
        
        # Emit DONE frame
        sys.stdout.write(f"DONE {req_id} STAT 4 125.500 0 0 {n_bytes} 0\\n")
        sys.stdout.flush()
    elif line.startswith("CANCEL "):
        parts = line.split()
        req_id = parts[1]
        sys.stdout.write(f"ERROR {req_id} CANCELLED\\n")
        sys.stdout.flush()
"""
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", mock_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        initial_pid = proc.pid

        try:
            # Request 1
            prompt_1 = "First prompt test"
            submit_cmd_1 = f"SUBMIT req-1 0 {len(prompt_1)} 64 0.000000 1.000000\n{prompt_1}\n"
            proc.stdin.write(submit_cmd_1)
            proc.stdin.flush()

            # Read frames for Request 1
            received_tokens_1 = []
            while True:
                header = proc.stdout.readline()
                if not header:
                    break
                header = header.strip()
                if header.startswith("DATA "):
                    _, req_id, n_bytes = header.split()
                    n = int(n_bytes)
                    token_data = proc.stdout.read(n)
                    trailing = proc.stdout.read(1) # consume newline
                    received_tokens_1.append(token_data)
                elif header.startswith("DONE "):
                    self.assertIn("req-1", header)
                    break

            self.assertEqual("".join(received_tokens_1), "Hello world\nfrom Qwanto!")
            # Verify PID did not change
            self.assertEqual(proc.pid, initial_pid)

            # Request 2 on the exact same running process
            prompt_2 = "Second prompt with \n newlines and symbols !"
            submit_cmd_2 = f"SUBMIT req-2 0 {len(prompt_2)} 64 0.700000 0.900000\n{prompt_2}\n"
            proc.stdin.write(submit_cmd_2)
            proc.stdin.flush()

            received_tokens_2 = []
            while True:
                header = proc.stdout.readline()
                if not header:
                    break
                header = header.strip()
                if header.startswith("DATA "):
                    _, req_id, n_bytes = header.split()
                    n = int(n_bytes)
                    token_data = proc.stdout.read(n)
                    trailing = proc.stdout.read(1)
                    received_tokens_2.append(token_data)
                elif header.startswith("DONE "):
                    self.assertIn("req-2", header)
                    break

            self.assertEqual("".join(received_tokens_2), "Hello world\nfrom Qwanto!")
            # PID still strictly identical
            self.assertEqual(proc.pid, initial_pid)

        finally:
            proc.terminate()
            proc.wait()

    def test_cancel_frame(self):
        mock_script = """
import sys
for line in sys.stdin:
    if line.startswith("CANCEL "):
        req_id = line.split()[1]
        sys.stdout.write(f"ERROR {req_id} CANCELLED\\n")
        sys.stdout.flush()
"""
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", mock_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            proc.stdin.write("CANCEL req-cancel-99\n")
            proc.stdin.flush()
            reply = proc.stdout.readline().strip()
            self.assertEqual(reply, "ERROR req-cancel-99 CANCELLED")
        finally:
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    unittest.main()
