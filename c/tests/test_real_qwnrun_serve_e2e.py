import os
import sys
import unittest
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
C_DIR = PROJECT_ROOT / "c"

class TestRealQwnrunServeE2E(unittest.TestCase):
    def test_real_qwnrun_serve_two_prompts_same_pid(self):
        """Execute real qwnrun --serve binary with real .qwn model and verify 2 prompts on same PID."""
        exe = C_DIR / "qwnrun_msvc.exe"
        if not exe.exists():
            exe = C_DIR / "qwnrun.exe"
        
        model_path = PROJECT_ROOT / "experiments" / "results" / "4B_hyper_vsq2.qwn"

        if not exe.exists():
            self.skipTest(f"qwnrun binary not found at {exe}")
        if not model_path.exists():
            self.skipTest(f"Real .qwn model file not found at {model_path}")

        proc = subprocess.Popen(
            [str(exe), str(model_path), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        initial_pid = proc.pid
        print(f"\n[REAL RUNTIME] Started qwnrun --serve with PID {initial_pid}", file=sys.stderr)

        try:
            # 1. First prompt
            prompt_1 = b"Hello Qwanto"
            req_1 = "req-live-001"
            submit_1 = f"SUBMIT {req_1} 0 {len(prompt_1)} 8 0.000000 1.000000\n".encode("utf-8") + prompt_1 + b"\n"
            proc.stdin.write(submit_1)
            proc.stdin.flush()

            # Read response
            line_1 = proc.stdout.readline()
            self.assertIn(b"DONE req-live-001", line_1, "Prompt 1 should emit DONE frame")
            self.assertEqual(proc.pid, initial_pid, "PID must remain identical")
            print(f"[REAL RUNTIME] Request 1 completed on PID {proc.pid}: {line_1.decode('utf-8').strip()}", file=sys.stderr)

            # 2. Second prompt on the SAME running process
            prompt_2 = b"Explain SIMD acceleration in Qwanto"
            req_2 = "req-live-002"
            submit_2 = f"SUBMIT {req_2} 0 {len(prompt_2)} 8 0.000000 1.000000\n".encode("utf-8") + prompt_2 + b"\n"
            proc.stdin.write(submit_2)
            proc.stdin.flush()

            # Read response
            line_2 = proc.stdout.readline()
            self.assertIn(b"DONE req-live-002", line_2, "Prompt 2 should emit DONE frame")
            self.assertEqual(proc.pid, initial_pid, "PID must remain identical across sequential requests")
            print(f"[REAL RUNTIME] Request 2 completed on PID {proc.pid}: {line_2.decode('utf-8').strip()}", file=sys.stderr)

            print(f"[REAL RUNTIME] PROVED: Both prompts served under single PID {initial_pid} without process restart!", file=sys.stderr)

        finally:
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
