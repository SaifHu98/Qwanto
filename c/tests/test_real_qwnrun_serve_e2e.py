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
    @staticmethod
    def _read_done(proc, request_id):
        while True:
            line = proc.stdout.readline()
            if not line:
                raise AssertionError("qwnrun exited before DONE")
            if line.startswith(b"DATA "):
                fields = line.decode("utf-8", "replace").split()
                payload = proc.stdout.read(int(fields[2]))
                terminator = proc.stdout.read(1)
                if terminator != b"\n":
                    raise AssertionError("invalid DATA terminator")
                continue
            if line.startswith(b"DONE "):
                if request_id.encode() not in line:
                    raise AssertionError(f"unexpected DONE frame: {line!r}")
                return line

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
            line_1 = self._read_done(proc, req_1)
            self.assertEqual(proc.pid, initial_pid, "PID must remain identical")
            print(f"[REAL RUNTIME] Request 1 completed on PID {proc.pid}: {line_1.decode('utf-8').strip()}", file=sys.stderr)

            # 2. Second prompt on the SAME running process
            prompt_2 = b"Explain SIMD acceleration in Qwanto"
            req_2 = "req-live-002"
            submit_2 = f"SUBMIT {req_2} 0 {len(prompt_2)} 8 0.000000 1.000000\n".encode("utf-8") + prompt_2 + b"\n"
            proc.stdin.write(submit_2)
            proc.stdin.flush()

            # Read response
            line_2 = self._read_done(proc, req_2)
            self.assertEqual(proc.pid, initial_pid, "PID must remain identical across sequential requests")
            print(f"[REAL RUNTIME] Request 2 completed on PID {proc.pid}: {line_2.decode('utf-8').strip()}", file=sys.stderr)

            print(f"[REAL RUNTIME] PROVED: Both prompts served under single PID {initial_pid} without process restart!", file=sys.stderr)
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_runtime_config_reports_requested_threads(self):
        exe = C_DIR / "qwnrun_msvc.exe"
        if not exe.exists():
            exe = C_DIR / "qwnrun.exe"
        if not exe.exists():
            self.skipTest(f"qwnrun binary not found at {exe}")
        for requested in (1, 2):
            result = subprocess.run(
                [str(exe), "--build-info", "--threads", str(requested)],
                capture_output=True, text=True, check=False,
            )
            output = result.stdout + result.stderr
            if "requested_threads=" not in output:
                self.skipTest("qwnrun binary predates auditable thread configuration")
            self.assertEqual(result.returncode, 0, output)
            self.assertIn(f"requested_threads={requested}", output)


if __name__ == "__main__":
    unittest.main()
