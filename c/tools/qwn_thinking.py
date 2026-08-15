"""
Qwanto Configurable Thinking Engine (Dynamic Reasoning Engine)
Python bindings & orchestration for per-request adaptive thinking depth.
"""

from __future__ import annotations

import enum
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
ROOT_DIR = C_DIR.parent


class ThinkingLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def from_value(cls, val: Union[str, int, ThinkingLevel, None]) -> ThinkingLevel:
        if val is None:
            return cls.MEDIUM
        if isinstance(val, ThinkingLevel):
            return val
        s = str(val).lower().strip()
        if s in ("low", "fast", "0"):
            return cls.LOW
        if s in ("high", "deep", "cot", "2"):
            return cls.HIGH
        return cls.MEDIUM


class QwnThinkingEngine:
    """High-level Python wrapper for Qwanto Configurable Thinking Inference Runtime."""

    def __init__(self, model_path: Union[str, Path], exe_path: Optional[Union[str, Path]] = None, ctx_size: int = 4096):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if exe_path is None:
            candidates = [
                C_DIR / "qwnrun.exe",
                C_DIR / "qwnrun_msvc.exe",
                C_DIR / "qwnrun",
            ]
            for c in candidates:
                if c.exists():
                    self.exe_path = c
                    break
            else:
                self.exe_path = C_DIR / "qwnrun.exe"
        else:
            self.exe_path = Path(exe_path).resolve()
            
        self.ctx_size = ctx_size

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        thinking_level: Union[str, int, ThinkingLevel] = ThinkingLevel.MEDIUM,
        timeout: float = 120.0
    ) -> Dict[str, Any]:
        """Runs autoregressive token generation with dynamic configurable thinking."""
        lvl = ThinkingLevel.from_value(thinking_level)
        env = os.environ.copy()
        env["QWN_THINKING_LEVEL"] = lvl.value
        env["QWANTO_TEMP"] = str(temperature)
        env["QWANTO_TOP_P"] = str(top_p)

        cmd = [
            str(self.exe_path),
            str(self.model_path),
            prompt,
            str(max_tokens),
            str(self.ctx_size),
            "--thinking",
            lvl.value,
        ]

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
        )
        t1 = time.perf_counter()

        if proc.returncode != 0:
            raise RuntimeError(f"qwnrun failed (rc={proc.returncode}):\n{proc.stderr}")

        # Parse metrics from stderr
        result_match = re.search(
            r"qwnrun result:\s+status=(\w+)\s+tokens=(\d+)\s+wall_seconds=([\d.]+)\s+ttft_ms=([\d.]+)\s+tok_per_sec=([\d.]+)",
            proc.stderr,
        )
        tokens_generated = int(result_match.group(2)) if result_match else 0
        wall_seconds = float(result_match.group(3)) if result_match else (t1 - t0)
        ttft_ms = float(result_match.group(4)) if result_match else 0.0
        tok_per_sec = float(result_match.group(5)) if result_match else (tokens_generated / wall_seconds if wall_seconds > 0 else 0.0)

        return {
            "text": proc.stdout.strip(),
            "tokens_generated": tokens_generated,
            "wall_seconds": wall_seconds,
            "ttft_ms": ttft_ms,
            "tok_per_sec": tok_per_sec,
            "thinking_level": lvl.value,
            "stderr": proc.stderr,
        }

    def benchmark(
        self,
        prompt: str = "Explain the theory of general relativity in simple terms.",
        thinking_level: Union[str, int, ThinkingLevel] = ThinkingLevel.MEDIUM,
        max_tokens: int = 32,
        iterations: int = 5
    ) -> Dict[str, Any]:
        """Runs iterative benchmark runs and calculates throughput statistics."""
        lvl = ThinkingLevel.from_value(thinking_level)
        runs = []
        for _ in range(iterations):
            res = self.generate(prompt=prompt, max_tokens=max_tokens, thinking_level=lvl)
            runs.append(res)

        tps_list = [r["tok_per_sec"] for r in runs if r["tok_per_sec"] > 0]
        avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
        min_tps = min(tps_list) if tps_list else 0.0
        max_tps = max(tps_list) if tps_list else 0.0

        return {
            "thinking_level": lvl.value,
            "iterations": len(runs),
            "avg_tok_per_sec": round(avg_tps, 2),
            "min_tok_per_sec": round(min_tps, 2),
            "max_tok_per_sec": round(max_tps, 2),
            "runs": runs,
        }
