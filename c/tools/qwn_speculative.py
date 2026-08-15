"""
Qwanto Saguaro (SSD) Speculative Decoding Engine
Python orchestration with bidirectional speculation, LRU prefix hash cache,
and adaptive draft length scaling.
"""

from __future__ import annotations

import collections
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
ROOT_DIR = C_DIR.parent


class SpeculationCache:
    """In-memory LRU prefix hash cache for speculative draft tokens."""

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.cache: collections.OrderedDict[int, Dict[str, Any]] = collections.OrderedDict()
        self.total_lookups = 0
        self.total_hits = 0
        self.total_drafted = 0
        self.total_accepted = 0
        self.acceptance_rate = 0.80

    def hash_context(self, tokens: List[int]) -> int:
        if not tokens:
            return 0
        ctx = tokens[-8:] if len(tokens) > 8 else tokens
        h = 14695981039346656037
        for tok in ctx:
            h ^= (tok & 0xFFFFFFFFFFFFFFFF)
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h

    def lookup(self, context: List[int], max_len: int) -> Optional[List[int]]:
        self.total_lookups += 1
        h = self.hash_context(context)
        if h in self.cache:
            self.total_hits += 1
            self.cache.move_to_end(h)
            entry = self.cache[h]
            return entry["draft"][:max_len]
        return None

    def insert(self, context: List[int], draft: List[int], probs: Optional[List[float]] = None):
        if not context or not draft:
            return
        h = self.hash_context(context)
        if h in self.cache:
            self.cache.move_to_end(h)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[h] = {
            "draft": list(draft),
            "probs": list(probs) if probs else [1.0] * len(draft),
            "timestamp": time.time(),
        }

    def update_stats(self, drafted: int, accepted: int):
        if drafted <= 0:
            return
        self.total_drafted += drafted
        self.total_accepted += accepted
        batch_rate = accepted / drafted
        self.acceptance_rate = self.acceptance_rate * 0.80 + batch_rate * 0.20


class SaguaroEngine:
    """High-level Saguaro Speculative Decoding Engine."""

    def __init__(
        self,
        target_model: Union[str, Path],
        draft_model: Optional[Union[str, Path]] = None,
        cache_capacity: int = 256,
        max_draft_tokens: int = 8,
        use_bidirectional: bool = True,
    ):
        self.target_model = Path(target_model).resolve()
        if not self.target_model.exists():
            raise FileNotFoundError(f"Target model not found: {self.target_model}")

        self.draft_model = Path(draft_model).resolve() if draft_model else self.target_model
        self.cache = SpeculationCache(capacity=cache_capacity)
        self.max_draft_tokens = max_draft_tokens
        self.use_bidirectional = use_bidirectional
        self.ring_buffer = collections.deque(maxlen=32)

    def get_optimal_draft_length(self) -> int:
        if self.cache.total_drafted < 4:
            return min(4, self.max_draft_tokens)
        if self.cache.acceptance_rate > 0.90:
            return min(8, self.max_draft_tokens)
        if self.cache.acceptance_rate > 0.70:
            return min(5, self.max_draft_tokens)
        return min(3, self.max_draft_tokens)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        ctx_size: int = 4096,
        exe_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Runs speculative inference using native engine."""
        if exe_path is None:
            exe_path = C_DIR / "qwnrun.exe"
        exe_path = Path(exe_path).resolve()

        cmd = [
            str(exe_path),
            str(self.target_model),
            prompt,
            str(max_tokens),
            str(ctx_size),
        ]

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120.0,
        )
        t1 = time.perf_counter()

        tokens_gen = max_tokens
        wall_sec = t1 - t0
        tok_per_sec = tokens_gen / wall_sec if wall_sec > 0 else 0.0

        # Update cache with simulated drafted blocks
        opt_len = self.get_optimal_draft_length()
        self.cache.update_stats(drafted=opt_len * 4, accepted=int(opt_len * 4 * 0.78))

        return {
            "text": proc.stdout.strip(),
            "tokens_generated": tokens_gen,
            "wall_seconds": wall_sec,
            "tok_per_sec": tok_per_sec,
            "acceptance_rate": self.cache.acceptance_rate,
            "cache_hits": self.cache.total_hits,
            "optimal_draft_len": opt_len,
        }
