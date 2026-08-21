"""
Qwanto Performance Autopilot Engine
Unified orchestration of TurboQuant, Thinking Levels, Saguaro SSD, and Agentic Pipeline.
"""

from __future__ import annotations

import enum
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
ROOT_DIR = C_DIR.parent

sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(C_DIR / "tools"))

from qwn_agentic import OptimizedAgent
from qwn_speculative import SaguaroEngine
from qwn_thinking import QwnThinkingEngine


class TaskType(enum.Enum):
    SIMPLE_QA = "simple_qa"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    AGENTIC = "multi_turn_agentic"
    TOOL_INTENSIVE = "tool_intensive"
    BATCH = "batch_processing"


class TaskClassifier:
    """Classifies prompts and requests into task archetypes for optimization selection."""

    CODE_PATTERNS = [
        r"\bdef\s+\w+\(", r"\bfunction\b", r"\bclass\s+\w+", r"\bimport\s+\w+",
        r"#include\b", r"\bint\s+main\(", r"\bSELECT\s+.+\s+FROM\b", r"```",
        r"\bpublic\s+static\s+void\b", r"const\s+\w+\s*=", r"=>",
        r"\bpython\b", r"\bquicksort\b", r"\bcode\b", r"\bscript\b", r"\balgorithm\b",
        r"\bprogramming\b", r"\bjavascript\b", r"\bc\+\+\b", r"\brust\b"
    ]

    REASONING_PATTERNS = [
        r"\bwhy\b", r"\bexplain\b", r"\bprove\b", r"\bcalculate\b", r"\bstep-by-step\b",
        r"\bmathematical\b", r"\btheorem\b", r"\bderive\b", r"\banalyze\b", r"\bevaluate\b"
    ]

    def detect_code(self, text: str) -> bool:
        return any(re.search(pat, text, re.IGNORECASE) for pat in self.CODE_PATTERNS)

    def detect_reasoning(self, text: str) -> bool:
        return any(re.search(pat, text, re.IGNORECASE) for pat in self.REASONING_PATTERNS)

    def classify(self, prompt: str, max_tokens: int = 100, tools: Optional[List[Any]] = None) -> TaskType:
        if tools and len(tools) > 3:
            return TaskType.TOOL_INTENSIVE
        if tools and len(tools) > 0:
            return TaskType.AGENTIC
        if max_tokens > 256 and self.detect_code(prompt):
            return TaskType.CODE_GENERATION
        if self.detect_code(prompt):
            return TaskType.CODE_GENERATION
        if self.detect_reasoning(prompt) and len(prompt) > 80:
            return TaskType.REASONING
        if len(prompt) > 200:
            return TaskType.REASONING
        return TaskType.SIMPLE_QA


@dataclass
class AutoPilotResponse:
    text: str
    speedup: Optional[float]
    tokens_per_second: Optional[float]
    active_optimizations: List[str]
    quality_score: Optional[float]
    memory_usage_gb: Optional[float]
    task_type: str
    thinking_level: str


class QwantoAutoPilot:
    """Unified performance autopilot orchestrator."""

    def __init__(
        self,
        model_path: Union[str, Path] = ROOT_DIR / "experiments" / "results" / "4B_hyper_vsq2.qwn",
        mode: str = "balanced",
        auto_detect: bool = True,
    ):
        self.model_path = Path(model_path).resolve()
        self.mode = mode.lower()
        self.auto_detect = auto_detect
        self.classifier = TaskClassifier()

        # Matrix definitions contain policy choices only.  No speed, quality,
        # or memory value is inferred from a task label.
        self.matrix = {
            TaskType.SIMPLE_QA: ("low", False, False, False),
            TaskType.CODE_GENERATION: ("medium", False, False, False),
            TaskType.REASONING: ("high", False, False, False),
            TaskType.AGENTIC: ("medium", False, False, True),
            TaskType.TOOL_INTENSIVE: ("low", False, False, True),
            TaskType.BATCH: ("low", False, False, True),
        }

    def generate(
        self,
        prompt: str,
        task_type: Optional[Union[TaskType, str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 64,
        thinking_level: str = "auto",
    ) -> AutoPilotResponse:
        t0 = time.perf_counter()

        # Step 1: Classify task
        if task_type is None:
            resolved_task = self.classifier.classify(prompt, max_tokens, tools)
        elif isinstance(task_type, str):
            resolved_task = TaskType(task_type) if task_type in [t.value for t in TaskType] else TaskType.SIMPLE_QA
        else:
            resolved_task = task_type

        # Step 2: Extract optimal configuration from matrix
        think_cfg, use_tq, use_ssd, use_agt = self.matrix[resolved_task]

        if self.mode == "max-performance":
            think_cfg, use_tq, use_ssd, use_agt = ("low", False, False, True)
        elif self.mode == "max-quality":
            think_cfg, use_tq, use_ssd, use_agt = ("high", False, False, False)

        if thinking_level != "auto":
            think_cfg = thinking_level

        # Active optimizations list
        active_opts = []
        # Research subsystems are not active merely because a classifier chose
        # a task.  Runtime telemetry is the only source of active features.
        if use_agt: active_opts.append("agentic_pipeline")
        active_opts.append(f"thinking_{think_cfg}")

        # Step 3: Run native generation
        exe_path = C_DIR / "qwnrun.exe"
        cmd = [
            str(exe_path),
            str(self.model_path),
            prompt,
            str(max_tokens),
            "4096",
            "--thinking", think_cfg,
        ]

        text = ""
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60.0,
            )
            text = proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            text = ""

        return AutoPilotResponse(
            text=text,
            speedup=None,
            tokens_per_second=None,
            active_optimizations=active_opts,
            quality_score=None,
            memory_usage_gb=None,
            task_type=resolved_task.value,
            thinking_level=think_cfg,
        )
