#!/usr/bin/env python3
"""Dependency-free OpenAI-compatible HTTP gateway for the qwanto engine."""

import argparse
import codecs
import collections
import contextlib
import hashlib
import json
import math
import mimetypes
import os
import select
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import urllib.request
import dataclasses
from typing import List, Dict, Any, Iterator, Union
import backends
import orchestrator
import io
import zipfile


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.resolve()


def _is_safe_path(target_path: Union[str, Path], allowed_dirs: List[Path] = None) -> bool:
    """Audit and validate that target_path resides safely within project or allowed directories."""
    try:
        resolved = Path(target_path).resolve()
        if allowed_dirs:
            return any(resolved == d.resolve() or d.resolve() in resolved.parents for d in allowed_dirs)
        # Default safety boundary: project root or user app data
        return resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents
    except Exception:
        return False


def _qwn_executable(engine):
    suffix = ".exe" if sys.platform == "win32" else ""
    candidate = Path(engine).with_name("qwnrun" + suffix)
    return candidate


def _ensure_llama_server() -> str | None:
    """Find llama-server in PATH or project dir; if missing, download it."""
    import shutil
    # 1. Check PATH
    exe = shutil.which("llama-server")
    if exe:
        return exe
    # 2. Check project directory
    local_exe = HERE / "llama-server.exe"
    if local_exe.exists():
        return str(local_exe)
    # 3. Detect GPU vendor for optimal backend selection
    gpu_vendor = "unknown"
    try:
        result = subprocess.run(["wmic", "path", "win32_videocontroller", "get", "name"], capture_output=True, text=True, timeout=5)
        gpu_name = result.stdout.lower()
        if "nvidia" in gpu_name or "geforce" in gpu_name or "rtx" in gpu_name or "gtx" in gpu_name:
            gpu_vendor = "nvidia"
        elif "amd" in gpu_name or "radeon" in gpu_name:
            gpu_vendor = "amd"
        elif "intel" in gpu_name:
            gpu_vendor = "intel"
    except Exception:
        pass
    print(f"Detected GPU vendor: {gpu_vendor}", file=sys.stderr)
    # 4. Download from GitHub
    print("Downloading llama-server from GitHub...", file=sys.stderr)
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest",
            headers={"Accept": "application/json", "User-Agent": "qwanto/1.0"}
        )
        resp = urllib.request.urlopen(req, timeout=15)
        release = json.loads(resp.read().decode("utf-8"))
        tag = release.get("tag_name", "")
        assets = release.get("assets", [])
        # Find best Windows build based on detected GPU
        def asset_score(name):
            n = name.lower()
            if not (n.startswith("llama-") and "win" in n and n.endswith(".zip")):
                return -1
            score = 0
            # Prefer CUDA for NVIDIA, Vulkan for AMD/Intel/others
            if gpu_vendor == "nvidia":
                if "cuda" in n: score = 10
                elif "vulkan" in n: score = 5
            else:
                if "vulkan" in n: score = 10
                elif "cuda" in n: score = 4
            if "hip" in n: score = max(score, 2)
            if "opencl" in n: score = max(score, 1)
            if "arm64" in n: score = 0
            return score
        zip_url = None
        best_score = -1
        for a in assets:
            s = asset_score(a.get("name", ""))
            if s > best_score:
                best_score = s
                zip_url = a.get("browser_download_url")
        if not zip_url:
            print(f"Could not find Windows release asset in {tag}", file=sys.stderr)
            return None
        fname = zip_url.split('/')[-1]
        print(f"Downloading {fname}...", file=sys.stderr)
        zip_req = urllib.request.Request(zip_url, headers={"User-Agent": "qwanto/1.0"})
        zip_resp = urllib.request.urlopen(zip_req, timeout=300)
        total = int(zip_resp.headers.get('Content-Length', 0))
        downloaded = 0
        chunks = []
        while True:
            chunk = zip_resp.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
            downloaded += len(chunk)
            if total:
                pct = int(downloaded * 100 / total)
                print(f"\r  Downloading {fname}... {pct}% ({downloaded/1024/1024:.0f}/{total/1024/1024:.0f} MB)", file=sys.stderr, end="")
            else:
                print(f"\r  Downloading {fname}... {downloaded/1024/1024:.0f} MB", file=sys.stderr, end="")
        print(file=sys.stderr)
        data = io.BytesIO(b"".join(chunks))
        print(f"Extracting bundle to {HERE}...", file=sys.stderr)
        extracted = []
        found = False
        with zipfile.ZipFile(data) as zf:
            for name in zf.namelist():
                stem = os.path.basename(name)
                if not stem:
                    continue
                target = HERE / stem
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                extracted.append(target)
                if stem == "llama-server.exe":
                    found = True
                print(f"  Extracted {stem}", file=sys.stderr)
        if found:
            print(f"Downloaded llama-server.exe (+ dependencies) to {HERE}", file=sys.stderr)
            return str(local_exe)
        # Clean up partial extraction on failure
        for p in extracted:
            try:
                p.unlink()
            except OSError:
                pass
        print("llama-server.exe not found in downloaded archive", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Failed to download llama-server: {e}", file=sys.stderr)
        return None


_LLAMA_HELP_CACHE: collections.OrderedDict = collections.OrderedDict()
_MAX_HELP_CACHE_SIZE = 32

ALLOWED_KV_QUANTS = ("f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0")


def _llama_help(exe) -> str:
    """`llama-server --help` output, cached — used to adapt flags to the
    installed llama.cpp version (e.g. -fa boolean vs -fa on|off|auto)."""
    exe = str(exe)
    if exe in _LLAMA_HELP_CACHE:
        _LLAMA_HELP_CACHE.move_to_end(exe)
        return _LLAMA_HELP_CACHE[exe]
    try:
        out = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=15)
        text = (out.stdout or "") + (out.stderr or "")
    except Exception:
        text = ""
    if len(_LLAMA_HELP_CACHE) >= _MAX_HELP_CACHE_SIZE:
        _LLAMA_HELP_CACHE.popitem(last=False)
    _LLAMA_HELP_CACHE[exe] = text
    return text


def _build_llama_cmd(exe, model_path, ctx_size, threads, server=None, port=8080):
    """Build the llama-server command line: orchestrator plan (GPU/RAM/CPU/
    NVMe split) + acceleration flags (-fa, KV quant, speculative decoding),
    each adapted to the flags the installed llama-server actually supports."""
    fa = bool(getattr(server, "flash_attention", True))
    kvq = str(getattr(server, "kv_cache_quant", "q4_0"))
    spec = bool(getattr(server, "speculative_decoding", False))
    draft = str(getattr(server, "draft_model_path", "") or "")
    resources = getattr(server, "resources", None) or {"cpu": 100}

    plan = None
    try:
        plan = orchestrator.plan(model_path, ctx_size=int(ctx_size),
                                 kv_cache_quant=kvq if fa else "f16",
                                 cpu_limit=int(resources.get("cpu", 100)))
        print("[orchestrator]\n  " + orchestrator.describe(plan).replace("\n", "\n  "),
              file=sys.stderr)
    except Exception as e:
        print(f"[orchestrator] planning failed ({e}); using defaults", file=sys.stderr)

    ngl, batch, ubatch, tensor_split = 999, 512, 512, None
    if plan:
        ngl, batch, ubatch = plan["ngl"], plan["batch"], plan["ubatch"]
        tensor_split = plan.get("tensor_split")
        threads = plan["threads"]

    help_text = _llama_help(exe)
    cmd = [str(exe), "-m", str(model_path), "--mmap",
           "--port", str(port), "--host", "0.0.0.0",
           "--ctx-size", str(ctx_size), "--parallel", "2",
           "-ngl", str(ngl), "-t", str(threads),
           "-b", str(batch), "-ub", str(ubatch)]

    if tensor_split and "--tensor-split" in help_text:
        cmd += ["-ts", tensor_split]

    # Flash Attention: recent llama.cpp takes on|off|auto, older is boolean
    if "--flash-attn" in help_text:
        import re as _re
        takes_value = bool(_re.search(r"flash-attn[^\n]*\bauto\b", help_text))
        if fa:
            cmd += ["-fa", "on"] if takes_value else ["-fa"]
        elif takes_value:
            cmd += ["-fa", "off"]

    # KV cache quantization (V-cache quant requires flash attention)
    if kvq in ALLOWED_KV_QUANTS and kvq != "f16" and "--cache-type-k" in help_text:
        cmd += ["-ctk", kvq]
        if fa and "--cache-type-v" in help_text:
            cmd += ["-ctv", kvq]

    # Speculative decoding with a small draft model
    if spec and draft and os.path.exists(draft) and "--model-draft" in help_text:
        cmd += ["-md", draft]
        if "--draft-max" in help_text:
            cmd += ["--draft-max", "16", "--draft-min", "4"]
        elif "--draft " in help_text:
            cmd += ["--draft", "16"]
        if "--gpu-layers-draft" in help_text:
            cmd += ["-ngld", "999"]
        print(f"[accel] speculative decoding: draft={draft}", file=sys.stderr)

    return cmd


END = b"\x01\x01END\x01\x01\n"
READY = b"\x01\x01READY\x01\x01\n"
MAX_BODY = 4 << 20
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://tauri.localhost",
    "tauri://localhost",
)


class APIError(Exception):
    def __init__(self, status, message, param=None, code=None, error_type="invalid_request_error",
                 headers=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code
        self.error_type = error_type
        self.headers = headers or {}


class ClientCancelled(Exception):
    pass


def error_object(error):
    return {"error": {"message": error.message, "type": error.error_type,
                      "param": error.param, "code": error.code}}


class GenerationScheduler:
    """Bounded FIFO admission for the engine's independent KV contexts."""

    def __init__(self, max_queue=8, queue_timeout=300, capacity=1):
        if max_queue < 0:
            raise ValueError("max_queue cannot be negative")
        if queue_timeout <= 0:
            raise ValueError("queue_timeout must be positive")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self.capacity = capacity
        self.free_slots = set(range(capacity))
        self.condition = threading.Condition()
        self.queue = collections.deque()
        self.active = 0
        self.closed = False
        self.admitted = 0
        self.completed = 0
        self.rejected = 0
        self.timed_out = 0
        self.cancelled = 0

    @contextlib.contextmanager
    def admit(self, cancelled=None, slot=None):
        ticket = object()
        queued_at = time.monotonic()
        with self.condition:
            if self.closed:
                raise APIError(503, "The inference scheduler is shutting down.", None,
                               "scheduler_closed", "server_error")
            if (self.active >= self.capacity or self.queue) and len(self.queue) >= self.max_queue:
                self.rejected += 1
                raise APIError(429, "The inference queue is full.", None, "queue_full",
                               "rate_limit_error", {"Retry-After": "1"})
            self.queue.append(ticket)
            deadline = queued_at + self.queue_timeout
            while True:
                if self.closed:
                    self.queue.remove(ticket)
                    self.condition.notify_all()
                    raise APIError(503, "The inference scheduler is shutting down.", None,
                                   "scheduler_closed", "server_error")
                available = min(self.free_slots) if slot is None and self.free_slots else slot
                if self.queue[0] is ticket and available in self.free_slots:
                    break
                if cancelled and cancelled():
                    self.queue.remove(ticket)
                    self.cancelled += 1
                    self.condition.notify_all()
                    raise ClientCancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.queue.remove(ticket)
                    self.timed_out += 1
                    self.condition.notify_all()
                    raise APIError(429, "Timed out waiting for the inference engine.", None,
                                   "queue_timeout", "rate_limit_error", {"Retry-After": "1"})
                self.condition.wait(min(remaining, 0.25))
            self.queue.popleft()
            self.free_slots.remove(available)
            self.active += 1
            self.admitted += 1
            wait_seconds = time.monotonic() - queued_at
        try:
            yield wait_seconds, available
        finally:
            with self.condition:
                self.active -= 1
                self.free_slots.add(available)
                self.completed += 1
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"active": self.active, "queued": len(self.queue),
                    "capacity": self.capacity,
                    "max_queue": self.max_queue, "queue_timeout_seconds": self.queue_timeout,
                    "admitted": self.admitted, "completed": self.completed,
                    "rejected": self.rejected, "timed_out": self.timed_out,
                    "cancelled": self.cancelled}

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def content_text(content, param):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise APIError(400, "Message content must be a string or an array of text parts.", param)
    parts = []
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") not in ("text", "input_text"):
            raise APIError(400, "Colibri currently supports text message content only.",
                           f"{param}.{index}", "unsupported_content_type")
        if not isinstance(part.get("text"), str):
            raise APIError(400, "Text content parts require a string `text` field.",
                           f"{param}.{index}.text")
        parts.append(part["text"])
    return "".join(parts)


# ---- GLM-5.2 tool calling -----------------------------------------------------------------
# The model expresses tool calls as ordinary text (from chat_template.jinja):
#   <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
# and tool results come back as <|observation|><tool_response>{content}</tool_response>.
# We render those markers into the prompt and parse them back into OpenAI `tool_calls`.
import re

BOX_START, BOX_END = "<tool_call>", "</tool_call>"
TR_OPEN,  TR_CLOSE = "<tool_response>", "</tool_response>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

_BOX_RE  = re.compile(re.escape(BOX_START) + r"(.*?)" + re.escape(BOX_END), re.DOTALL)
_ARG_RE  = re.compile(r"<arg_key>([^<]*)</arg_key><arg_value>(.*?)</arg_value>", re.DOTALL)
_NAME_RE = re.compile(r"\s*([A-Za-z0-9_.\-]+)")
_TAG_RE  = re.compile(r"</?arg_key>|</?arg_value>")

# De-mangler: opt-in recovery for heavily-quantized models that drop the
# <arg_key>K</arg_key><arg_value> structure. Default OFF (never rewrites well-formed output).
_SALVAGE = os.environ.get("QWANTO_TOOL_SALVAGE", "0") == "1"


def _tool_param_order(tools):
    """name -> ordered param names (required first) from the request schema, for de-mangling."""
    out = {}
    for tool in (tools or []):
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        params = ((fn.get("parameters") or {}).get("properties") or {})
        required = list((fn.get("parameters") or {}).get("required") or [])
        out[name] = required + [p for p in params if p not in required]
    return out


def _tool_param_types(tools):
    """name -> {param: declared JSON-schema type}. The model emits every argument as text;
    without the schema a string-typed value that happens to look numeric ("12345" for an
    order id, an SKU, a phone number) would be json.loads()'d into an int and the tool would
    receive the wrong type."""
    out = {}
    for tool in (tools or []):
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        types = {}
        for key, spec in props.items():
            if isinstance(spec, dict):
                t = spec.get("type")
                if isinstance(t, list):          # {"type": ["string", "null"]}
                    t = next((x for x in t if x != "null"), None)
                types[key] = t
        out[name] = types
    return out


def _coerce_arg(value, declared):
    """Decode a raw <arg_value> according to the declared schema type.

    A string-typed parameter is kept verbatim -- never parsed as JSON. Everything else keeps
    the previous permissive behaviour (parse if it parses, otherwise leave as text)."""
    if declared == "string":
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
    if declared in ("integer", "number") and isinstance(parsed, bool):
        return value                              # `true` is not a number
    if declared and declared not in ("integer", "number", "boolean", "object", "array"):
        return value
    return parsed


def parse_tool_calls(reply, tools=None):
    """Return (content, tool_calls). Strict GLM parse; optional de-mangler (QWANTO_TOOL_SALVAGE=1)
    rescues malformed int4 output by mapping a lone payload onto the tool's primary parameter."""
    param_order = _tool_param_order(tools)
    param_types = _tool_param_types(tools)
    calls, salvaged = [], []
    for match in _BOX_RE.finditer(reply):
        inner = match.group(1)
        name_match = _NAME_RE.match(inner)
        name = name_match.group(1) if name_match else inner.strip()
        args = {}
        types = param_types.get(name, {})
        for arg in _ARG_RE.finditer(inner):
            key, value = arg.group(1), arg.group(2)
            args[key] = _coerce_arg(value, types.get(key))
        if not args and _SALVAGE:
            rest = inner[name_match.end():] if name_match else ""
            payload = _TAG_RE.sub("", rest).strip()
            if payload.startswith("(") and payload.endswith(")"):
                payload = payload[1:-1].strip()
            if payload:
                key = (param_order.get(name) or ["input"])[0]
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                args = {key: payload}
                salvaged.append(name)
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    text = _BOX_RE.sub("", reply)
    if THINK_CLOSE in text:
        text = text.split(THINK_CLOSE, 1)[1]
    text = text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    if calls:
        dm = len(salvaged)
        sys.stderr.write("[api] tool-calls: %d total, %d strict, %d de-mangled [%s]%s\n"
                         % (len(calls), len(calls) - dm, dm, "CLEAN" if dm == 0 else "DE-MANGLED",
                            (" -> " + ", ".join(salvaged)) if dm else ""))
        sys.stderr.flush()
    return text.strip(), calls


class StreamParser:
    def __init__(self, param_types):
        self.param_types = param_types
        self.state = "TEXT"
        self.buf = ""
        self.tc_index = 0
        self.call_id = None
        self.name = ""
        self.arg_key = ""
        self.first_arg = True
        
        # Lookahead buffering bounds
        # Max length of any marker we search for
        self.MAX_LOOKAHEAD = max(len(m) for m in [
            BOX_START, BOX_END, THINK_OPEN, THINK_CLOSE,
            "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"
        ])

    def add_chunk(self, text):
        """Processes an incoming string chunk and returns a list of delta dictionaries."""
        self.buf += text
        deltas = []
        
        while True:
            if self.state == "TEXT":
                pos_box = self.buf.find(BOX_START)
                pos_think = self.buf.find(THINK_OPEN)
                
                # Find earliest marker
                pos = -1
                next_state = None
                marker_len = 0
                
                if pos_box >= 0:
                    pos = pos_box
                    next_state = "TOOL_NAME"
                    marker_len = len(BOX_START)
                
                if pos_think >= 0 and (pos == -1 or pos_think < pos):
                    pos = pos_think
                    next_state = "THINKING"
                    marker_len = len(THINK_OPEN)
                
                if pos >= 0:
                    if pos > 0:
                        deltas.append({"content": self.buf[:pos]})
                    self.buf = self.buf[pos + marker_len:]
                    self.state = next_state
                    if next_state == "TOOL_NAME":
                        self.call_id = "call_" + uuid.uuid4().hex[:24]
                        self.name = ""
                        self.first_arg = True
                    continue
                else:
                    # Flush safe portion
                    flush = max(0, len(self.buf) - self.MAX_LOOKAHEAD)
                    if flush > 0:
                        deltas.append({"content": self.buf[:flush]})
                        self.buf = self.buf[flush:]
                    break

            elif self.state == "THINKING":
                pos = self.buf.find(THINK_CLOSE)
                if pos >= 0:
                    if pos > 0:
                        deltas.append({"reasoning_content": self.buf[:pos]})
                    self.buf = self.buf[pos + len(THINK_CLOSE):]
                    self.state = "TEXT"
                    continue
                else:
                    flush = max(0, len(self.buf) - self.MAX_LOOKAHEAD)
                    if flush > 0:
                        deltas.append({"reasoning_content": self.buf[:flush]})
                        self.buf = self.buf[flush:]
                    break

            elif self.state == "TOOL_NAME":
                pos_ak = self.buf.find("<arg_key>")
                pos_end = self.buf.find(BOX_END)
                
                pos = -1
                next_state = None
                marker_len = 0
                
                if pos_ak >= 0:
                    pos = pos_ak
                    next_state = "TOOL_KEY"
                    marker_len = len("<arg_key>")
                
                if pos_end >= 0 and (pos == -1 or pos_end < pos):
                    pos = pos_end
                    next_state = "TEXT"
                    marker_len = len(BOX_END)
                
                if pos >= 0:
                    self.name = self.buf[:pos].strip()
                    self.buf = self.buf[pos + marker_len:]
                    self.state = next_state
                    
                    deltas.append({
                        "tool_calls": [{
                            "index": self.tc_index,
                            "id": self.call_id,
                            "type": "function",
                            "function": {"name": self.name, "arguments": "{"}
                        }]
                    })
                    
                    if next_state == "TEXT":
                        deltas.append({
                            "tool_calls": [{
                                "index": self.tc_index,
                                "function": {"arguments": "}"}
                            }]
                        })
                        self.tc_index += 1
                    else:
                        self.arg_key = ""
                    continue
                else:
                    break # Buffer more for tool name (usually very short)

            elif self.state == "TOOL_WAIT_KEY":
                pos_ak = self.buf.find("<arg_key>")
                pos_end = self.buf.find(BOX_END)
                
                pos = -1
                next_state = None
                marker_len = 0
                
                if pos_ak >= 0:
                    pos = pos_ak
                    next_state = "TOOL_KEY"
                    marker_len = len("<arg_key>")
                
                if pos_end >= 0 and (pos == -1 or pos_end < pos):
                    pos = pos_end
                    next_state = "TEXT"
                    marker_len = len(BOX_END)
                
                if pos >= 0:
                    self.buf = self.buf[pos + marker_len:]
                    self.state = next_state
                    
                    if next_state == "TEXT":
                        deltas.append({
                            "tool_calls": [{
                                "index": self.tc_index,
                                "function": {"arguments": "}"}
                            }]
                        })
                        self.tc_index += 1
                    else:
                        self.arg_key = ""
                    continue
                else:
                    break

            elif self.state == "TOOL_KEY":
                pos = self.buf.find("</arg_key>")
                if pos >= 0:
                    self.arg_key = self.buf[:pos].strip()
                    self.buf = self.buf[pos + len("</arg_key>"):]
                    self.state = "TOOL_WAIT_VAL"
                    continue
                else:
                    break

            elif self.state == "TOOL_WAIT_VAL":
                pos = self.buf.find("<arg_value>")
                if pos >= 0:
                    self.buf = self.buf[pos + len("<arg_value>"):]
                    self.state = "TOOL_VAL"
                    prefix = "" if self.first_arg else ", "
                    self.first_arg = False
                    
                    declared = self.param_types.get(self.name, {}).get(self.arg_key, "string")
                    key_str = json.dumps(self.arg_key)
                    if declared == "string":
                        deltas.append({
                            "tool_calls": [{
                                "index": self.tc_index,
                                "function": {"arguments": f"{prefix}{key_str}: \""}
                            }]
                        })
                    else:
                        deltas.append({
                            "tool_calls": [{
                                "index": self.tc_index,
                                "function": {"arguments": f"{prefix}{key_str}: "}
                            }]
                        })
                    continue
                else:
                    break

            elif self.state == "TOOL_VAL":
                pos = self.buf.find("</arg_value>")
                declared = self.param_types.get(self.name, {}).get(self.arg_key, "string")
                
                if pos >= 0:
                    chunk = self.buf[:pos]
                    self.buf = self.buf[pos + len("</arg_value>"):]
                    self.state = "TOOL_WAIT_KEY" # Wait for next key or BOX_END
                    
                    if declared == "string":
                        if chunk:
                            val_str = json.dumps(chunk)[1:-1]
                        else:
                            val_str = ""
                        deltas.append({
                            "tool_calls": [{
                                "index": self.tc_index,
                                "function": {"arguments": val_str + "\""}
                            }]
                        })
                    else:
                        # Emitting raw non-string JSON character sequence safely
                        deltas.append({
                            "tool_calls": [{
                                "index": self.tc_index,
                                "function": {"arguments": chunk}
                            }]
                        })
                    continue
                else:
                    flush = max(0, len(self.buf) - self.MAX_LOOKAHEAD)
                    if flush > 0:
                        chunk = self.buf[:flush]
                        self.buf = self.buf[flush:]
                        if declared == "string":
                            val_str = json.dumps(chunk)[1:-1]
                            deltas.append({
                                "tool_calls": [{
                                    "index": self.tc_index,
                                    "function": {"arguments": val_str}
                                }]
                            })
                        else:
                            deltas.append({
                                "tool_calls": [{
                                    "index": self.tc_index,
                                    "function": {"arguments": chunk}
                                }]
                            })
                    break

        return deltas

    def finalize(self):
        """Called at end of stream. Flushes any remaining buffer."""
        deltas = []
        if self.state == "TEXT" and self.buf:
            deltas.append({"content": self.buf})
        elif self.state == "THINKING" and self.buf:
            deltas.append({"reasoning_content": self.buf})
        elif self.state == "TOOL_VAL":
            declared = self.param_types.get(self.name, {}).get(self.arg_key, "string")
            if declared == "string" and self.buf:
                val_str = json.dumps(self.buf)[1:-1]
                deltas.append({
                    "tool_calls": [{
                        "index": self.tc_index,
                        "function": {"arguments": val_str}
                    }]
                })
            elif self.buf:
                deltas.append({
                    "tool_calls": [{
                        "index": self.tc_index,
                        "function": {"arguments": self.buf}
                    }]
                })
        
        # Cleanup dangling JSON formatting if aborted mid-tool call
        if self.state in ("TOOL_NAME", "TOOL_KEY", "TOOL_WAIT_VAL", "TOOL_WAIT_KEY"):
            # They already got '{' but didn't finish
            deltas.append({"tool_calls": [{"index": self.tc_index, "function": {"arguments": "}"}}]})
        elif self.state == "TOOL_VAL":
            declared = self.param_types.get(self.name, {}).get(self.arg_key, "string")
            if declared == "string":
                deltas.append({"tool_calls": [{"index": self.tc_index, "function": {"arguments": "\"}"}}]})
            else:
                deltas.append({"tool_calls": [{"index": self.tc_index, "function": {"arguments": "}"}}]})

        self.buf = ""
        return deltas


def render_chat(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                tool_choice=None):
    """Render the text-only subset of the official GLM-5.2 chat template."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    prompt = ["[gMASK]<sop>"]
    if enable_thinking:
        effort = "High" if reasoning_effort == "high" else "Max"
        prompt.append(f"<|system|>Reasoning Effort: {effort}")
    forced = None
    if isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name")
                  or tool_choice.get("name"))
        if forced:
            tools = [t for t in (tools or [])
                     if ((t.get("function", t) if isinstance(t, dict) else {}).get("name") == forced)]
    elif tool_choice == "none":
        tools = None                              # the client forbade tools: do not offer them
    if tools:
        # AUTHORITATIVE GLM-5.2 tool-declaration block (byte-matches chat_template.jinja): the
        # `# Tools` + <tools></tools> XML structure is what the model was trained on. A made-up
        # preamble makes it hallucinate other frameworks' syntax (e.g. `end_action`).
        prompt.append("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the "
                      "user query.\n\nYou are provided with function signatures within <tools></tools> "
                      "XML tags:\n<tools>\n")
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            clean = {k: v for k, v in fn.items() if k not in ("defer_loading", "strict")}
            prompt.append(json.dumps(clean, ensure_ascii=False) + "\n")
        prompt.append("</tools>\n\nFor each function call, output the function name and arguments "
                      "within the following XML format:\n<tool_call>{function-name}"
                      "<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
                      "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>")
        if forced:
            prompt.append(f"\n\nYou must call the function `{forced}`. Do not answer directly.")
        elif tool_choice == "required":
            prompt.append("\n\nYou must call one of the functions above. Do not answer directly.")
    prev_tool = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role in ("system", "developer"):
            prompt.append(f"<|system|>{content_text(message.get('content'), f'messages.{index}.content')}")
        elif role == "user":
            prompt.append(f"<|user|>{content_text(message.get('content'), f'messages.{index}.content')}")
        elif role == "assistant":
            # content may be null when the message is purely tool_calls
            raw = message.get("content")
            text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
            prompt.append(f"<|assistant|><think></think>{text.strip()}")
            for tc in (message.get("tool_calls") or []):
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                prompt.append(BOX_START + (fn.get("name") or ""))
                for key, value in (args or {}).items():
                    prompt.append(f"<arg_key>{key}</arg_key><arg_value>"
                                  + (value if isinstance(value, str)
                                     else json.dumps(value, ensure_ascii=False)) + "</arg_value>")
                prompt.append(BOX_END)
        elif role == "tool":
            if not prev_tool:                       # one <|observation|> per consecutive tool run
                prompt.append("<|observation|>")
            prompt.append(TR_OPEN + content_text(message.get("content"), f"messages.{index}.content") + TR_CLOSE)
        else:
            raise APIError(400, f"Unsupported message role: {role!r}.",
                           f"messages.{index}.role", "unsupported_role")
        prev_tool = (role == "tool")
    prompt.append("<|assistant|><think>" if enable_thinking else
                  "<|assistant|><think></think>")
    return "".join(prompt)


def generation_options(body, limit):
    if body.get("n", 1) != 1:
        raise APIError(400, "Colibri currently supports `n=1` only.", "n", "unsupported_value")
    # `tools`/`functions` are handled by render_chat (declaration) + parse_tool_calls (output).
    choice = body.get("tool_choice")
    if choice is not None:
        if isinstance(choice, str):
            if choice not in ("auto", "none", "required"):
                raise APIError(400, "`tool_choice` must be one of \"auto\", \"none\", \"required\", "
                                    "or a function object.", "tool_choice", "unsupported_value")
        elif isinstance(choice, dict):
            name = (choice.get("function") or {}).get("name") or choice.get("name")
            if not name:
                raise APIError(400, "`tool_choice` function object must include a name.",
                               "tool_choice", "invalid_value")
            declared = [(t.get("function", t) if isinstance(t, dict) else {}).get("name")
                        for t in (body.get("tools") or body.get("functions") or [])]
            if name not in declared:
                raise APIError(400, f"`tool_choice` names {name!r}, which is not in `tools`.",
                               "tool_choice", "invalid_value")
        else:
            raise APIError(400, "`tool_choice` must be a string or a function object.",
                           "tool_choice", "invalid_value")
        if choice != "none" and not (body.get("tools") or body.get("functions")):
            raise APIError(400, "`tool_choice` requires `tools`.", "tool_choice", "invalid_value")
    if body.get("stop") is not None:
        raise APIError(400, "Custom stop sequences are not supported yet.", "stop", "unsupported_parameter")
    if body.get("logprobs"):
        raise APIError(400, "Log probabilities are not supported yet.", "logprobs", "unsupported_parameter")
    if body.get("frequency_penalty", 0) or body.get("presence_penalty", 0):
        raise APIError(400, "Token penalties are not supported yet.", None, "unsupported_parameter")
    if body.get("seed") is not None:
        raise APIError(400, "Per-request seeds are not supported yet.", "seed", "unsupported_parameter")
    response_format = body.get("response_format")
    if response_format not in (None, {"type": "text"}):
        raise APIError(400, "Only the default text response format is supported.",
                       "response_format", "unsupported_parameter")

    maximum = body.get("max_completion_tokens")
    maximum_param = "max_completion_tokens"
    if maximum is None:
        maximum = body.get("max_tokens")
        maximum_param = "max_tokens"
    if maximum is None:
        maximum = min(256, limit)
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    temperature = 0.7 if temperature is None else temperature
    top_p = 0.9 if top_p is None else top_p
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise APIError(400, f"`{maximum_param}` must be a positive integer.", maximum_param)
    if maximum > limit:
        maximum = limit   # clamp to the server's --max-tokens cap instead of 400 (#260): OpenAI
                          # clients (opencode/ai-sdk) default to large max_tokens; rejecting breaks them.
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or
            not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise APIError(400, "`temperature` must be between 0 and 2.", "temperature")
    if (isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or
            not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise APIError(400, "`top_p` must be greater than 0 and at most 1.", "top_p")
    return maximum, float(temperature), float(top_p)


def read_engine_turn(stream, sentinel, on_bytes):
    pending = b""
    while True:
        byte = stream.read(1)
        if byte == b"":
            raise RuntimeError("colibri engine exited unexpectedly")
        pending += byte
        if pending.endswith(sentinel):
            data = pending[:-len(sentinel)]
            if data:
                on_bytes(data)
            break
        if len(pending) > len(sentinel):
            on_bytes(pending[:-len(sentinel)])
            pending = pending[-len(sentinel):]

    fields = stream.readline().decode("utf-8", "replace").strip().split()
    if len(fields) < 5 or fields[0] != "STAT":
        raise RuntimeError(f"invalid engine status: {' '.join(fields)}")
    return {
        "completion_tokens": int(fields[1]),
        "tokens_per_second": float(fields[2]),
        "cache_hit_percent": float(fields[3]),
        "rss_gb": float(fields[4]),
        "prompt_tokens": int(fields[5]) if len(fields) > 5 else 0,
        "length_limited": bool(int(fields[6])) if len(fields) > 6 else False,
    }


class Engine:
    def __init__(self, executable, model, cap=8, max_tokens=1024, env=None, kv_slots=1):
        child_env = dict(env or os.environ, SNAP=str(model), SERVE="1", SERVE_BATCH="1",
                         NGEN=str(max_tokens), KV_SLOTS=str(kv_slots))
        self.process = subprocess.Popen(
            [str(executable), str(cap)], env=child_env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, bufsize=0,
        )
        self.write_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending = {}
        self.next_request_id = 1
        self.closed = False
        self.dispatcher_error = None
        self.kv_slots = kv_slots
        self.tiers = None
        self.hwinfo = None
        self.emap = None
        self.hits = None
        self.hits_seq = 0                      # latest "TIERS" snapshot from the engine
        read_engine_turn(self.process.stdout, READY, lambda _: None)
        self.dispatcher = threading.Thread(target=self._dispatch_stdout,
                                            name="qwanto-stdout", daemon=True)
        self.dispatcher.start()

    @staticmethod
    def _stats(fields):
        if len(fields) < 5 or fields[0] != "STAT":
            raise RuntimeError(f"invalid engine status: {' '.join(fields)}")
        return {
            "completion_tokens": int(fields[1]),
            "tokens_per_second": float(fields[2]),
            "cache_hit_percent": float(fields[3]),
            "rss_gb": float(fields[4]),
            "prompt_tokens": int(fields[5]) if len(fields) > 5 else 0,
            "length_limited": bool(int(fields[6])) if len(fields) > 6 else False,
        }

    def _fail_pending(self, error):
        with self.pending_lock:
            requests = list(self.pending.values())
            self.pending.clear()
        for events in requests:
            events.put(("error", error))

    def _read_exact(self, size):
        chunks = []
        remaining = size
        while remaining:
            chunk = self.process.stdout.read(remaining)
            if chunk == b"":
                raise RuntimeError("truncated engine DATA payload")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _dispatch_stdout(self):
        try:
            while True:
                line = self.process.stdout.readline()
                if line == b"":
                    raise RuntimeError("qwanto engine exited unexpectedly")
                fields = line.decode("utf-8", "replace").strip().split()
                if not fields:
                    continue
                kind = fields[0]
                if kind == "DATA" and len(fields) == 3:
                    request_id = fields[1]
                    size = int(fields[2])
                    if not 0 <= size <= 65536:
                        raise RuntimeError("invalid engine DATA size")
                    data = self._read_exact(size)
                    if self._read_exact(1) != b"\n":
                        raise RuntimeError("invalid engine DATA terminator")
                    with self.pending_lock:
                        events = self.pending.get(request_id)
                    if events is not None:
                        events.put(("data", data))
                elif kind == "DONE" and len(fields) >= 7:
                    request_id = fields[1]
                    stats = self._stats(fields[2:])
                    with self.pending_lock:
                        events = self.pending.pop(request_id, None)
                    if events is not None:
                        events.put(("done", stats))
                elif kind == "HWINFO" and len(fields) >= 7:
                    parts = " ".join(fields[6:]).split("|")
                    self.hwinfo = {"cores": int(fields[1]), "ram_total_gb": float(fields[2]),
                                   "ram_avail_gb": float(fields[3]), "gpus": int(fields[4]),
                                   "vram_total_gb": float(fields[5]),
                                   "cpu": parts[0].strip() if len(parts)>0 else "",
                                   "gpu": parts[1].strip() if len(parts)>1 else ""}
                elif kind == "EMAP" and len(fields) == 4:
                    self.emap = {"rows": int(fields[1]), "cols": int(fields[2]), "map": fields[3]}
                elif kind == "HITS" and len(fields) == 4:
                    self.hits = fields[3]
                    self.hits_seq += 1
                elif kind == "TIERS" and len(fields) >= 6:
                    self.tiers = {"vram": int(fields[1]), "ram": int(fields[2]),
                                  "disk": int(fields[3]), "vram_gb": float(fields[4]),
                                  "ram_gb": float(fields[5])}
                elif kind == "ERROR" and len(fields) >= 2:
                    request_id = fields[1]
                    message = " ".join(fields[2:]) or "engine request failed"
                    with self.pending_lock:
                        events = self.pending.pop(request_id, None)
                    if events is not None:
                        events.put(("error", RuntimeError(message)))
                else:
                    raise RuntimeError(f"invalid engine response: {' '.join(fields)}")
        except Exception as error:
            if not self.closed:
                self.dispatcher_error = error
                self._fail_pending(error)

    def generate(self, prompt, max_tokens, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None):
        if isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or not 0 <= cache_slot < self.kv_slots:
            raise APIError(400, "Invalid cache slot.", "cache_slot")
        payload = prompt.encode("utf-8")
        if b"\0" in payload:
            raise APIError(400, "NUL bytes are not supported in prompts.", "messages")
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def decode(data):
            text = decoder.decode(data)
            if text:
                on_text(text)

        events = queue.Queue()
        with self.pending_lock:
            if self.closed:
                raise RuntimeError("colibri engine is shutting down")
            if self.dispatcher_error is not None:
                raise RuntimeError("colibri engine dispatcher stopped") from self.dispatcher_error
            if self.process.poll() is not None:
                raise RuntimeError("colibri engine is not running")
            request_id = str(self.next_request_id)
            self.next_request_id += 1
            self.pending[request_id] = events
        header = (f"SUBMIT {request_id} {cache_slot} {len(payload)} {max_tokens} "
                  f"{temperature:.8g} {top_p:.8g}\n").encode()
        try:
            with self.write_lock:
                if self.process.poll() is not None:
                    raise RuntimeError("colibri engine is not running")
                self.process.stdin.write(header + payload + b"\n")
                self.process.stdin.flush()
        except Exception:
            with self.pending_lock:
                self.pending.pop(request_id, None)
            raise

        cancel_sent = False
        while True:
            kind, value = events.get()
            if kind == "data":
                if not cancel_sent:
                    decode(value)
                    if cancelled and cancelled():
                        cancel_sent = True
                        with self.write_lock:
                            self.process.stdin.write(f"CANCEL {request_id}\n".encode())
                            self.process.stdin.flush()
            elif kind == "done":
                tail = decoder.decode(b"", final=True)
                if tail:
                    on_text(tail)
                return value
            elif cancel_sent and isinstance(value, RuntimeError) and str(value) == "CANCELLED":
                raise ClientCancelled()
            else:
                raise value

    def close(self):
        with self.pending_lock:
            if self.closed:
                return
            self.closed = True
        self._fail_pending(RuntimeError("qwanto engine is shutting down"))
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.dispatcher is not threading.current_thread():
            self.dispatcher.join(timeout=5)


def model_object(model_id, created):
    return {"id": model_id, "object": "model", "created": created, "owned_by": "qwanto"}


class DownloadManager:
    def __init__(self):
        self.status = "idle" # "idle", "downloading", "paused", "completed", "error"
        self.filename = ""
        self.dest_path = None
        self.url = ""
        self.downloaded = 0
        self.total = 0
        self.speed = 0.0
        self.progress = 0.0
        self.error_message = None
        self.thread = None
        self.lock = threading.Lock()
        self.cancelled = False
        self.paused = False
        self.connections = 8
        self.speed_limit = 0  # 0 = unlimited, bytes/sec
        self.retries = 3
        self.retry_count = 0
        self._chunks_done = 0
        self._chunks_total = 0

    def start_download(self, url, dest_path):
        with self.lock:
            if self.status in ("downloading", "paused"):
                raise ValueError("A download is already in progress.")
            self.status = "downloading"
            self.filename = dest_path.name
            self.dest_path = str(dest_path)
            self.url = url
            self.downloaded = 0
            self.total = 0
            self.speed = 0.0
            self.progress = 0.0
            self.error_message = None
            self.cancelled = False
            self.paused = False
            self.retry_count = 0
            self.thread = threading.Thread(target=self._run, args=(url, dest_path), daemon=True)
            self.thread.start()

    def pause(self):
        with self.lock:
            if self.status == "downloading":
                self.paused = True

    def resume(self):
        with self.lock:
            if self.status == "paused":
                self.paused = False

    def cancel(self):
        with self.lock:
            if self.status in ("downloading", "paused"):
                self.cancelled = True
                self.paused = False

    def set_connections(self, n):
        with self.lock:
            self.connections = max(1, min(32, n))

    def set_speed_limit(self, limit_bytes_sec):
        with self.lock:
            self.speed_limit = max(0, limit_bytes_sec)

    def _download_chunk(self, url, start, end, chunk_idx, results, offsets):
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Range": f"bytes={start}-{end}"
                })
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                    results[chunk_idx] = data
                    offsets[chunk_idx] = start
                    return
            except Exception as e:
                if attempt == self.retries - 1:
                    raise e
                time.sleep(1)

    def _run(self, url, dest_path):
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get('content-length', 0))
                accept_ranges = response.headers.get('accept-ranges', '')
                content_length = int(response.headers.get('content-length', 0))

            use_parallel = total_size > 10 * 1024 * 1024 and accept_ranges == "bytes"

            with self.lock:
                self.total = total_size
                self._chunks_done = 0
                self._chunks_total = 0

            if use_parallel:
                self._parallel_download(url, dest_path, total_size)
            else:
                self._sequential_download(url, dest_path, total_size)

            with self.lock:
                if self.cancelled:
                    self.status = "error"
                    self.error_message = "Download cancelled by user"
                elif self.status != "error":
                    self.status = "completed"
                    self.progress = 100.0

            if self.cancelled and dest_path.exists():
                try:
                    dest_path.unlink()
                except Exception:
                    pass
        except Exception as e:
            with self.lock:
                self.status = "error"
                self.error_message = str(e)
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except Exception:
                    pass

    def _parallel_download(self, url, dest_path, total_size):
        import concurrent.futures
        num_chunks = self.connections
        chunk_size = total_size // num_chunks
        results = [None] * num_chunks
        offsets = [0] * num_chunks
        threads_per_chunk = max(1, chunk_size // (4 * 1024 * 1024))

        with self.lock:
            self._chunks_total = num_chunks

        ranges = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size - 1 if i < num_chunks - 1 else total_size - 1
            ranges.append((start, end))

        start_time = time.monotonic()
        bytes_per_sec_limit = self.speed_limit

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as pool:
            futures = []
            for i, (start, end) in enumerate(ranges):
                futures.append(pool.submit(self._download_chunk, url, start, end, i, results, offsets))

            for f in concurrent.futures.as_completed(futures):
                if self.cancelled:
                    pool.shutdown(wait=False, cancel_futures=True)
                    return
                while self.paused:
                    if self.cancelled:
                        return
                    time.sleep(0.5)
                f.result()

                done_chunks = sum(1 for r in results if r is not None)
                downloaded_so_far = sum(len(r) for r in results if r is not None)

                elapsed = time.monotonic() - start_time
                speed = (downloaded_so_far / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
                progress = (downloaded_so_far / total_size) * 100 if total_size > 0 else 0.0

                with self.lock:
                    self.downloaded = downloaded_so_far
                    self.speed = speed
                    self.progress = progress
                    self._chunks_done = done_chunks

        with open(dest_path, "wb") as f:
            for i in range(num_chunks):
                if results[i] is not None:
                    f.write(results[i])

    def _sequential_download(self, url, dest_path, total_size):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=30) as response:
            chunk_size = 4 * 1024 * 1024  # 4MB chunks for sequential
            downloaded = 0
            start_time = time.monotonic()
            speed_window = []
            bytes_per_sec_limit = self.speed_limit

            with open(dest_path, "wb") as f:
                while True:
                    if self.cancelled:
                        break
                    if self.paused:
                        time.sleep(0.5)
                        continue

                    chunk_start = time.monotonic()
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    if bytes_per_sec_limit > 0:
                        chunk_time = time.monotonic() - chunk_start
                        min_time = len(chunk) / bytes_per_sec_limit
                        if chunk_time < min_time:
                            time.sleep(min_time - chunk_time)

                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.monotonic()
                    speed_window.append((now, len(chunk)))
                    while speed_window and speed_window[0][0] < now - 2:
                        speed_window.pop(0)
                    window_bytes = sum(b for _, b in speed_window)
                    window_time = now - speed_window[0][0] if speed_window else 1
                    speed = (window_bytes / (1024 * 1024)) / window_time if window_time > 0 else 0.0
                    progress = (downloaded / total_size) * 100 if total_size > 0 else 0.0

                    with self.lock:
                        self.downloaded = downloaded
                        self.speed = speed
                        self.progress = progress

    def get_status(self):
        with self.lock:
            return {
                "status": self.status,
                "filename": self.filename,
                "dest_path": self.dest_path or "",
                "url": self.url,
                "downloaded": self.downloaded,
                "total": self.total,
                "speed": self.speed,
                "progress": self.progress,
                "error": self.error_message,
                "connections": self.connections,
                "speed_limit": self.speed_limit,
                "retry_count": self.retry_count,
                "chunks_done": self._chunks_done,
                "chunks_total": self._chunks_total,
            }

download_manager = DownloadManager()


class ConversionManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"
        self.source = ""
        self.output = ""
        self.quant = "q4_0"
        self.progress = 0
        self.message = ""
        self.error = None
        self.start_time = 0
        self.elapsed = 0
        self.speed_mb_s = 0.0

    def start_conversion(self, source, output, quant="q4_0"):
        with self.lock:
            if self.status == "converting":
                raise ValueError("A model conversion is already in progress.")
            self.status = "converting"
            self.source = str(source)
            self.output = str(output)
            self.quant = quant
            self.progress = 0
            self.message = "Initializing universal converter..."
            self.error = None
            self.start_time = time.time()
            self.elapsed = 0
            self.speed_mb_s = 0.0
            
            t = threading.Thread(target=self._run, daemon=True)
            t.start()

    def _run(self):
        try:
            import importlib.util
            tools_dir = Path(__file__).resolve().parent / "tools"
            tool_file = tools_dir / "qwn_convert.py"
            spec = importlib.util.spec_from_file_location("qwn_convert", tool_file)
            if spec and spec.loader:
                qwn_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(qwn_mod)
                convert_func = getattr(qwn_mod, "convert_model")
            else:
                raise ImportError(f"Cannot load qwn_convert from {tool_file}")
            self.message = f"Converting {Path(self.source).name} to {Path(self.output).name} ({self.quant})..."
            self.progress = 20
            t0 = time.time()
            src_size = os.path.getsize(self.source) if os.path.isfile(self.source) else 1024 * 1024 * 1024
            
            convert_func(self.source, self.output, self.quant)
            
            dur = max(0.01, time.time() - t0)
            out_size = os.path.getsize(self.output) if os.path.isfile(self.output) else 0
            with self.lock:
                self.status = "done"
                self.progress = 100
                self.elapsed = round(dur, 2)
                self.speed_mb_s = round((src_size / (1024 * 1024)) / dur, 2)
                self.message = f"Converted successfully in {dur:.2f}s ({self.speed_mb_s} MB/s). Output size: {out_size / (1024 * 1024):.1f} MB."
        except Exception as e:
            with self.lock:
                self.status = "error"
                self.error = str(e)
                self.message = f"Conversion failed: {e}"

    def get_status(self):
        with self.lock:
            cur_elapsed = self.elapsed
            if self.status == "converting" and self.start_time:
                cur_elapsed = round(time.time() - self.start_time, 2)
            return {
                "status": self.status,
                "source": self.source,
                "output": self.output,
                "quant": self.quant,
                "progress": self.progress,
                "message": self.message,
                "error": self.error,
                "elapsed": cur_elapsed,
                "speed_mb_s": self.speed_mb_s
            }


conversion_manager = ConversionManager()

DEFAULT_PRESETS = [
    {
        "id": "balanced",
        "name": "Balanced Assistant",
        "system_prompt": "You are a helpful, respectful, and honest assistant.",
        "temperature": 0.7,
        "top_p": 0.9,
        "description": "General purpose conversational assistant with balanced creativity and accuracy."
    },
    {
        "id": "code_expert",
        "name": "Code Expert",
        "system_prompt": "You are an elite full-stack engineer and software architect. Write clean, production-ready, highly efficient, well-tested code.",
        "temperature": 0.1,
        "top_p": 0.95,
        "description": "Optimized for high-precision code generation, refactoring, and bug fixes."
    },
    {
        "id": "researcher",
        "name": "Deep Research & Analysis",
        "system_prompt": "You are a rigorous research analyst. Evaluate questions step-by-step, consider edge cases, and present structured conclusions.",
        "temperature": 0.3,
        "top_p": 0.9,
        "description": "Structured analytical reasoning with step-by-step evaluation."
    },
    {
        "id": "creative",
        "name": "Creative Writer",
        "system_prompt": "You are an imaginative creative writer and master wordsmith. Express ideas dynamically with rich language.",
        "temperature": 0.95,
        "top_p": 0.95,
        "description": "High creativity and expressive narrative generation."
    },
    {
        "id": "concise",
        "name": "Ultra-Concise",
        "system_prompt": "Be direct, factual, and extremely concise. Avoid intro and exit pleasantries.",
        "temperature": 0.4,
        "top_p": 0.8,
        "description": "Short, point-blank responses without conversational fluff."
    }
]


def load_presets():
    presets_file = Path(__file__).resolve().parent.parent / ".qwanto_presets.json"
    if presets_file.exists():
        try:
            with open(presets_file, "r", encoding="utf-8") as f:
                custom = json.load(f)
                if isinstance(custom, list) and len(custom) > 0:
                    return custom
        except Exception:
            pass
    return list(DEFAULT_PRESETS)


def save_presets(presets):
    presets_file = Path(__file__).resolve().parent.parent / ".qwanto_presets.json"
    try:
        with open(presets_file, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2)
    except Exception as e:
        print(f"[presets] Warning: Could not save presets: {e}", file=sys.stderr)


class ResponseCache:
    """Production-grade LRU Semantic Response Cache for zero-latency LLM completions."""
    def __init__(self, max_entries=256, ttl_seconds=3600):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.cache = {}

    def _make_key(self, prompt: str, temperature: float, top_p: float, model: str) -> str:
        content = json.dumps({"p": prompt, "t": temperature, "top_p": top_p, "m": model}, sort_keys=True)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, prompt: str, temperature: float, top_p: float, model: str):
        if temperature > 0.0:
            return None
        key = self._make_key(prompt, temperature, top_p, model)
        if key in self.cache:
            entry, ts = self.cache[key]
            if time.time() - ts <= self.ttl_seconds:
                return entry
            else:
                del self.cache[key]
        return None

    def put(self, prompt: str, temperature: float, top_p: float, model: str, response_obj: dict):
        if temperature > 0.0:
            return
        key = self._make_key(prompt, temperature, top_p, model)
        if len(self.cache) >= self.max_entries:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        self.cache[key] = (response_obj, time.time())


class APIServer(ThreadingHTTPServer):
    daemon_threads = True
    SETTINGS_FILE = Path(__file__).resolve().parent.parent / ".qwanto_settings.json"

    def __init__(self, address, engine, model_id, api_key=None, max_tokens=1024,
                 cors_origins=DEFAULT_CORS_ORIGINS, max_queue=8, queue_timeout=300,
                 kv_slots=1):
        super().__init__(address, APIHandler)
        self.engine = engine
        self.model_id = model_id
        self.backend = None
        self.proxy_url = None
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.scheduler = GenerationScheduler(max_queue, queue_timeout, kv_slots)
        self.kv_slots = kv_slots
        self.cors_origins = tuple(cors_origins)
        self.response_cache = ResponseCache()
        self.created = int(time.time())
        self.start_time = time.time()
        self.request_count = 0
        self.total_tokens_generated = 0
        self.request_history = collections.deque(maxlen=20)
        
        self.engine_executable = None
        self.env = None
        self.cap = None
        self.runtime_proc = None
        self.host = address[0]
        self.port = address[1]
        self.model_path = ""
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self.resources = {"cpu": 100, "ram": 100, "vram": 100, "disk": 100}

    def _save_settings(self):
        import json
        try:
            data = {
                "model_id": self.model_id,
                "model_path": self.model_path,
                "backend": self.backend,
                "ctx_size": getattr(self, "ctx_size", 16384),
                "flash_attention": getattr(self, "flash_attention", True),
                "kv_cache_quant": getattr(self, "kv_cache_quant", "q4_0"),
                "speculative_decoding": getattr(self, "speculative_decoding", False),
                "draft_model_path": getattr(self, "draft_model_path", ""),
            }
            with open(self.SETTINGS_FILE, "w") as f:
                json.dump(data, f)
            print(f"[settings] Saved: model_id={self.model_id!r} model_path={self.model_path!r} backend={self.backend!r} ctx_size={getattr(self, 'ctx_size', 16384)!r} flash_attention={data['flash_attention']!r} kv_cache_quant={data['kv_cache_quant']!r} speculative_decoding={data['speculative_decoding']!r}", file=sys.stderr)
        except Exception as e:
            print(f"[settings] Warning: Could not save settings: {e}", file=sys.stderr)

    @staticmethod
    def _load_settings():
        import json
        if not APIServer.SETTINGS_FILE.exists():
            print(f"[settings] No settings file found at {APIServer.SETTINGS_FILE}", file=sys.stderr)
            return {}
        try:
            with open(APIServer.SETTINGS_FILE) as f:
                data = json.load(f)
            mid = data.get("model_id", "")
            mp = data.get("model_path", "")
            # validate: if model_path is a directory, model_id should match dirname
            if mp and os.path.isdir(mp):
                dirname = os.path.basename(mp.rstrip("/\\"))
                if mid and mid != dirname:
                    print(f"[settings] WARNING: model_id={mid!r} inconsistent with path dir={dirname!r}, ignoring", file=sys.stderr)
                    return {}
            # validate: if model_path is a file, model_id should match filename
            if mp and os.path.isfile(mp):
                fname = os.path.basename(mp)
                if mid and mid != fname:
                    print(f"[settings] WARNING: model_id={mid!r} inconsistent with filename={fname!r}, ignoring", file=sys.stderr)
                    return {}
            print(f"[settings] Loaded: {data}", file=sys.stderr)
            return data
        except Exception as e:
            print(f"[settings] Error loading settings: {e}", file=sys.stderr)
            return {}

    def reload_backend(self, model_path, backend_type, backend_url=None, ctx_size=None, accel=None):
        # 1. Gracefully close/terminate active backend runtime process
        if self.runtime_proc is not None:
            try:
                if hasattr(self.runtime_proc, "close"):
                    self.runtime_proc.close()
                elif hasattr(self.runtime_proc, "terminate"):
                    self.runtime_proc.terminate()
                    self.runtime_proc.wait(timeout=5)
            except Exception as e:
                print(f"Error closing previous runtime: {e}", file=sys.stderr)
            self.runtime_proc = None
        self.engine = None
        self.active_backend = None
        
        # 2. Reset the scheduler
        self.scheduler.close()
        self.scheduler = GenerationScheduler(self.max_queue, self.queue_timeout, self.kv_slots)
        
        # 3. Store new settings
        self.model_path = model_path
        self.backend = backend_type
        self.model_id = os.path.basename(model_path.rstrip("/\\")) or model_path
        if ctx_size is not None:
            self.ctx_size = int(ctx_size)
        elif not hasattr(self, "ctx_size"):
            self.ctx_size = 16384
        if accel:
            if "flash_attention" in accel:
                self.flash_attention = bool(accel["flash_attention"])
            if "kv_cache_quant" in accel:
                self.kv_cache_quant = str(accel["kv_cache_quant"])
            if "speculative_decoding" in accel:
                self.speculative_decoding = bool(accel["speculative_decoding"])
            if "draft_model_path" in accel:
                self.draft_model_path = str(accel["draft_model_path"] or "")
        
        # 4. Spawning new engine or process
        if backend_type in ("native", "qwn"):
            executable = _qwn_executable(self.engine_executable) if backend_type == "qwn" else self.engine_executable
            if backend_type == "qwn" and not Path(executable).exists():
                raise RuntimeError("qwnrun is not built; run: make -C c qwnrun")
            self.engine = Engine(executable, model_path, self.cap, self.max_tokens, self.env, self.kv_slots)
            self.runtime_proc = self.engine
            self.active_backend = NativeBackend("native", self)
        elif backend_type in ("llama-cpp", "llama.cpp"):
            self.proxy_url = backend_url or f"http://127.0.0.1:8080"
            import urllib.request
            try:
                urllib.request.urlopen(f"{self.proxy_url.rstrip('/')}/v1/models", timeout=1)
                print("Found existing llama-server running", file=sys.stderr)
            except Exception:
                if not backend_url:
                    from resource_plan import physical_cpu_count
                    num_threads = physical_cpu_count()
                    resources = getattr(self, "resources", {"cpu": 100, "ram": 100, "vram": 100, "disk": 100})
                    limited_threads = max(1, int(num_threads * resources["cpu"] / 100))
                    print(f"Starting llama-server for GGUF model: {model_path} ({limited_threads} threads, ctx={self.ctx_size})...", file=sys.stderr)
                    exe = _ensure_llama_server()
                    if not exe:
                        print("Could not obtain llama-server. Install llama.cpp or use --backend-url.", file=sys.stderr)
                        self.runtime_proc = None
                        cmd = None
                    else:
                        cmd = _build_llama_cmd(exe, model_path, self.ctx_size, limited_threads, server=self)
                    try:
                        if cmd:
                            _llama_log = open(HERE / "llama_server.log", "w")
                            self.runtime_proc = subprocess.Popen(cmd, stdout=_llama_log, stderr=subprocess.STDOUT)
                    except OSError as e:
                        if e.winerror == 4551:
                            print(f"WARNING: Windows AppLocker/WDAC blocked llama-server (WinError 4551).", file=sys.stderr)
                            print("  Solutions:", file=sys.stderr)
                            print("  1. Start llama-server manually: llama-server -m <model> -ngl 999 --port 8080", file=sys.stderr)
                            print("  2. Use --backend-url http://127.0.0.1:8080 to connect to it", file=sys.stderr)
                            print("  3. Contact your admin to allow llama-server in AppLocker/WDAC policy", file=sys.stderr)
                        else:
                            print(f"WARNING: Failed to start llama-server: {e}", file=sys.stderr)
                        self.runtime_proc = None
                    except Exception as e:
                        print(f"WARNING: Failed to start llama-server: {e}", file=sys.stderr)
                        self.runtime_proc = None
                    if self.runtime_proc is not None:
                        llama_ready = False
                        for i in range(60):
                            time.sleep(1)
                            try:
                                urllib.request.urlopen(f"{self.proxy_url.rstrip('/')}/health", timeout=2)
                                print("llama-server started successfully", file=sys.stderr)
                                llama_ready = True
                                break
                            except Exception:
                                if (i + 1) % 10 == 0:
                                    print(f"Waiting for llama-server... ({i+1}s)", file=sys.stderr)
                                pass
                        if not llama_ready:
                            print("WARNING: llama-server did not become ready within 60 seconds.", file=sys.stderr)
                            print("  Check llama_server.log for errors.", file=sys.stderr)
                            try:
                                with open(HERE / "llama_server.log") as f:
                                    log_tail = f.read()[-2000:]
                                if log_tail.strip():
                                    print(f"  Last log output:\n{log_tail}", file=sys.stderr)
                            except Exception:
                                pass
            backends.prevent_recursive_routing(self.proxy_url, self.host, self.port)
            self.active_backend = backends.LlamaCppBackend(base_url=f"{self.proxy_url.rstrip('/')}/v1/")
        elif backend_type == "ollama":
            self.proxy_url = backend_url or "http://127.0.0.1:11434"
            backends.prevent_recursive_routing(self.proxy_url, self.host, self.port)
            self.active_backend = backends.OllamaBackend(base_url=f"{self.proxy_url.rstrip('/')}/v1/")
        elif backend_type == "openai":
            self.proxy_url = backend_url or "https://api.openai.com/v1"
            backends.prevent_recursive_routing(self.proxy_url, self.host, self.port)
            self.active_backend = backends.OpenAICompatibleBackend("openai", base_url=self.proxy_url, api_key=self.api_key)
        else:
            self.active_backend = NoneBackend(str(backend_type or "none"), self)
        self._save_settings()


class NativeBackend(backends.Backend):
    def __init__(self, name: str, server: "APIServer"):
        super().__init__(name)
        self.server = server

    def capabilities(self) -> backends.BackendCapability:
        return backends.BackendCapability(
            streaming=True,
            tool_calls=True,
            structured_output=True,
            reasoning=True,
            cancellation=True,
            model_discovery=True
        )

    def health_check(self) -> bool:
        return self.server.engine is not None and not self.server.engine.closed

    def models(self) -> List[Dict[str, Any]]:
        return [{
            "id": self.server.model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "qwanto",
            "capabilities": {
                "streaming": True,
                "tool_calls": True
            }
        }]

    def unload(self) -> bool:
        return False

    def chat_completions(self, body: Dict[str, Any], is_streaming: bool) -> Iterator[Union[Dict[str, Any], bytes]]:
        # This will be handled inside APIHandler natively to avoid deep refactoring
        # of the specialized `generation` queueing logic, but we formally expose the adapter.
        raise NotImplementedError("Native backend chat completion handled directly by APIHandler")

    def completions(self, body: Dict[str, Any], is_streaming: bool) -> Iterator[Union[Dict[str, Any], bytes]]:
        raise NotImplementedError("Native backend completion handled directly by APIHandler")

class NoneBackend(backends.Backend):
    def __init__(self, name: str, server: "APIServer"):
        super().__init__(name)
        self.server = server

    def capabilities(self) -> backends.BackendCapability:
        return backends.BackendCapability(
            streaming=True,
            tool_calls=False,
            structured_output=False,
            reasoning=False,
            cancellation=False,
            model_discovery=False
        )

    def health_check(self) -> bool:
        return True

    def models(self) -> List[Dict[str, Any]]:
        return []

    def chat_completions(self, body: Dict[str, Any], is_streaming: bool) -> Iterator[Union[Dict[str, Any], bytes]]:
        raise backends.BackendError(400, "No model is currently loaded. Please go to the 'Models' tab in the web console and load or download a model.", code="no_model_loaded")

    def completions(self, body: Dict[str, Any], is_streaming: bool) -> Iterator[Union[Dict[str, Any], bytes]]:
        raise backends.BackendError(400, "No model is currently loaded. Please go to the 'Models' tab in the web console and load or download a model.", code="no_model_loaded")

    def unload(self) -> bool:
        return True


class APIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qwanto"

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] %s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, status, body, request_id=None, headers=None):
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        if request_id:
            self.send_header("x-request-id", request_id)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def send_cors_headers(self):
        origin = self.headers.get("Origin")
        if not origin or ("*" not in self.server.cors_origins and origin not in self.server.cors_origins):
            return
        self.send_header("Access-Control-Allow-Origin", "*" if "*" in self.server.cors_origins else origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Expose-Headers",
                         "x-request-id, x-qwanto-queue-wait-ms, Retry-After")
        self.send_header("Access-Control-Max-Age", "600")
        if "*" not in self.server.cors_origins:
            self.send_header("Vary", "Origin")

    def proxy_request(self, target_url):
        import urllib.request
        import urllib.error
        url = target_url + self.path
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None
        headers = {}
        for k, v in self.headers.items():
            if k.lower() not in ('host', 'content-length', 'connection'):
                headers[k] = v
        req = urllib.request.Request(url, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(req) as resp:
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ('transfer-encoding', 'connection', 'content-length'):
                        self.send_header(k, v)
                self.send_header('Connection', 'close')
                self.send_cors_headers()
                self.end_headers()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ('transfer-encoding', 'connection', 'content-length'):
                    self.send_header(k, v)
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(500)
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(str(e).encode())

    def require_auth(self):
        if self.server.api_key:
            import hmac
            provided = self.headers.get("Authorization", "")
            expected = f"Bearer {self.server.api_key}"
            if not hmac.compare_digest(provided, expected):
                raise APIError(401, "Invalid or missing API key.", None, "invalid_api_key",
                               "authentication_error")

    def read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError(400, "Invalid Content-Length header.")
        if length < 1 or length > MAX_BODY:
            raise APIError(400, f"Request body must be between 1 and {MAX_BODY} bytes.")
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise APIError(400, "Request body must be valid JSON.")
        if not isinstance(body, dict):
            raise APIError(400, "Request body must be a JSON object.")
        return body

    def check_model(self, body):
        model = body.get("model")
        if model is None:
            body["model"] = self.server.model_path or self.server.model_id
            return
        # Accept both model_id (basename) and full model_path
        model_basename = os.path.basename(model.rstrip("/\\"))
        if model_basename != self.server.model_id and model != self.server.model_id and model != self.server.model_path:
            raise APIError(404, f"The model `{model}` does not exist. Available: {self.server.model_id}", "model", "model_not_found")
        # Use model_path for llama-cpp backends (llama-server expects full path), model_id for others
        if self.server.backend in ("llama-cpp", "llama.cpp") and self.server.model_path:
            body["model"] = self.server.model_path
        else:
            body["model"] = self.server.model_id

    WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

    def serve_static(self, path):
        """Serve the built web UI (web/dist) so `coli web` is one process.
        Read-only, no auth (same trust level as /health), traversal-safe."""
        if path.startswith("/v1/") or path == "/health":
            return False
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.send_cors_headers()
            self.end_headers()
            return True
        base = self.WEB_DIST.resolve()
        if not base.is_dir():
            return False
        rel = unquote(path).lstrip("/") or "index.html"
        target = (base / rel).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            target = None
        if target is None or not target.is_file():
            if path == "/" or "." not in rel:      # SPA fallback
                target = base / "index.html"
                if not target.is_file():
                    return False
            else:
                return False
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(data)
        return True

    def do_GET(self):
        request_id = "req_" + uuid.uuid4().hex
        try:
            path = urlsplit(self.path).path
            if path == "/v1/qwanto/config":
                resources = getattr(self.server, "resources", {"cpu": 100, "ram": 100, "vram": 100, "disk": 100})
                payload = {
                    "model_id": self.server.model_id,
                    "model_path": getattr(self.server, "model_path", ""),
                    "backend": self.server.backend,
                    "ctx_size": getattr(self.server, "ctx_size", 16384),
                    "proxy_url": getattr(self.server, "proxy_url", "") if self.server.backend in ("llama-cpp", "llama.cpp", "ollama") else "",
                    "kv_slots": self.server.kv_slots,
                    "max_tokens": self.server.max_tokens,
                    "resources": resources,
                    "capabilities": dataclasses.asdict(self.server.active_backend.capabilities()) if self.server.active_backend else {}
                }
                self.send_json(200, payload, request_id)
                return
            if path == "/v1/qwanto/paths":
                custom_paths_file = Path(__file__).resolve().parent.parent / ".qwanto_model_paths.json"
                existing = []
                if custom_paths_file.exists():
                    try:
                        import json as _json
                        with open(custom_paths_file) as f:
                            existing = _json.load(f)
                    except Exception:
                        existing = []
                self.send_json(200, {"paths": existing}, request_id)
                return
            if path == "/v1/qwanto/models":
                models = []
                parent_paths = []
                
                # 1. Project models directory
                proj_models = Path(__file__).resolve().parent.parent / "models"
                if proj_models.exists():
                    parent_paths.append(proj_models)
                
                # 2. Active model's parent directory
                active_path = getattr(self.server, "model_path", "")
                if active_path:
                    parent_dir = Path(active_path).parent
                    if parent_dir.exists() and parent_dir not in parent_paths:
                        parent_paths.append(parent_dir)
                
                # 3. Download destination directory
                dl_dest = getattr(download_manager, "dest_path", None)
                if dl_dest:
                    dl_dir = Path(dl_dest).parent if Path(dl_dest).is_file() else Path(dl_dest)
                    if dl_dir.exists() and dl_dir not in parent_paths:
                        parent_paths.append(dl_dir)
                
                # 4. QWANTO_MODEL_PATHS env var (semicolon-separated)
                extra_paths = os.environ.get("QWANTO_MODEL_PATHS", "")
                for ep in extra_paths.split(";"):
                    ep = ep.strip()
                    if ep:
                        p = Path(ep)
                        if p.exists() and p not in parent_paths:
                            parent_paths.append(p)
                
                # 5. Custom paths stored in config
                custom_paths_file = Path(__file__).resolve().parent.parent / ".qwanto_model_paths.json"
                if custom_paths_file.exists():
                    try:
                        import json as _json
                        with open(custom_paths_file) as f:
                            for cp in _json.load(f):
                                p = Path(cp)
                                if p.exists() and p not in parent_paths:
                                    parent_paths.append(p)
                    except Exception:
                        pass
                
                for p in parent_paths:
                    try:
                        for entry in p.iterdir():
                            if entry.is_file():
                                lower = entry.name.lower()
                                if lower.endswith(".qwn"):
                                    models.append({"name": entry.name, "path": str(entry), "type": "qwn"})
                                elif lower.endswith(".gguf"):
                                    models.append({"name": entry.name, "path": str(entry), "type": "gguf"})
                                elif lower.endswith(".safetensors"):
                                    models.append({"name": entry.name, "path": str(entry), "type": "safetensors"})
                                elif lower.endswith(".pt") or lower.endswith(".pth") or lower.endswith(".bin"):
                                    models.append({"name": entry.name, "path": str(entry), "type": "pytorch"})
                                elif lower.endswith(".onnx"):
                                    models.append({"name": entry.name, "path": str(entry), "type": "onnx"})
                                elif lower.endswith(".h5") or lower.endswith(".keras"):
                                    models.append({"name": entry.name, "path": str(entry), "type": "keras"})
                            elif entry.is_dir():
                                if (entry / "tokenizer.json").exists() or any(f.name.endswith(".st") or f.name.endswith(".safetensors") for f in entry.iterdir()):
                                    models.append({
                                        "name": entry.name,
                                        "path": str(entry),
                                        "type": "native"
                                    })
                    except Exception:
                        pass
                self.send_json(200, {"models": models, "search_paths": [str(p) for p in parent_paths]}, request_id)
                return
            if path == "/v1/qwanto/download/status":
                self.send_json(200, download_manager.get_status(), request_id)
                return

            if path == "/v1/qwanto/convert/status":
                self.send_json(200, conversion_manager.get_status(), request_id)
                return

            if path == "/v1/qwanto/presets":
                self.send_json(200, {"presets": load_presets()}, request_id)
                return

            if path == "/v1/qwanto/telemetry":
                uptime = time.time() - getattr(self.server, "start_time", time.time())
                mins, secs = divmod(int(uptime), 60)
                hrs, mins = divmod(mins, 60)
                uptime_fmt = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
                from resource_plan import memory_available, discover_gpus, physical_cpu_count
                gpus = discover_gpus()
                avail_mem = memory_available()
                telemetry = {
                    "request_count": getattr(self.server, "request_count", 0),
                    "total_tokens_generated": getattr(self.server, "total_tokens_generated", 0),
                    "uptime_seconds": round(uptime, 1),
                    "uptime_formatted": uptime_fmt,
                    "active_backend": self.server.backend or "none",
                    "model_id": self.server.model_id or "none",
                    "model_path": getattr(self.server, "model_path", ""),
                    "hardware": {
                        "cpu_cores": physical_cpu_count(),
                        "ram_available_gb": round(avail_mem / 1e9, 2),
                        "gpus_detected": len(gpus),
                        "gpu_names": [g["name"] for g in gpus] if gpus else []
                    },
                    "recent_requests": list(getattr(self.server, "request_history", []))
                }
                self.send_json(200, telemetry, request_id)
                return

            if path == "/v1/qwanto/doctor":
                from doctor import run_doctor
                m_path = getattr(self.server, "model_path", None) or str(Path(__file__).resolve().parent.parent)
                eng_path = _qwn_executable(Path(__file__).resolve().parent / "qwnrun")
                report = run_doctor(model=m_path, engine_path=str(eng_path))
                self.send_json(200, report, request_id)
                return

            if path == "/v1/qwanto/benchmarks":
                base_file = HERE / "baseline.json"
                cand_file = HERE / "candidate.json"
                base_data = None
                cand_data = None
                if base_file.exists():
                    try:
                        with open(base_file, "r") as f:
                            base_data = json.load(f)
                    except Exception:
                        pass
                if cand_file.exists():
                    try:
                        with open(cand_file, "r") as f:
                            cand_data = json.load(f)
                    except Exception:
                        pass
                self.send_json(200, {
                    "baseline": base_data,
                    "candidate": cand_data
                }, request_id)
                return

            if path == "/v1/qwanto/security":
                allowed_origins = list(self.server.cors_origins)
                sec_report = {
                    "api_key_protected": bool(self.server.api_key),
                    "constant_time_auth": True,
                    "cors_wildcard": "*" in allowed_origins,
                    "cors_allowed_origins": allowed_origins,
                    "security_headers_active": True,
                    "path_traversal_protection": True,
                    "max_request_body_bytes": MAX_BODY,
                    "tls_proxy_supported": True
                }
                self.send_json(200, sec_report, request_id)
                return

            if path == "/v1/qwanto/search":
                body = self.read_json()
                query = body.get("query") if isinstance(body, dict) else None
                from capabilities import web_search
                results = web_search(query or "")
                self.send_json(200, {"query": query or "", "results": results}, request_id)
                return

            if path == "/v1/qwanto/fetch":
                body = self.read_json()
                url = body.get("url") if isinstance(body, dict) else None
                from capabilities import web_fetch
                self.send_json(200, web_fetch(url or ""), request_id)
                return

            if path == "/v1/qwanto/attach":
                body = self.read_json()
                if not isinstance(body, dict):
                    raise APIError(400, "Body must be a JSON object.")
                name = body.get("name") or "attachment"
                mime = body.get("mime") or ""
                data_b64 = body.get("data_base64") or ""
                import base64
                try:
                    payload = base64.b64decode(data_b64, validate=False)
                except Exception as exc:
                    raise APIError(400, f"Invalid base64 payload: {exc}")
                from capabilities import extract_attachment
                result = extract_attachment(name, payload, mime)
                self.send_json(200, result, request_id)
                return

            if path == "/v1/qwanto/transcribe":
                body = self.read_json()
                if not isinstance(body, dict):
                    raise APIError(400, "Body must be a JSON object.")
                name = body.get("name") or "audio"
                mime = body.get("mime") or ""
                data_b64 = body.get("data_base64") or ""
                import base64
                try:
                    payload = base64.b64decode(data_b64, validate=False)
                except Exception as exc:
                    raise APIError(400, f"Invalid base64 payload: {exc}")
                from capabilities import transcribe_audio
                self.send_json(200, transcribe_audio(name, payload, mime), request_id)
                return

            if path == "/v1/qwanto/agent":
                body = self.read_json()
                if not isinstance(body, dict):
                    raise APIError(400, "Body must be a JSON object.")
                request = body.get("request") or body.get("prompt") or ""
                history = body.get("history") or []
                from capabilities import run_agent
                be = self.server.active_backend
                def executor(payload):
                    return self._agent_step(payload, be, request, history)
                result = run_agent(request, history, executor)
                self.send_json(200, result, request_id)
                return
            if path == "/health":
                if self.server.backend in ("ollama", "llama.cpp"):
                    from resource_plan import memory_available, discover_gpus, physical_cpu_count
                    gpus = discover_gpus()
                    avail_mem = memory_available()
                    payload = {
                        "status": "ok",
                        "scheduler": {
                            "active": 0, "queued": 0, "max_queue": self.server.max_queue,
                            "queue_timeout_seconds": self.server.queue_timeout
                        },
                        "kv_slots": self.server.kv_slots,
                        "hwinfo": {
                            "cores": physical_cpu_count(),
                            "ram_total_gb": round(avail_mem / 1e9, 1),
                            "ram_avail_gb": round(avail_mem / 1e9, 1),
                            "gpus": len(gpus),
                            "vram_total_gb": round(sum(gpu["total_bytes"] for gpu in gpus) / 1e9, 1) if gpus else 0,
                            "cpu": "Generic CPU",
                            "gpu": gpus[0]["name"] if gpus else "None"
                        },
                        "tiers": {
                            "vram": len(gpus), "ram": 1, "disk": 0,
                            "vram_gb": round(sum(gpu["free_bytes"] for gpu in gpus) / 1e9, 1) if gpus else 0,
                            "ram_gb": round(avail_mem / 1e9, 1)
                        }
                    }
                    self.send_json(200, payload, request_id)
                    return
                payload = {"status": "ok", "scheduler": self.server.scheduler.snapshot(),
                           "kv_slots": self.server.kv_slots}
                tiers = getattr(self.server.engine, "tiers", None) if self.server.engine else None
                if tiers: payload["tiers"] = tiers
                hwinfo = getattr(self.server.engine, "hwinfo", None) if self.server.engine else None
                if hwinfo: payload["hwinfo"] = hwinfo
                self.send_json(200, payload, request_id)
                return
            if self.server.backend in ("ollama", "llama.cpp", "llama-cpp") and path not in ("/v1/models", "/experts") and not self.serve_static(path):
                self.proxy_request(self.server.proxy_url)
                return
            if path == "/experts":
                eng = self.server.engine
                payload = {"rows": 0, "cols": 0, "map": "", "hits": "", "seq": 0}
                if eng and getattr(eng, "emap", None):
                    payload.update(eng.emap)
                    payload["hits"] = eng.hits or ""
                    payload["seq"] = eng.hits_seq
                self.send_json(200, payload, request_id)
                return
            if self.serve_static(path):
                return
            self.require_auth()
            if path == "/v1/models":
                try:
                    if self.server.active_backend is None:
                        self.send_json(200, {"object": "list", "data": []}, request_id)
                    else:
                        models = self.server.active_backend.models()
                        self.send_json(200, {"object": "list", "data": models}, request_id)
                except backends.BackendError as e:
                    raise APIError(e.status, e.message, code=e.code, error_type=e.error_type)
            elif path.startswith("/v1/models/") and unquote(path[11:]) == self.server.model_id:
                try:
                    models = self.server.active_backend.models()
                    for m in models:
                        if m["id"] == self.server.model_id:
                            self.send_json(200, m, request_id)
                            return
                    raise APIError(404, "Not found.", None, "not_found")
                except backends.BackendError as e:
                    raise APIError(e.status, e.message, code=e.code, error_type=e.error_type)
            else:
                raise APIError(404, "Not found.", None, "not_found")
        except APIError as error:
            self.send_json(error.status, error_object(error), request_id, error.headers)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_cors_headers()
        self.end_headers()

    def _ensure_backend_alive(self):
        backend = self.server.backend
        if backend not in ("llama-cpp", "llama.cpp") or not self.server.model_path:
            return True
        url = f"{self.server.proxy_url.rstrip('/')}/health" if self.server.proxy_url else "http://127.0.0.1:8080/health"
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            print(f"[api] llama-server not responding on {url}. Attempting restart...", file=sys.stderr)
            self.server.reload_backend(self.server.model_path, self.server.backend)
            if getattr(self.server, "runtime_proc", None) is None:
                print(f"[api] llama-server restart failed (server not running)", file=sys.stderr)
                return False
            import time, urllib.request
            for _ in range(30):
                time.sleep(1)
                try:
                    urllib.request.urlopen(url, timeout=2)
                    print(f"[api] llama-server restarted successfully", file=sys.stderr)
                    return True
                except Exception:
                    pass
            print(f"[api] llama-server restart timed out after 30s", file=sys.stderr)
            return False

    def _forward_to_backend(self, body, request_id, chat=True):
        stream = body.get("stream", False)
        backend_name = getattr(self.server.active_backend, 'name', 'unknown') if self.server.active_backend else 'none'
        proxy = getattr(self.server, 'proxy_url', '')
        if not self._ensure_backend_alive():
            raise APIError(502, "Backend is not available and could not be restarted.", code="backend_unavailable")
        print(f"[api] Forwarding chat to {backend_name} ({proxy}): stream={stream}, model={body.get('model','?')}", file=sys.stderr)
        try:
            if chat:
                it = self.server.active_backend.chat_completions(body, stream)
            else:
                it = self.server.active_backend.completions(body, stream)
                
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_cors_headers()
                self.end_headers()
                self._streaming = True
                for chunk in it:
                    if isinstance(chunk, bytes):
                        self.wfile.write(chunk)
                    else:
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                self._streaming = False
            else:
                result = next(it)
                self.send_json(200, result, request_id)
        except backends.BackendError as e:
            print(f"[api] Backend error: {e.message} (code={e.code}, status={e.status})", file=sys.stderr)
            if stream and getattr(self, '_streaming', False):
                try:
                    self.wfile.write(f"data: {json.dumps({'error': {'message': e.message, 'code': e.code}})}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass
                self._streaming = False
                return
            raise APIError(e.status, e.message, code=e.code, error_type=e.error_type)
        except Exception as e:
            print(f"[api] Forward error: {type(e).__name__}: {e}", file=sys.stderr)
            if stream and getattr(self, '_streaming', False):
                try:
                    self.wfile.write(f"data: {json.dumps({'error': {'message': str(e), 'code': 'upstream_error'}})}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass
                self._streaming = False
                return
            raise

    def do_POST(self):
        request_id = "req_" + uuid.uuid4().hex
        try:
            self.require_auth()
            path = urlsplit(self.path).path
            print(f"[api] POST {path} (backend={self.server.backend}, model_id={self.server.model_id})", file=sys.stderr)
            
            if path == "/v1/qwanto/presets":
                body = self.read_json()
                if not isinstance(body, dict):
                    raise APIError(400, "Body must be a JSON object.")
                preset_id = body.get("id") or str(uuid.uuid4())[:8]
                new_preset = {
                    "id": preset_id,
                    "name": body.get("name") or "Custom Preset",
                    "system_prompt": body.get("system_prompt") or "",
                    "temperature": float(body.get("temperature", 0.7)),
                    "top_p": float(body.get("top_p", 0.9)),
                    "description": body.get("description") or "Custom user preset."
                }
                presets = load_presets()
                updated = False
                for i, p in enumerate(presets):
                    if p.get("id") == preset_id:
                        presets[i] = new_preset
                        updated = True
                        break
                if not updated:
                    presets.append(new_preset)
                save_presets(presets)
                self.send_json(200, {"status": "success", "preset": new_preset, "presets": presets}, request_id)
                return

            if path == "/v1/qwanto/presets/delete":
                body = self.read_json()
                preset_id = body.get("id") if isinstance(body, dict) else None
                if not preset_id:
                    raise APIError(400, "Missing preset id.", "id")
                presets = [p for p in load_presets() if p.get("id") != preset_id]
                save_presets(presets)
                self.send_json(200, {"status": "success", "presets": presets}, request_id)
                return

            if path == "/v1/qwanto/load":
                body = self.read_json()
                model_path = body.get("model_path")
                backend_type = body.get("backend", "auto")
                backend_url = body.get("backend_url")
                ctx_size = body.get("ctx_size")
                if not model_path:
                    raise APIError(400, "Missing model_path parameter.", "model_path")
                p = Path(model_path)
                if not p.exists():
                    raise APIError(404, f"Path does not exist: {model_path}", "model_path")
                accel = {}
                for key in ("flash_attention", "kv_cache_quant",
                            "speculative_decoding", "draft_model_path"):
                    if key in body:
                        accel[key] = body[key]
                if "kv_cache_quant" in accel and accel["kv_cache_quant"] not in ALLOWED_KV_QUANTS:
                    raise APIError(400, f"kv_cache_quant must be one of: {', '.join(ALLOWED_KV_QUANTS)}", "kv_cache_quant")
                draft_p = accel.get("draft_model_path")
                if accel.get("speculative_decoding") and draft_p and not os.path.exists(draft_p):
                    raise APIError(404, f"Draft model does not exist: {draft_p}", "draft_model_path")
                if backend_type == "auto":
                    if model_path.endswith(".gguf") or (p.is_file() and model_path.lower().endswith(".gguf")):
                        backend_type = "llama-cpp"
                    elif p.is_file() and model_path.lower().endswith(".qwn"):
                        backend_type = "qwn"
                    elif p.is_dir():
                        backend_type = "native"
                    else:
                        backend_type = "native"
                try:
                    self.server.reload_backend(model_path, backend_type, backend_url, ctx_size=ctx_size, accel=accel or None)
                    self.send_json(200, {
                        "status": "success",
                        "model_id": self.server.model_id,
                        "backend": self.server.backend
                    }, request_id)
                except Exception as e:
                    raise APIError(500, f"Failed to load model: {str(e)}", "model_path")
                return
                
            if path == "/v1/qwanto/download":
                body = self.read_json()
                url = body.get("url")
                filename = body.get("filename")
                dest_path_str = body.get("dest_path")
                if not url:
                    raise APIError(400, "Missing url parameter.", "url")
                inferred_name = filename or url.split("/")[-1].split("?")[0] or "model.gguf"
                if dest_path_str:
                    dest_path = Path(dest_path_str)
                    if not dest_path.is_absolute():
                        dest_path = Path(__file__).resolve().parent.parent / "models" / dest_path_str
                    if dest_path.is_dir() or not dest_path.suffix:
                        dest_path = dest_path / inferred_name
                else:
                    dest_path = Path(__file__).resolve().parent.parent / "models" / inferred_name
                try:
                    download_manager.start_download(url, dest_path)
                    self.send_json(200, {"status": "success", "message": f"Downloading to {dest_path}"}, request_id)
                except Exception as e:
                    raise APIError(400, str(e), "url")
                return
                
            if path == "/v1/qwanto/download/cancel":
                download_manager.cancel()
                self.send_json(200, {"status": "success", "message": "Download cancellation requested."}, request_id)
                return
                
            if path == "/v1/qwanto/download/pause":
                download_manager.pause()
                self.send_json(200, {"status": "success", "message": "Download paused."}, request_id)
                return
                
            if path == "/v1/qwanto/download/resume":
                download_manager.resume()
                self.send_json(200, {"status": "success", "message": "Download resumed."}, request_id)
                return
                
            if path == "/v1/qwanto/download/config":
                body = self.read_json()
                connections = body.get("connections")
                speed_limit = body.get("speed_limit")
                if connections is not None:
                    download_manager.set_connections(int(connections))
                if speed_limit is not None:
                    download_manager.set_speed_limit(int(speed_limit))
                self.send_json(200, {"status": "success", "connections": download_manager.connections, "speed_limit": download_manager.speed_limit}, request_id)
                return
                
            if path == "/v1/qwanto/convert":
                body = self.read_json()
                source = body.get("source")
                output = body.get("output")
                quant = body.get("quant", "q4_0")
                if not source:
                    raise APIError(400, "Missing 'source' model parameter.", "source")
                p = Path(source)
                if not p.exists():
                    raise APIError(404, f"Source model does not exist: {source}", "source")
                if not output:
                    if p.is_file():
                        output = str(p.parent / f"{p.stem}.qwn")
                    else:
                        output = str(p.parent / f"{p.name}.qwn")
                try:
                    conversion_manager.start_conversion(source, output, quant)
                    self.send_json(200, {"status": "started", "output": output, "message": f"Conversion started for {p.name}"}, request_id)
                except Exception as e:
                    raise APIError(400, str(e), "source")
                return

            if path == "/v1/qwanto/resources":
                body = self.read_json() if self.headers.get("content-length", "0") != "0" else {}
                if body:
                    resources = getattr(self.server, "resources", {"cpu": 100, "ram": 100, "vram": 100, "disk": 100})
                    for key in ("cpu", "ram", "vram", "disk"):
                        if key in body:
                            resources[key] = max(0, min(100, int(body[key])))
                    self.server.resources = resources
                    self.send_json(200, {"status": "success", "resources": resources}, request_id)
                else:
                    resources = getattr(self.server, "resources", {"cpu": 100, "ram": 100, "vram": 100, "disk": 100})
                    self.send_json(200, {"resources": resources}, request_id)
                return
                
            if path == "/v1/qwanto/delete":
                body = self.read_json()
                model_path = body.get("path")
                if not model_path:
                    raise APIError(400, "Missing path parameter.", "path")
                target = Path(model_path)
                if not _is_safe_path(target):
                    raise APIError(403, "Access denied: Path traversal or unauthorized directory modification blocked.", "path")
                if not target.exists():
                    raise APIError(404, "Model path not found.", "path")
                try:
                    import shutil
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                    self.send_json(200, {"status": "success", "message": f"Deleted {target}"}, request_id)
                except Exception as e:
                    raise APIError(500, f"Failed to delete: {str(e)}", "path")
                return
                
            if path == "/v1/qwanto/paths":
                body = self.read_json() if self.headers.get("content-length", "0") != "0" else {}
                custom_paths_file = Path(__file__).resolve().parent.parent / ".qwanto_model_paths.json"
                if body.get("action") == "add":
                    new_path = body.get("path")
                    if not new_path:
                        raise APIError(400, "Missing path parameter.", "path")
                    p = Path(new_path)
                    if not p.exists():
                        raise APIError(404, f"Path does not exist: {new_path}", "path")
                    existing = []
                    if custom_paths_file.exists():
                        try:
                            import json as _json
                            with open(custom_paths_file) as f:
                                existing = _json.load(f)
                        except Exception:
                            existing = []
                    if str(p) not in existing:
                        existing.append(str(p))
                        import json as _json
                        with open(custom_paths_file, "w") as f:
                            _json.dump(existing, f, indent=2)
                    self.send_json(200, {"status": "success", "paths": existing}, request_id)
                elif body.get("action") == "remove":
                    rm_path = body.get("path")
                    if not rm_path:
                        raise APIError(400, "Missing path parameter.", "path")
                    existing = []
                    if custom_paths_file.exists():
                        try:
                            import json as _json
                            with open(custom_paths_file) as f:
                                existing = _json.load(f)
                        except Exception:
                            existing = []
                    existing = [p for p in existing if p != rm_path]
                    import json as _json
                    with open(custom_paths_file, "w") as f:
                        _json.dump(existing, f, indent=2)
                    self.send_json(200, {"status": "success", "paths": existing}, request_id)
                else:
                    existing = []
                    if custom_paths_file.exists():
                        try:
                            import json as _json
                            with open(custom_paths_file) as f:
                                existing = _json.load(f)
                        except Exception:
                            existing = []
                    self.send_json(200, {"paths": existing}, request_id)
                return
                
            body = self.read_json()
            self.check_model(body)
            
            if path == "/v1/chat/completions":
                be = self.server.active_backend
                if be and be.name == "native":
                    self.chat_completion(body, request_id)
                elif be:
                    self._forward_to_backend(body, request_id, chat=True)
                else:
                    raise APIError(503, "No active backend loaded. Load a model first.", code="no_backend")
            elif path == "/v1/completions":
                be = self.server.active_backend
                if be and be.name == "native":
                    self.completion(body, request_id)
                elif be:
                    self._forward_to_backend(body, request_id, chat=False)
                else:
                    raise APIError(503, "No active backend loaded. Load a model first.", code="no_backend")
            else:
                raise APIError(404, "Not found.", None, "not_found")
        except APIError as error:
            self.send_json(error.status, error_object(error), request_id, error.headers)
        except ClientCancelled:
            pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.log_error("request failed: %s", error)
            api_error = APIError(500, "The qwanto engine failed to process the request.",
                                 None, "engine_error", "server_error")
            try:
                self.send_json(500, error_object(api_error), request_id)
            except OSError:
                pass

    def generation(self, body, prompt, request_id, chat):
        # QWANTO_DEBUG tees the engine transaction to stderr: 1 = decoded output stream only,
        # 2 = both sides (rendered prompt + output). render_chat already folds prior turns and
        # tool results into `prompt`, so level 2 is the full conversation the engine saw.
        try:
            dbg = int(os.environ.get("QWANTO_DEBUG", "0"))
        except ValueError:
            dbg = 0
        if dbg >= 2:
            sys.stderr.write(f"\n===== PROMPT [{request_id}] =====\n{prompt}\n===== OUTPUT [{request_id}] =====\n")
            sys.stderr.flush()
        maximum, temperature, top_p = generation_options(body, self.server.max_tokens)
        tools = (body.get("tools") or body.get("functions") or None) if chat else None
        if body.get("tool_choice") == "none":
            tools = None          # client forbade tools: never surface tool_calls
        cache_slot = body.get("cache_slot")
        if (cache_slot is not None and
                (isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or
                 not 0 <= cache_slot < self.server.kv_slots)):
            raise APIError(400, f"`cache_slot` must be an integer between 0 and {self.server.kv_slots - 1}.",
                           "cache_slot")
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise APIError(400, "`stream` must be a boolean.", "stream")
        stream_options = body.get("stream_options") if stream else None
        if stream and stream_options is not None and not isinstance(stream_options, dict):
            raise APIError(400, "`stream_options` must be an object.", "stream_options")
        include_usage = bool((stream_options or {}).get("include_usage"))
        object_name = "chat.completion" if chat else "text_completion"
        id_prefix = "chatcmpl-" if chat else "cmpl-"
        completion_id = id_prefix + uuid.uuid4().hex
        created = int(time.time())

        if not stream and hasattr(self.server, "response_cache"):
            cached_resp = self.server.response_cache.get(prompt, temperature, top_p, self.server.model_id)
            if cached_resp:
                self.send_json(200, cached_resp, request_id, {"x-qwanto-cache-hit": "true"})
                return

        with self.server.scheduler.admit(self.client_disconnected, cache_slot) as admission:
            queue_wait, cache_slot = admission
            queue_headers = {"x-qwanto-queue-wait-ms": str(round(queue_wait * 1000))}
            if not stream:
                output = []
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, output.append, cache_slot,
                    self.client_disconnected)
                text = "".join(output)
                length_finish = "length" if stats["length_limited"] else "stop"
                if chat and tools:
                    content, calls = parse_tool_calls(text, tools)
                    message = {"role": "assistant", "content": content or None, "refusal": None}
                    if calls:
                        message["tool_calls"] = calls
                    finish = "tool_calls" if calls else length_finish
                    choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": finish}
                else:
                    choice = ({"index": 0, "message": {"role": "assistant", "content": text,
                               "refusal": None}, "logprobs": None, "finish_reason": length_finish} if chat else
                              {"index": 0, "text": text, "logprobs": None, "finish_reason": length_finish})
                payload = {"id": completion_id, "object": object_name, "created": created,
                           "model": self.server.model_id, "choices": [choice], "usage": self.usage(stats)}
                if hasattr(self.server, "response_cache"):
                    self.server.response_cache.put(prompt, temperature, top_p, self.server.model_id, payload)
                self.send_json(200, payload, request_id, queue_headers)
                return

            stream_object = "chat.completion.chunk" if chat else object_name
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("x-request-id", request_id)
            for name, value in queue_headers.items(): self.send_header(name, value)
            self.send_cors_headers()
            self.end_headers()
            connected = True
            # KEEPALIVE: engine.generate() blocks SILENTLY during the (minutes-long) cold
            # prefill, and the client drops the socket after its idle timeout. A background pump
            # emits a reasoning_content "." delta (the channel that reliably resets the client's
            # timer and lands in the thinking panel, so answer content stays clean) whenever no
            # event has been written for KA_GAP seconds. All wfile writes share ka_lock so the
            # pump and event() never interleave; last_write gates the pump so it stays quiet
            # while real tokens are flowing (e.g. during decode).
            ka_lock = threading.Lock()
            last_write = [time.time()]
            ka_stop = threading.Event()
            KA_GAP = 10.0
            dbg_echo = dbg >= 1   # tee decoded tokens to stderr (QWANTO_DEBUG level parsed in generation())

            def event(choices, usage_marker=False):
                nonlocal connected
                if not connected:
                    return
                event_body = {"id": completion_id, "object": stream_object, "created": created,
                              "model": self.server.model_id, "choices": choices}
                if include_usage:
                    event_body["usage"] = None if not usage_marker else usage_marker
                data = json.dumps(event_body, ensure_ascii=False, separators=(",", ":"))
                with ka_lock:
                    try:
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                        last_write[0] = time.time()
                    except OSError:
                        connected = False

            def _keepalive():
                ping = [{"index": 0, "delta": ({"reasoning_content": "."} if chat else {"content": ""}),
                         "logprobs": None, "finish_reason": None}]
                while not ka_stop.wait(1.0):
                    if not connected:
                        return
                    if time.time() - last_write[0] >= KA_GAP:
                        event(ping)

            if chat:
                event([{"index": 0, "delta": {"role": "assistant", "content": ""},
                        "logprobs": None, "finish_reason": None}])

            def emit(text):
                choice = ({"index": 0, "delta": {"content": text}, "logprobs": None,
                           "finish_reason": None} if chat else
                          {"index": 0, "text": text, "logprobs": None, "finish_reason": None})
                event([choice])

            ka_thread = threading.Thread(target=_keepalive, daemon=True)
            ka_thread.start()
            if chat and tools:
                # Suppress tool-call markers from the streamed content and parse the authoritative
                # calls from the FULL reply after generation. Hold back a marker-length tail so a
                # <tool_call> split across engine chunks is still caught.
                parser = StreamParser(_tool_param_types(tools))
                
                # Metrics
                first_token_time = [None]
                first_content_time = [None]
                first_tool_time = [None]
                total_bytes = [0]
                backpressure_time = [0.0]
                
                def emit_tools(chunk):
                    if dbg_echo:
                        sys.stderr.write(chunk); sys.stderr.flush()
                        
                    now = time.time()
                    if first_token_time[0] is None:
                        first_token_time[0] = now
                        
                    deltas = parser.add_chunk(chunk)
                    for delta in deltas:
                        if "content" in delta and first_content_time[0] is None and delta["content"]:
                            first_content_time[0] = time.time()
                        if "tool_calls" in delta and first_tool_time[0] is None:
                            first_tool_time[0] = time.time()
                            
                        # Format delta for event
                        choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": None}
                        
                        # We hook into event directly to track bytes/backpressure
                        event_body = {"id": completion_id, "object": stream_object, "created": created,
                                      "model": self.server.model_id, "choices": [choice]}
                        data = json.dumps(event_body, ensure_ascii=False, separators=(",", ":"))
                        payload = f"data: {data}\n\n".encode()
                        total_bytes[0] += len(payload)
                        
                        nonlocal connected
                        if not connected: return
                        
                        with ka_lock:
                            try:
                                t0 = time.time()
                                self.wfile.write(payload)
                                self.wfile.flush()
                                backpressure_time[0] += (time.time() - t0)
                                last_write[0] = time.time()
                            except OSError:
                                connected = False
                
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, emit_tools, cache_slot,
                    lambda: not connected)
                
                # Finalize parser
                final_deltas = parser.finalize()
                for delta in final_deltas:
                    choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": None}
                    event_body = {"id": completion_id, "object": stream_object, "created": created,
                                  "model": self.server.model_id, "choices": [choice]}
                    data = json.dumps(event_body, ensure_ascii=False, separators=(",", ":"))
                    payload = f"data: {data}\n\n".encode()
                    total_bytes[0] += len(payload)
                    if connected:
                        with ka_lock:
                            try:
                                t0 = time.time()
                                self.wfile.write(payload)
                                self.wfile.flush()
                                backpressure_time[0] += (time.time() - t0)
                                last_write[0] = time.time()
                            except OSError:
                                connected = False
                
                if dbg >= 1:
                    sys.stderr.write(f"\n[metrics] Stream: {total_bytes[0]} bytes, "
                                     f"TTFT: {(first_token_time[0] or time.time()) - created:.3f}s, "
                                     f"FirstContent: {(first_content_time[0] or time.time()) - created:.3f}s, "
                                     f"FirstTool: {(first_tool_time[0] or time.time()) - created:.3f}s, "
                                     f"Backpressure: {backpressure_time[0]:.3f}s\n")
                    sys.stderr.flush()
                    
                finish = "tool_calls" if parser.tc_index > 0 else ("length" if stats["length_limited"] else "stop")
            else:
                def emit_plain(chunk):
                    if dbg_echo:
                        sys.stderr.write(chunk); sys.stderr.flush()
                    emit(chunk)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, emit_plain, cache_slot,
                    lambda: not connected)
                finish = "length" if stats["length_limited"] else "stop"
            ka_stop.set()                          # generation done: stop the keepalive pump
            ka_thread.join(timeout=2)
            final_choice = ({"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish}
                            if chat else {"index": 0, "text": "", "logprobs": None,
                                          "finish_reason": finish})
            event([final_choice])
            if include_usage:
                event([], self.usage(stats))
            if connected:
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass
            self.close_connection = True

    def client_disconnected(self):
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            flags = socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)
            return self.connection.recv(1, flags) == b""
        except (OSError, ValueError):
            return True

    @staticmethod
    def usage(stats):
        prompt = stats["prompt_tokens"]
        completion = stats["completion_tokens"]
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": prompt + completion}

    def chat_completion(self, body, request_id):
        reasoning_effort = body.get("reasoning_effort")
        efforts = (None, "none", "minimal", "low", "medium", "high", "xhigh")
        if reasoning_effort not in efforts:
            raise APIError(400, "`reasoning_effort` must be none, minimal, low, medium, high, or xhigh.",
                           "reasoning_effort")
        # QWANTO_THINK=1 makes thinking the default when the client sends NEITHER reasoning_effort
        # nor enable_thinking (a global switch, like the old server's --think). An explicit
        # client value always wins. Default off => exact OpenAI-standard behavior.
        if (reasoning_effort is None and "enable_thinking" not in body
                and os.environ.get("QWANTO_THINK", "0") == "1"):
            reasoning_effort = "high"
        enable_thinking = body.get("enable_thinking", reasoning_effort not in (None, "none"))
        if not isinstance(enable_thinking, bool):
            raise APIError(400, "`enable_thinking` must be a boolean.", "enable_thinking")
        tools = body.get("tools") or body.get("functions") or None
        prompt = render_chat(body.get("messages"), enable_thinking, reasoning_effort, tools,
                             body.get("tool_choice"))
        self.generation(body, prompt, request_id, True)

    def completion(self, body, request_id):
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise APIError(400, "Qwanto currently requires `prompt` to be a string.", "prompt")
        if not prompt:
            raise APIError(400, "`prompt` must not be empty.", "prompt")
        self.generation(body, prompt, request_id, False)


def serve(model, host="127.0.0.1", port=8000, model_id="glm-5.2-qwanto", api_key=None,
          cap=8, max_tokens=1024, engine=HERE / "glm", env=None, cors_origins=None,
          max_queue=8, queue_timeout=300, kv_slots=1, backend="auto", backend_url=None):
    if not 1 <= max_tokens:
        raise ValueError("max_tokens must be positive")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if max_queue < 0:
        raise ValueError("max_queue cannot be negative")
    if queue_timeout <= 0:
        raise ValueError("queue_timeout must be positive")
    if not 1 <= kv_slots <= 16:
        raise ValueError("kv_slots must be between 1 and 16")
    if host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        print("WARNING: API is listening beyond localhost without QWANTO_API_KEY", file=sys.stderr)

    # Restore last session settings (overrides QWANTO_MODEL env var if saved path exists)
    settings = APIServer._load_settings()
    saved_path = settings.get("model_path", "")
    saved_ctx_size = settings.get("ctx_size", 16384)
    if saved_path and os.path.exists(saved_path):
        model = saved_path
        model_exists = True
        print(f"Restored model from last session: {model} (ctx_size={saved_ctx_size})", file=sys.stderr)
        # Model path changed from original arg, force re-detection
        backend = "auto"
    else:
        model_exists = False
        if model:
            if os.path.exists(model):
                model_exists = True

    # Normalize backend name (coli detect_backend uses dot, serve uses hyphen)
    if backend is None:
        backend = "auto"
    elif backend == "llama.cpp":
        backend = "llama-cpp"

    # If auto, re-detect based on the (possibly restored) model path
    if backend == "auto":
        if not model or not model_exists:
            detected = "none"
        elif model.endswith(".gguf") or model.lower().endswith(".gguf"):
            detected = "llama-cpp"
        elif model.lower().endswith(".qwn"):
            detected = "qwn"
        elif model.startswith("hf.co/"):
            detected = "ollama"
        elif "/" in model or "\\" in model or model.startswith("."):
            detected = "native"
        elif not os.path.isdir(model):
            detected = "ollama"
        else:
            detected = "native"
    else:
        detected = backend

    if detected in ("native", "qwn", "llama-cpp") and not model_exists:
        print(f"Warning: Model path '{model}' not found. Starting in standby mode with no active model.", file=sys.stderr)
        detected = "none"

    origins = DEFAULT_CORS_ORIGINS if cors_origins is None else tuple(cors_origins)
    server = APIServer((host, port), None, model_id, api_key, max_tokens, origins,
                       max_queue, queue_timeout, kv_slots)
    server.backend = detected
    server.ctx_size = saved_ctx_size
    server.flash_attention = bool(settings.get("flash_attention", True))
    server.kv_cache_quant = str(settings.get("kv_cache_quant", "q4_0"))
    server.speculative_decoding = bool(settings.get("speculative_decoding", False))
    server.draft_model_path = str(settings.get("draft_model_path", "") or "")
    
    server.engine_executable = engine
    server.env = env
    server.cap = cap
    server.model_path = model
    if model:
        server.model_id = os.path.basename(model.rstrip("/\\")) or model
    
    runtime = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        if detected == "none":
            server.active_backend = NoneBackend("none", server)
        elif detected in ("native", "qwn"):
            executable = _qwn_executable(engine) if detected == "qwn" else engine
            if detected == "qwn" and not Path(executable).exists():
                raise RuntimeError("qwnrun is not built; run: make -C c qwnrun")
            runtime = Engine(executable,model,cap,max_tokens,env,kv_slots)
            server.engine = runtime
            server.active_backend = NativeBackend("native", server)
        elif detected == "ollama":
            server.proxy_url = backend_url or "http://127.0.0.1:11434"
            import urllib.request
            try:
                urllib.request.urlopen(f"{server.proxy_url.rstrip('/')}/", timeout=2)
            except Exception:
                print("Ollama is not running. Attempting to start Ollama...", file=sys.stderr)
                try:
                    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(3)
                except Exception as e:
                    print(f"Warning: Could not start Ollama: {e}", file=sys.stderr)
            print(f"Ensuring Ollama model '{model}' is available...", file=sys.stderr)
            try:
                subprocess.run(["ollama", "pull", model], check=True)
            except Exception as e:
                print(f"Warning: 'ollama pull' failed: {e}", file=sys.stderr)
            
            backends.prevent_recursive_routing(server.proxy_url, host, port)
            server.active_backend = backends.OllamaBackend(base_url=f"{server.proxy_url.rstrip('/')}/v1/")
        elif detected == "llama-cpp":
            server.proxy_url = backend_url or "http://127.0.0.1:8080"
            import urllib.request
            try:
                urllib.request.urlopen(f"{server.proxy_url.rstrip('/')}/", timeout=1)
                print("Found existing llama-server running", file=sys.stderr)
            except Exception:
                if not backend_url:
                    from resource_plan import physical_cpu_count
                    num_threads = physical_cpu_count()
                    print(f"Starting llama-server for GGUF model: {model} ({num_threads} threads)...", file=sys.stderr)
                    llm_exe = _ensure_llama_server()
                    if llm_exe:
                        cmd = _build_llama_cmd(llm_exe, model, server.ctx_size, num_threads, server=server)
                        try:
                            _llama_log = open(HERE / "llama_server.log", "w")
                            runtime = subprocess.Popen(cmd, stdout=_llama_log, stderr=subprocess.STDOUT)
                            server.runtime_proc = runtime
                            for _ in range(30):
                                time.sleep(1)
                                try:
                                    urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=1)
                                    print("llama-server started successfully", file=sys.stderr)
                                    break
                                except Exception:
                                    pass
                        except Exception as e:
                            print(f"Error: Could not spawn llama-server: {e}", file=sys.stderr)
                    else:
                        print("Could not obtain llama-server. Install llama.cpp or use --backend-url.", file=sys.stderr)
            backends.prevent_recursive_routing(server.proxy_url, host, port)
            server.active_backend = backends.LlamaCppBackend(base_url=f"{server.proxy_url.rstrip('/')}/v1/")
        elif detected == "openai":
            server.proxy_url = backend_url or "https://api.openai.com/v1"
            backends.prevent_recursive_routing(server.proxy_url, host, port)
            server.active_backend = backends.OpenAICompatibleBackend("openai", base_url=server.proxy_url, api_key=api_key)
                    
        server.runtime_proc = runtime
        server._save_settings()
        print(f"OpenAI-compatible API listening on http://{host}:{port}/v1", file=sys.stderr)
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.scheduler.close()
        server.server_close()
        if runtime is not None:
            runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("QWANTO_MODEL"), required=not os.environ.get("QWANTO_MODEL"))
    parser.add_argument("--engine", default=str(HERE / "glm"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=os.environ.get("QWANTO_MODEL_ID", "glm-5.2-qwanto"))
    parser.add_argument("--api-key", default=os.environ.get("QWANTO_API_KEY"))
    parser.add_argument("--cors-origin", action="append", default=None,
                        help="allowed browser origin; repeat as needed (use '*' for any origin)")
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-queue", type=int, default=int(os.environ.get("QWANTO_MAX_QUEUE", "8")))
    parser.add_argument("--queue-timeout", type=float,
                        default=float(os.environ.get("QWANTO_QUEUE_TIMEOUT", "300")))
    parser.add_argument("--kv-slots", type=int, default=int(os.environ.get("QWANTO_KV_SLOTS", "1")))
    parser.add_argument("--backend", choices=("native", "ollama", "llama-cpp", "llama.cpp", "openai", "auto"), default="auto")
    parser.add_argument("--backend-url", default=None, help="Explicit URL for openai, ollama, or llama-cpp backends")
    args = parser.parse_args()
    
    # normalize legacy names
    if args.backend == "colibri":
        args.backend = "native"
    elif args.backend == "llama.cpp":
        args.backend = "llama-cpp"

    serve(args.model, args.host, args.port, args.model_id, args.api_key,
          args.cap,args.max_tokens,args.engine,cors_origins=args.cors_origin,
          max_queue=args.max_queue,queue_timeout=args.queue_timeout,kv_slots=args.kv_slots,
          backend=args.backend, backend_url=args.backend_url)


if __name__ == "__main__":
    main()
