import abc
import dataclasses
import json
import logging
import socket
import time
import uuid
import collections
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union
from urllib import request, error
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class BackendCapability:
    streaming: bool = True
    tool_calls: bool = False
    structured_output: bool = False
    reasoning: bool = False
    cancellation: bool = True
    model_discovery: bool = True

class BackendError(Exception):
    def __init__(self, status: int, message: str, code: str = None, error_type: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.error_type = error_type

class Backend(abc.ABC):
    def __init__(self, name: str, base_url: Optional[str] = None):
        self.name = name
        self.base_url = base_url

    @abc.abstractmethod
    def capabilities(self) -> BackendCapability:
        """Return the capabilities of this backend."""
        pass

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Check if the backend is reachable and healthy."""
        pass

    @abc.abstractmethod
    def models(self) -> List[Dict[str, Any]]:
        """Return a list of OpenAI-formatted model objects."""
        pass

    @abc.abstractmethod
    def chat_completions(self, body: Dict[str, Any], is_streaming: bool) -> Iterator[Union[Dict[str, Any], bytes]]:
        """
        Execute a chat completion.
        If is_streaming is True, yield SSE bytes.
        If is_streaming is False, yield the final JSON dictionary.
        """
        pass

    @abc.abstractmethod
    def completions(self, body: Dict[str, Any], is_streaming: bool) -> Iterator[Union[Dict[str, Any], bytes]]:
        """Execute a raw text completion."""
        pass

    @abc.abstractmethod
    def unload(self) -> bool:
        """Unload the model from memory if supported."""
        pass

def prevent_recursive_routing(url: str, server_host: str, server_port: int):
    """Raise BackendError if the target URL points to the current server."""
    if not url:
        return
    parsed = urlparse(url)
    target_port = parsed.port or (80 if parsed.scheme == "http" else 443)
    target_host = parsed.hostname
    if not target_host:
        return
    
    # Simple check for localhost loopback
    loopback_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
    
    server_is_loopback = server_host in loopback_hosts or server_host == ""
    target_is_loopback = target_host in loopback_hosts
    
    if server_is_loopback and target_is_loopback and target_port == server_port:
        raise BackendError(400, "Recursive routing detected: backend URL points to this server.", code="recursive_routing")
    
    # Try resolving if not simple loopback
    try:
        target_ip = socket.gethostbyname(target_host)
        server_ip = socket.gethostbyname(server_host) if server_host else "127.0.0.1"
    except socket.gaierror:
        return # DNS resolution failed, let the actual request fail later

    if target_ip == server_ip and target_port == server_port:
        raise BackendError(400, "Recursive routing detected: backend URL resolves to this server.", code="recursive_routing")

class OpenAICompatibleBackend(Backend):
    def __init__(self, name: str, base_url: str, api_key: str = None, timeout: float = 30.0):
        super().__init__(name, base_url)
        self.api_key = api_key
        self.timeout = timeout
        if not self.base_url.endswith("/"):
            self.base_url += "/"

    def capabilities(self) -> BackendCapability:
        return BackendCapability(
            streaming=True,
            tool_calls=True, # We assume True and let the remote endpoint fail if unsupported
            structured_output=True,
            reasoning=True,
            cancellation=False, # standard HTTP/1.1 cancellation propagates via dropped connections, but no explicit abort API
            model_discovery=True
        )

    def _request(self, endpoint: str, data: dict = None, stream: bool = False) -> request.Request:
        url = self.base_url + endpoint.lstrip("/")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if data is not None:
            data = json.dumps(data).encode("utf-8")
        req = request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        return req

    def health_check(self) -> bool:
        try:
            req = self._request("models")
            with request.urlopen(req, timeout=5.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def models(self) -> List[Dict[str, Any]]:
        try:
            req = self._request("models")
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("data", [])
        except Exception:
            return []

    def _stream_response(self, resp, read_timeout: float = 60.0) -> Iterator[bytes]:
        try:
            if hasattr(resp, 'fp') and hasattr(resp.fp, 'raw'):
                resp.fp.raw._sock.settimeout(read_timeout)
            elif hasattr(resp, 'fp') and hasattr(resp.fp, 'settimeout'):
                resp.fp.settimeout(read_timeout)
        except Exception:
            pass
        for line in resp:
            yield line

    def _forward_completions(self, endpoint: str, body: Dict[str, Any], is_streaming: bool) -> Iterator[Union[Dict[str, Any], bytes]]:
        body["stream"] = is_streaming
        req = self._request(endpoint, data=body)
        try:
            resp = request.urlopen(req, timeout=self.timeout)
            if is_streaming:
                yield from self._stream_response(resp)
            else:
                yield json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            err_body = e.read().decode("utf-8", "replace")
            try:
                err_json = json.loads(err_body)
                msg = err_json.get("error", {}).get("message", str(e))
                code = err_json.get("error", {}).get("code", "upstream_error")
            except Exception:
                msg = str(e)
                code = "upstream_error"
            raise BackendError(e.code, msg, code=code)
        except Exception as e:
            raise BackendError(502, f"Upstream error: {str(e)}", code="upstream_error")

    def chat_completions(self, body: Dict[str, Any], is_streaming: bool) -> Iterator[Union[Dict[str, Any], bytes]]:
        return self._forward_completions("chat/completions", body, is_streaming)

    def completions(self, body: Dict[str, Any], is_streaming: bool) -> Iterator[Union[Dict[str, Any], bytes]]:
        return self._forward_completions("completions", body, is_streaming)

    def unload(self) -> bool:
        return False # generic OpenAI doesn't support model unloading

class OllamaBackend(OpenAICompatibleBackend):
    def __init__(self, name: str = "ollama", base_url: str = "http://localhost:11434/v1/"):
        super().__init__(name, base_url)

    def capabilities(self) -> BackendCapability:
        cap = super().capabilities()
        cap.tool_calls = False # Usually partially supported but often broken in Ollama
        return cap

class LlamaCppBackend(OpenAICompatibleBackend):
    def __init__(self, name: str = "llama-cpp", base_url: str = "http://localhost:8080/v1/"):
        super().__init__(name, base_url, timeout=30.0)

    def health_check(self) -> bool:
        try:
            req = self._request("health")
            with request.urlopen(req, timeout=5.0) as resp:
                return resp.status == 200
        except Exception:
            # Fallback to models endpoint
            return super().health_check()
