import unittest
from unittest.mock import patch, MagicMock
import urllib.error
import urllib.request
import json
import socket
import sys
import os

# Add parent dir to path to import backends
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backends

class TestBackends(unittest.TestCase):

    def test_recursive_routing_loopback(self):
        with self.assertRaises(backends.BackendError) as ctx:
            backends.prevent_recursive_routing("http://127.0.0.1:8000/v1", "127.0.0.1", 8000)
        self.assertEqual(ctx.exception.code, "recursive_routing")

    @patch('socket.gethostbyname')
    def test_recursive_routing_dns(self, mock_gethostbyname):
        mock_gethostbyname.side_effect = lambda h: "192.168.1.100"
        with self.assertRaises(backends.BackendError) as ctx:
            backends.prevent_recursive_routing("http://my-domain.com:8000/v1", "my-server", 8000)
        self.assertEqual(ctx.exception.code, "recursive_routing")

    def test_recursive_routing_safe(self):
        # Should not raise
        backends.prevent_recursive_routing("http://127.0.0.1:8080/v1", "127.0.0.1", 8000)

    def test_ollama_capabilities(self):
        backend = backends.OllamaBackend()
        caps = backend.capabilities()
        self.assertTrue(caps.streaming)
        self.assertFalse(caps.tool_calls) # Ollama overrides tool_calls to False
        self.assertFalse(caps.cancellation)

    def test_llama_cpp_capabilities(self):
        backend = backends.LlamaCppBackend()
        caps = backend.capabilities()
        self.assertTrue(caps.streaming)
        self.assertTrue(caps.tool_calls) # LlamaCpp retains True
        self.assertFalse(caps.cancellation)

    @patch('urllib.request.urlopen')
    def test_timeout_enforcement(self, mock_urlopen):
        # Simulate a timeout
        mock_urlopen.side_effect = urllib.error.URLError(socket.timeout("timed out"))
        
        backend = backends.OpenAICompatibleBackend("openai", "https://api.openai.com/v1", timeout=0.1)
        with self.assertRaises(backends.BackendError) as ctx:
            list(backend.chat_completions({"messages": [{"role": "user", "content": "hello"}]}, False))
        
        self.assertEqual(ctx.exception.status, 502)
        self.assertIn("timed out", ctx.exception.message)

    @patch('urllib.request.urlopen')
    def test_unsupported_functionality_errors(self, mock_urlopen):
        # Mock an error response from an upstream simulating "unsupported parameter"
        err_response = MagicMock()
        err_response.read.return_value = json.dumps({
            "error": {"message": "Invalid parameter 'tool_choice'", "code": "invalid_request_error"}
        }).encode('utf-8')
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=err_response
        )

        backend = backends.OpenAICompatibleBackend("openai", "https://api.openai.com/v1")
        with self.assertRaises(backends.BackendError) as ctx:
            list(backend.chat_completions({"tool_choice": "required"}, False))
        
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request_error")
        self.assertIn("Invalid parameter", ctx.exception.message)

    @patch('urllib.request.urlopen')
    def test_cancellation(self, mock_urlopen):
        # Python urllib doesn't have an explicit async cancel API that fits sync generators without
        # dropping the socket. We simulate a dropped connection mid-stream.
        mock_resp = MagicMock()
        
        def mock_iter():
            yield b'data: {"choices": []}\n\n'
            raise ConnectionResetError("Connection closed by peer")

        mock_resp.__iter__.return_value = mock_iter()
        mock_urlopen.return_value = mock_resp

        backend = backends.OpenAICompatibleBackend("openai", "https://api.openai.com/v1")
        
        generator = backend.chat_completions({"messages": []}, True)
        first_chunk = next(generator)
        self.assertEqual(first_chunk, b'data: {"choices": []}\n\n')
        
        with self.assertRaises(backends.BackendError) as ctx:
            next(generator)
        self.assertEqual(ctx.exception.status, 502)
        self.assertIn("Connection closed", ctx.exception.message)

if __name__ == '__main__':
    unittest.main()
