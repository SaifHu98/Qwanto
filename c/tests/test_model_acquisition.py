import hashlib
import http.server
import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from model_acquisition import (
    AcquisitionError,
    DirectHttpsProvider,
    HuggingFaceProvider,
    SafeDownloadManager,
    detect_source_format,
    provider_catalog,
    validate_download_url,
    convert_to_qwn,
)
from tools import qwn_convert


PAYLOAD = b"GGUF" + bytes(range(256)) * 64


class RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = PAYLOAD
    slow = False

    def do_GET(self):
        start = 0
        end = len(self.payload) - 1
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            start = int(range_header[6:].split("-", 1)[0])
            if start >= len(self.payload):
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.payload)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        for offset in range(start, end + 1, 4096):
            self.wfile.write(self.payload[offset:min(offset + 4096, end + 1)])
            self.wfile.flush()
            if self.slow:
                time.sleep(0.002)

    def log_message(self, *_args):
        return


class ModelAcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.httpd.server_port}/model.gguf"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

    def test_provider_manifest_is_metadata_only_and_gated_download_requires_confirmation(self):
        catalog = provider_catalog()
        self.assertEqual({item["id"] for item in catalog}, {"huggingface", "direct_https", "local_file"})
        manifest = HuggingFaceProvider.manifest("Qwanto/example", "model.gguf", sha256="a" * 64)
        self.assertEqual(manifest.provider, "huggingface")
        self.assertEqual(manifest.verification, "verified")
        with self.assertRaises(AcquisitionError):
            HuggingFaceProvider.manifest("Qwanto/example", "model.gguf", gated=True)

    def test_url_and_destination_boundaries(self):
        validate_download_url(self.url, allowed_hosts={"127.0.0.1"}, allow_localhost_http=True)
        with self.assertRaises(AcquisitionError):
            validate_download_url("http://example.test/model.gguf", allowed_hosts={"example.test"})
        with self.assertRaises(AcquisitionError):
            DirectHttpsProvider.manifest(self.url, "../escape.gguf", allowed_hosts={"127.0.0.1"}, allow_localhost_http=True)

    def test_download_resume_and_checksum_verification(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.gguf"
            partial = target.with_name("model.gguf.part")
            partial.write_bytes(PAYLOAD[:4096])
            manifest = DirectHttpsProvider.manifest(
                self.url, allowed_hosts={"127.0.0.1"}, allow_localhost_http=True,
                expected_size=len(PAYLOAD), sha256=hashlib.sha256(PAYLOAD).hexdigest(),
            )
            manager = SafeDownloadManager(root, max_bytes=len(PAYLOAD) + 1024)
            manager.start_download(manifest, target, allow_localhost_http=True)
            manager.thread.join(timeout=5)
            status = manager.get_status()
            self.assertEqual(status["status"], "completed")
            self.assertEqual(target.read_bytes(), PAYLOAD)
            self.assertEqual(status["verification"], "verified")
            self.assertFalse(partial.exists())

    def test_checksum_mismatch_and_cancel_cleanup(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = DirectHttpsProvider.manifest(
                self.url, allowed_hosts={"127.0.0.1"}, allow_localhost_http=True,
                expected_size=len(PAYLOAD), sha256="0" * 64,
            )
            manager = SafeDownloadManager(root, max_bytes=len(PAYLOAD) + 1024)
            manager.start_download(manifest, allow_localhost_http=True)
            manager.thread.join(timeout=5)
            self.assertEqual(manager.get_status()["status"], "error")
            self.assertIn("SHA-256 mismatch", manager.get_status()["error"])
            self.assertFalse((root / "model.gguf").exists())
            self.assertFalse((root / "model.gguf.part").exists())

        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.gguf"
            target.write_bytes(b"old model")
            manifest = DirectHttpsProvider.manifest(
                self.url, allowed_hosts={"127.0.0.1"}, allow_localhost_http=True,
                expected_size=len(PAYLOAD), sha256=hashlib.sha256(PAYLOAD).hexdigest(),
            )
            manager = SafeDownloadManager(root, max_bytes=len(PAYLOAD) + 1024)
            manager.start_download(manifest, target, overwrite=True, allow_localhost_http=True)
            manager.thread.join(timeout=5)
            self.assertEqual(manager.get_status()["status"], "completed")
            self.assertEqual(target.read_bytes(), PAYLOAD)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            RangeHandler.slow = True
            try:
                manifest = DirectHttpsProvider.manifest(
                    self.url, allowed_hosts={"127.0.0.1"}, allow_localhost_http=True,
                    expected_size=len(PAYLOAD),
                )
                manager = SafeDownloadManager(root, max_bytes=len(PAYLOAD) + 1024)
                manager.start_download(manifest, allow_localhost_http=True)
                time.sleep(0.01)
                manager.cancel()
                manager.thread.join(timeout=5)
                self.assertEqual(manager.get_status()["status"], "error")
                self.assertIn("cancelled", manager.get_status()["error"].lower())
                self.assertFalse((root / "model.gguf.part").exists())
                self.assertFalse((root / "model.gguf").exists())
            finally:
                RangeHandler.slow = False

    def test_disk_preflight_and_unsupported_formats_fail_fast(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = DirectHttpsProvider.manifest(
                self.url, allowed_hosts={"127.0.0.1"}, allow_localhost_http=True,
                expected_size=len(PAYLOAD),
            )
            manager = SafeDownloadManager(root, max_bytes=len(PAYLOAD) + 1024)
            with patch("model_acquisition.shutil.disk_usage", return_value=SimpleNamespace(free=1)):
                manager.start_download(manifest, allow_localhost_http=True)
                manager.thread.join(timeout=5)
            self.assertEqual(manager.get_status()["status"], "error")
            self.assertIn("free disk", manager.get_status()["error"])

            onnx = root / "model.onnx"
            onnx.write_bytes(b"not an onnx model")
            with self.assertRaises(AcquisitionError):
                detect_source_format(onnx)
            with self.assertRaisesRegex(ValueError, "unsupported source format"):
                qwn_convert.convert_model(str(onnx), str(root / "bad.qwn"), "q4_0")

    def test_conversion_publishes_atomic_qwn_and_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "weights.safetensors"
            raw = json.dumps({"weight": {"dtype": "F32", "shape": [1, 32], "data_offsets": [0, 128]}}).encode()
            source.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(128))
            output = root / "weights.qwn"
            manifest = convert_to_qwn(source, output, "q4_0")
            self.assertTrue(output.exists())
            self.assertTrue(output.with_name("weights.qwn.manifest.json").exists())
            self.assertFalse(output.with_name("weights.qwn.part").exists())
            self.assertEqual(manifest["qwn_validation"], "passed")
            self.assertIn(manifest["native_smoke_test"]["status"], {"passed", "unavailable"})


if __name__ == "__main__":
    unittest.main()
