"""Local-first model acquisition, verification, and conversion helpers.

This module intentionally uses only the Python standard library.  Network
access is performed only after a caller has built an explicit provider
manifest; CI tests use an explicitly enabled loopback HTTP server.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_LIBRARY = PROJECT_ROOT / "models"
DEFAULT_MAX_BYTES = 64 * 1024 * 1024 * 1024
MIN_FREE_BYTES = 64 * 1024 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
SUPPORTED_SOURCE_FORMATS = {"gguf", "safetensors", "pytorch"}
UNSUPPORTED_SOURCE_EXTENSIONS = {".onnx": "ONNX", ".h5": "Keras/H5", ".keras": "Keras"}


class AcquisitionError(ValueError):
    """A user-actionable acquisition or validation failure."""


@dataclass(frozen=True)
class ProviderManifest:
    provider: str
    artifact_id: str
    filename: str
    url: str
    format: str
    expected_size: Optional[int] = None
    sha256: Optional[str] = None
    gated: bool = False
    license_url: Optional[str] = None
    license_confirmed: bool = False
    verification: str = "unverified"

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_filename(filename: str) -> str:
    decoded = urllib.parse.unquote(filename).replace("\\", "/")
    name = Path(decoded).name
    if not name or name in {".", ".."} or "/" in decoded:
        raise AcquisitionError("The model filename must be a single safe path component.")
    if any(ord(char) < 32 for char in name) or name.startswith("."):
        raise AcquisitionError("The model filename contains a forbidden character.")
    return name


def _format_for_path(filename: str, *, directory: bool = False) -> str:
    lower = filename.lower()
    if directory or lower.endswith(".safetensors"):
        return "safetensors"
    if lower.endswith(".gguf"):
        return "gguf"
    if lower.endswith((".pt", ".pth", ".bin")):
        return "pytorch"
    for suffix, label in UNSUPPORTED_SOURCE_EXTENSIONS.items():
        if lower.endswith(suffix):
            raise AcquisitionError(f"{label} input is unsupported; no verified QWN reader is implemented.")
    raise AcquisitionError("Unsupported model format. Supported sources are GGUF, Safetensors, and PyTorch checkpoints.")


def detect_source_format(path: Path | str) -> str:
    source = Path(path)
    if source.is_dir():
        if not any(source.glob("*.safetensors")):
            raise AcquisitionError("A model directory must contain at least one .safetensors shard.")
        return "safetensors"
    if not source.is_file():
        raise AcquisitionError(f"Model source does not exist: {source}")
    try:
        with source.open("rb") as stream:
            magic = stream.read(4)
    except OSError as exc:
        raise AcquisitionError(f"Cannot inspect model source: {exc}") from exc
    if magic == b"GGUF":
        return "gguf"
    return _format_for_path(source.name)


def _host_allowed(host: str, allowed_hosts: set[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == allowed or host.endswith("." + allowed) for allowed in allowed_hosts)


def validate_download_url(
    url: str,
    *,
    allowed_hosts: set[str],
    allow_localhost_http: bool = False,
) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise AcquisitionError("Model downloads require an HTTPS URL.")
    if parsed.username or parsed.password or parsed.fragment:
        raise AcquisitionError("Model URLs must not contain credentials or fragments.")
    host = parsed.hostname.lower().rstrip(".")
    if parsed.scheme == "http":
        if not (allow_localhost_http and host in LOOPBACK_HOSTS):
            raise AcquisitionError("Plain HTTP is allowed only for an explicitly enabled loopback test server.")
    if not _host_allowed(host, allowed_hosts):
        raise AcquisitionError(f"Download host is not allowlisted: {host}")
    return parsed


class HuggingFaceProvider:
    HOSTS = {"huggingface.co"}

    @staticmethod
    def manifest(
        repository: str,
        filename: str,
        *,
        revision: str = "main",
        expected_size: Optional[int] = None,
        sha256: Optional[str] = None,
        gated: bool = False,
        license_url: Optional[str] = None,
        license_confirmed: bool = False,
    ) -> ProviderManifest:
        if "/" not in repository or any(part in {"", ".", ".."} for part in repository.split("/")):
            raise AcquisitionError("Hugging Face repository must be an owner/name pair.")
        if not revision or "/" in revision or ".." in revision:
            raise AcquisitionError("Hugging Face revision must be a single safe identifier.")
        decoded_filename = urllib.parse.unquote(filename).replace("\\", "/")
        if any(part in {"", ".", ".."} for part in decoded_filename.split("/")):
            raise AcquisitionError("Hugging Face filename contains an unsafe path component.")
        safe_name = _safe_filename(Path(decoded_filename).name)
        if gated and not license_confirmed:
            raise AcquisitionError("The gated model license must be explicitly confirmed before download.")
        url = "https://huggingface.co/{}/resolve/{}/{}".format(
            repository,
            urllib.parse.quote(revision, safe=""),
            urllib.parse.quote(filename.replace("\\", "/"), safe="/"),
        )
        return ProviderManifest(
            provider="huggingface",
            artifact_id=f"{repository}@{revision}:{filename}",
            filename=safe_name,
            url=url,
            format=_format_for_path(filename),
            expected_size=expected_size,
            sha256=sha256.lower() if sha256 else None,
            gated=gated,
            license_url=license_url,
            license_confirmed=license_confirmed,
            verification="verified" if sha256 else "unverified",
        )


class DirectHttpsProvider:
    @staticmethod
    def manifest(
        url: str,
        filename: Optional[str] = None,
        *,
        allowed_hosts: Optional[set[str]] = None,
        allow_localhost_http: bool = False,
        expected_size: Optional[int] = None,
        sha256: Optional[str] = None,
    ) -> ProviderManifest:
        parsed = urllib.parse.urlsplit(url)
        hosts = {h.lower().rstrip(".") for h in (allowed_hosts or set())}
        if not hosts:
            raise AcquisitionError("A direct URL requires an explicit approved host allowlist.")
        validate_download_url(url, allowed_hosts=hosts, allow_localhost_http=allow_localhost_http)
        chosen = _safe_filename(filename or Path(parsed.path).name)
        return ProviderManifest(
            provider="direct_https",
            artifact_id=url,
            filename=chosen,
            url=url,
            format=_format_for_path(chosen),
            expected_size=expected_size,
            sha256=sha256.lower() if sha256 else None,
            verification="verified" if sha256 else "unverified",
        )


class LocalFileProvider:
    @staticmethod
    def manifest(path: Path | str, *, expected_sha256: Optional[str] = None) -> ProviderManifest:
        source = Path(path)
        if not source.is_file():
            raise AcquisitionError(f"Local model file does not exist: {source}")
        return ProviderManifest(
            provider="local_file",
            artifact_id=str(source.resolve()),
            filename=_safe_filename(source.name),
            url="",
            format=detect_source_format(source),
            expected_size=source.stat().st_size,
            sha256=expected_sha256.lower() if expected_sha256 else None,
            verification="verified" if expected_sha256 else "unverified",
        )


def provider_catalog() -> list[dict]:
    """Return metadata only; this function performs no network access."""
    return [
        {
            "id": "huggingface",
            "name": "Hugging Face public artifacts",
            "network": True,
            "requires_https": True,
            "requires_license_confirmation_for_gated": True,
            "formats": ["gguf", "safetensors", "pytorch"],
        },
        {
            "id": "direct_https",
            "name": "Direct HTTPS",
            "network": True,
            "requires_https": True,
            "formats": ["gguf", "safetensors", "pytorch"],
        },
        {
            "id": "local_file",
            "name": "Local file import",
            "network": False,
            "requires_https": False,
            "formats": ["gguf", "safetensors", "pytorch"],
        },
    ]


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _within(root: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def destination_for(root: Path | str, filename: str) -> Path:
    library = Path(root).resolve()
    name = _safe_filename(filename)
    library.mkdir(parents=True, exist_ok=True)
    destination = (library / name).resolve()
    if not _within(library, destination):
        raise AcquisitionError("Model destination escapes the model library.")
    return destination


class SafeDownloadManager:
    def __init__(self, model_library: Path | str = DEFAULT_MODEL_LIBRARY, *, max_bytes: int = DEFAULT_MAX_BYTES):
        self.model_library = Path(model_library).resolve()
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.connections = 1
        self.speed_limit = 0
        self._status = self._initial_status()

    @staticmethod
    def _initial_status() -> dict:
        return {
            "status": "idle", "filename": "", "dest_path": "", "url": "",
            "provider": "", "downloaded": 0, "total": 0, "speed": 0.0,
            "speed_bytes_per_sec": 0.0, "progress": 0.0, "eta_seconds": None,
            "error": None, "verification": "unverified", "sha256": None,
            "partial_path": "", "retry_count": 0, "max_bytes": DEFAULT_MAX_BYTES,
            "connections": 1, "speed_limit": 0, "chunks_done": 0, "chunks_total": 0,
        }

    def _update(self, **values) -> None:
        with self.lock:
            self._status.update(values)

    def get_status(self) -> dict:
        with self.lock:
            return dict(self._status)

    def start_download(self, manifest: ProviderManifest, destination: Optional[Path | str] = None, *, overwrite: bool = False, allow_localhost_http: bool = False) -> None:
        with self.lock:
            if self._status["status"] in {"downloading", "paused"}:
                raise AcquisitionError("A download is already in progress.")
        target = destination_for(self.model_library, manifest.filename) if destination is None else Path(destination).resolve()
        if not _within(self.model_library, target):
            raise AcquisitionError("Download destination must remain inside the model library.")
        if target.exists() and not overwrite:
            raise AcquisitionError(f"Refusing to overwrite existing model: {target.name}")
        if manifest.expected_size is not None and (manifest.expected_size < 0 or manifest.expected_size > self.max_bytes):
            raise AcquisitionError("The model exceeds the configured maximum download size.")
        if manifest.provider != "local_file":
            hosts = {urllib.parse.urlsplit(manifest.url).hostname.lower()}
            validate_download_url(manifest.url, allowed_hosts=hosts, allow_localhost_http=allow_localhost_http)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.cancel_event.clear()
        self.pause_event.clear()
        partial = target.with_name(target.name + ".part")
        self._update(
            status="downloading", filename=manifest.filename, dest_path=str(target),
            partial_path=str(partial), url=manifest.url, provider=manifest.provider,
            downloaded=partial.stat().st_size if partial.exists() else 0,
            total=manifest.expected_size or 0, speed=0.0, speed_bytes_per_sec=0.0,
            progress=0.0, eta_seconds=None, error=None,
            verification=manifest.verification, sha256=manifest.sha256, retry_count=0,
            max_bytes=self.max_bytes, connections=self.connections, speed_limit=self.speed_limit,
        )
        self.thread = threading.Thread(
            target=self._run,
            args=(manifest, target, partial, allow_localhost_http, overwrite),
            daemon=True,
        )
        self.thread.start()

    def pause(self) -> None:
        with self.lock:
            if self._status["status"] == "downloading":
                self.pause_event.set()
                self._status["status"] = "paused"

    def resume(self) -> None:
        with self.lock:
            if self._status["status"] == "paused":
                self.pause_event.clear()
                self._status["status"] = "downloading"

    def cancel(self) -> None:
        with self.lock:
            if self._status["status"] in {"downloading", "paused"}:
                self.cancel_event.set()
                self.pause_event.clear()

    def set_connections(self, count: int) -> None:
        self.connections = max(1, min(32, int(count)))
        self._update(connections=self.connections)

    def set_speed_limit(self, bytes_per_second: int) -> None:
        self.speed_limit = max(0, int(bytes_per_second))
        self._update(speed_limit=self.speed_limit)

    def _check_disk(self, path: Path, required: int) -> None:
        usage = shutil.disk_usage(path.parent)
        if usage.free < required + MIN_FREE_BYTES:
            raise AcquisitionError(
                f"Insufficient free disk space: {usage.free} bytes available, {required + MIN_FREE_BYTES} required."
            )

    def _open_response(self, manifest: ProviderManifest, offset: int, allow_localhost_http: bool):
        if manifest.provider == "local_file":
            return None
        hosts = {urllib.parse.urlsplit(manifest.url).hostname.lower()}
        validate_download_url(manifest.url, allowed_hosts=hosts, allow_localhost_http=allow_localhost_http)
        headers = {"User-Agent": "Qwanto/0.1", "Accept": "application/octet-stream"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        response = urllib.request.urlopen(urllib.request.Request(manifest.url, headers=headers), timeout=60)
        final_url = response.geturl()
        final_hosts = {urllib.parse.urlsplit(manifest.url).hostname.lower()}
        validate_download_url(final_url, allowed_hosts=final_hosts, allow_localhost_http=allow_localhost_http)
        return response

    def _run(
        self,
        manifest: ProviderManifest,
        target: Path,
        partial: Path,
        allow_localhost_http: bool,
        overwrite: bool,
    ) -> None:
        try:
            if manifest.provider == "local_file":
                self._import_local(manifest, target, partial)
            else:
                self._download_remote(manifest, target, partial, allow_localhost_http)
            digest = sha256_file(partial)
            if manifest.sha256 and digest.lower() != manifest.sha256.lower():
                raise AcquisitionError(f"SHA-256 mismatch: expected {manifest.sha256}, got {digest}")
            if manifest.expected_size is not None and partial.stat().st_size != manifest.expected_size:
                raise AcquisitionError(f"Downloaded size mismatch: expected {manifest.expected_size}, got {partial.stat().st_size}")
            if target.exists() and not overwrite:
                raise AcquisitionError(f"Refusing to overwrite existing model: {target.name}")
            os.replace(partial, target)
            size = target.stat().st_size
            self._update(status="completed", downloaded=size, total=size, progress=100.0,
                         eta_seconds=0.0, verification="verified" if manifest.sha256 else "unverified",
                         sha256=digest, partial_path="", error=None)
        except Exception as exc:
            cancelled = self.cancel_event.is_set()
            if cancelled:
                try:
                    partial.unlink()
                except FileNotFoundError:
                    pass
            elif "SHA-256 mismatch" in str(exc):
                partial.unlink(missing_ok=True)
            self._update(status="error", error="Download cancelled by user" if cancelled else str(exc),
                         partial_path="" if cancelled or "SHA-256 mismatch" in str(exc) else str(partial))

    def _import_local(self, manifest: ProviderManifest, target: Path, partial: Path) -> None:
        source = Path(urllib.parse.urlparse(manifest.artifact_id).path if manifest.artifact_id.startswith("file:") else manifest.artifact_id)
        source = source.resolve()
        if not source.is_file():
            raise AcquisitionError(f"Local model file does not exist: {source}")
        size = source.stat().st_size
        if size > self.max_bytes:
            raise AcquisitionError("The local model exceeds the configured maximum size.")
        self._check_disk(target, size)
        shutil.copyfile(source, partial)

    def _download_remote(self, manifest: ProviderManifest, target: Path, partial: Path, allow_localhost_http: bool) -> None:
        offset = partial.stat().st_size if partial.exists() else 0
        started = time.monotonic()
        for attempt in range(4):
            if self.cancel_event.is_set():
                raise AcquisitionError("Download cancelled by user")
            try:
                response = self._open_response(manifest, offset, allow_localhost_http=allow_localhost_http)
                if response is None:
                    raise AcquisitionError("Remote provider did not return a response.")
                status = getattr(response, "status", response.getcode())
                accepts_range = status == 206 and offset > 0
                if offset and not accepts_range:
                    response.close()
                    offset = 0
                    partial.unlink(missing_ok=True)
                    response = self._open_response(manifest, 0, allow_localhost_http=allow_localhost_http)
                    status = getattr(response, "status", response.getcode())
                content_length = int(response.headers.get("Content-Length") or 0)
                total = manifest.expected_size or (offset + content_length if accepts_range else content_length)
                if total > self.max_bytes:
                    raise AcquisitionError("The remote model exceeds the configured maximum size.")
                self._check_disk(target, max(0, total - offset))
                mode = "ab" if accepts_range else "wb"
                if not accepts_range:
                    offset = 0
                with response, partial.open(mode) as output:
                    downloaded = offset
                    self._update(total=total, downloaded=downloaded)
                    while True:
                        if self.cancel_event.is_set():
                            raise AcquisitionError("Download cancelled by user")
                        while self.pause_event.is_set():
                            if self.cancel_event.wait(0.2):
                                raise AcquisitionError("Download cancelled by user")
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > self.max_bytes:
                            raise AcquisitionError("The remote model exceeded the configured maximum size.")
                        output.write(chunk)
                        elapsed = max(0.001, time.monotonic() - started)
                        speed = downloaded / elapsed
                        if self.speed_limit:
                            target_elapsed = downloaded / self.speed_limit
                            if target_elapsed > elapsed:
                                time.sleep(target_elapsed - elapsed)
                                elapsed = max(0.001, time.monotonic() - started)
                                speed = downloaded / elapsed
                        eta = (total - downloaded) / speed if total and speed > 0 else None
                        self._update(downloaded=downloaded, total=total,
                                    progress=(downloaded / total * 100.0) if total else 0.0,
                                    speed=speed / (1024 * 1024), speed_bytes_per_sec=speed,
                                    eta_seconds=eta)
                return
            except Exception:
                if attempt >= 3 or self.cancel_event.is_set():
                    raise
                self._update(retry_count=attempt + 1)
                offset = partial.stat().st_size if partial.exists() else 0
                time.sleep(0.2 * (attempt + 1))


def validate_qwn(path: Path | str) -> dict:
    """Validate QWN header, descriptor bounds, alignment, and output hash."""
    from tools import qwn_convert

    target = Path(path)
    if not target.is_file():
        raise AcquisitionError(f"QWN output does not exist: {target}")
    info = qwn_convert.inspect_qwn(str(target))
    file_size = target.stat().st_size
    if file_size < qwn_convert.HEADER_SIZE or file_size % 8:
        raise AcquisitionError("QWN output has an invalid file size.")
    if info["tail_offset"] % qwn_convert.ALIGN:
        raise AcquisitionError("QWN tail block is not 4KiB aligned.")
    for tensor in info["tensors"]:
        if tensor["byte_offset"] % qwn_convert.ALIGN or tensor["byte_size"] % 64:
            raise AcquisitionError(f"QWN tensor is not aligned: {tensor['name']}")
        if tensor["byte_offset"] < qwn_convert.HEADER_SIZE or tensor["byte_offset"] + tensor["byte_size"] > file_size:
            raise AcquisitionError(f"QWN tensor is outside the file: {tensor['name']}")
    return {"format": "qwn", "size_bytes": file_size, "sha256": sha256_file(target), "info": info}


def native_smoke_test(path: Path | str, executable: Optional[Path | str]) -> dict:
    if executable is None or not Path(executable).is_file():
        return {"status": "unavailable", "reason": "qwnrun executable is not available on this host"}
    try:
        env = dict(os.environ)
        env["SERVE"] = "1"
        process = subprocess.Popen(
            [str(executable), str(Path(path)), "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env,
        )
        stdout, stderr = process.communicate(input=b"PING\n", timeout=10)
        if b"PONG" in stdout:
            return {"status": "passed", "reason": "qwnrun opened the container and answered PING"}
        detail = stderr.decode("utf-8", "replace").strip()
        return {"status": "failed", "reason": detail or "qwnrun did not answer PING"}
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return {"status": "failed", "reason": "qwnrun smoke test timed out"}
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}


def convert_to_qwn(
    source: Path | str,
    destination: Path | str,
    quant: str,
    *,
    qwnrun: Optional[Path | str] = None,
    overwrite: bool = False,
    cancel_check=None,
) -> dict:
    source_path = Path(source).resolve()
    target = Path(destination).resolve()
    if target.suffix.lower() != ".qwn":
        raise AcquisitionError("Conversion output must use the .qwn extension.")
    if target.exists() and not overwrite:
        raise AcquisitionError(f"Refusing to overwrite existing QWN output: {target.name}")
    source_format = detect_source_format(source_path)
    part = target.with_name(target.name + ".part")
    if part.exists():
        part.unlink()
    try:
        from tools import qwn_convert

        qwn_convert.convert_model(str(source_path), str(part), quant)
        if cancel_check is not None and cancel_check():
            raise AcquisitionError("Conversion cancelled by user")
        validation = validate_qwn(part)
        smoke = native_smoke_test(part, qwnrun)
        if smoke["status"] == "failed":
            raise AcquisitionError(f"Native qwnrun smoke test failed: {smoke['reason']}")
        if cancel_check is not None and cancel_check():
            raise AcquisitionError("Conversion cancelled by user")
        os.replace(part, target)
        source_hash = sha256_file(source_path) if source_path.is_file() else None
        manifest = {
            "schema_version": 1,
            "source": str(source_path),
            "source_format": source_format,
            "source_sha256": source_hash,
            "output": str(target),
            "output_sha256": validation["sha256"],
            "output_size_bytes": validation["size_bytes"],
            "quantization": quant,
            "qwn_validation": "passed",
            "native_smoke_test": smoke,
            "verification": "verified" if smoke["status"] == "passed" else "verified_container_smoke_unavailable",
        }
        manifest_path = target.with_name(target.name + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    except Exception:
        part.unlink(missing_ok=True)
        raise
