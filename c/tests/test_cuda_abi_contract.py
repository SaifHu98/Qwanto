"""Static/host-side checks for the versioned CUDA contract.

These checks deliberately do not classify a GPU as compiled or tested. They
make the no-CUDA-toolkit path auditable while the real CUDA test remains an
NVCC/device gate.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ABI = ROOT / "cuda" / "qwn_cuda_abi.h"


def test_versioned_abi_declares_exact_hypervsq2_contract():
    source = ABI.read_text(encoding="utf-8")
    assert "QWN_CUDA_ABI_VERSION 1u" in source
    assert "QWN_CUDA_HYPERVSQ2_BLOCK_BYTES 74u" in source
    assert "QWN_CUDA_HYPERVSQ2_BLOCK_ELEMENTS 256u" in source
    assert "QWN_CUDA_MAX_RESIDENT_TENSORS 512u" in source
    for symbol in (
        "qwn_cuda_abi_query",
        "qwn_cuda_abi_get_capabilities",
        "qwn_cuda_abi_enumerate_devices",
        "qwn_cuda_abi_context_create",
        "qwn_cuda_abi_context_destroy",
        "qwn_cuda_abi_upload_tensor",
        "qwn_cuda_abi_hypervsq2_gemv",
        "qwn_cuda_abi_hypervsq2_gemm",
        "qwn_cuda_abi_synchronize",
        "qwn_cuda_abi_get_telemetry",
        "qwn_cuda_abi_last_error",
    ):
        assert symbol in source
    assert "struct_size" in source and "reserved[4]" in source


def test_windows_loader_does_not_accept_arbitrary_dll_path():
    loader = (ROOT / "qwanto_decode.c").read_text(encoding="utf-8")
    assert "QWANTO_CUDA_DLL" not in loader
    assert "LoadLibraryExA" in loader
    assert "LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR" in loader
    assert "qwn_cuda_abi_query" in loader
    assert "legacy CUDA DLLs are rejected" in loader


def test_host_can_parse_the_abi_header_when_clang_is_available():
    clang = shutil.which("clang")
    if not clang:
        return
    with tempfile.TemporaryDirectory() as temporary:
        probe = Path(temporary) / "abi_probe.c"
        probe.write_text(
            '#include "cuda/qwn_cuda_abi.h"\n'
            'int main(void) { QwnCudaAbiInfo info = {0}; '
            'qwn_cuda_abi_header_init(&info.header, sizeof(info)); '
            'return info.header.abi_version == 1u ? 0 : 1; }\n',
            encoding="utf-8",
        )
        result = subprocess.run(
            [clang, "-std=c11", "-I", str(ROOT), "-fsyntax-only", str(probe)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
