"""
qwn_loader.py — Pure-Python, dependency-free reader for .qwn containers.

Exposes :class:`QwnTensor` (mmap-ed payload view) and
:class:`QwnModel` (header + tensor index + tokenizer access).  Every
tensor row can be decoded to float32 on demand, or returned as raw
bytes for format-preserving consumers (the same payload can be re-
written as GGUF or safetensors without re-quantizing).

This is the foundation of the Qwanto universal compatibility layer:
the same .qwn file is consumed by ``qwnrun`` (native decoder) and by
``qwn2gguf.py`` / ``qwn2safetensors.py`` (exporters for llama.cpp,
HuggingFace, PyTorch).
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union


HEADER_MAGIC = b"QWANTO_NATIVE_V1"
HEADER_PREFIX_FMT = "<16sIIIIIIQ8Q"
HEADER_PREFIX_SIZE = struct.calcsize(HEADER_PREFIX_FMT)   # 112
DESC_FMT = "<64sIII4Q3QI"
DESC_SIZE = struct.calcsize(DESC_FMT)                     # 136
ALIGN = 4096
INLINE_MAX = 29

# dtype ids (mirror c/tools/qwn_convert.py)
DT_F32, DT_F16, DT_Q4_0, DT_Q8_0, DT_BF16, DT_BYTES, \
    DT_VSQ, DT_VSQ_ULTRA, DT_HYPER_VSQ, DT_HYPER_VSQ2 = range(10)
DT_NAME = {
    DT_F32: "F32", DT_F16: "F16", DT_Q4_0: "Q4_0", DT_Q8_0: "Q8_0",
    DT_BF16: "BF16", DT_BYTES: "BYTES", DT_VSQ: "VSQ",
    DT_VSQ_ULTRA: "VSQ_ULTRA", DT_HYPER_VSQ: "HYPER_VSQ",
    DT_HYPER_VSQ2: "HYPER_VSQ2",
}


@dataclass
class QwnTensor:
    name: str
    shape: Tuple[int, ...]
    dtype: int
    byte_offset: int
    byte_size: int
    payload_size: int
    _mmap: Optional[mmap.mmap] = field(default=None, repr=False)
    _file: Optional[object] = field(default=None, repr=False)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    @property
    def n_bytes(self) -> int:
        return self.byte_size

    @property
    def dtype_name(self) -> str:
        return DT_NAME.get(self.dtype, f"DT_{self.dtype}")

    def bytes(self) -> bytes:
        """Return the raw payload bytes (mmap-backed, copies on slice)."""
        if self._mmap is None:
            raise RuntimeError("tensor not attached to a QwnModel")
        return self._mmap[self.byte_offset:self.byte_offset + self.payload_size]

    def as_float32(self, out: Optional[bytes] = None) -> bytes:
        """Decode this tensor to float32 bytes.

        Supports F32, F16, BF16, Q4_0 payloads.  Q4_0 superblock layout:
        18-byte blocks of (fp16 scale, 32 nibbles) for 32 elements.
        Returns ``len(self.numel) * 4`` bytes.
        """
        if self.dtype == DT_F32:
            raw = self.bytes()
            return raw if len(raw) == self.numel * 4 else raw[:self.numel * 4]
        if self.dtype == DT_F16:
            raw = self.bytes()
            return _f16_to_f32(raw, self.numel)
        if self.dtype == DT_BF16:
            raw = self.bytes()
            return _bf16_to_f32(raw, self.numel)
        if self.dtype == DT_Q4_0:
            return _q4_0_to_f32(self.bytes(), self.numel)
        if self.dtype == DT_BYTES:
            return self.bytes()  # opaque
        raise NotImplementedError(
            f"as_float32: unsupported dtype {self.dtype_name} "
            f"for tensor {self.name!r}")


def _f16_to_f32(raw: bytes, numel: int) -> bytes:
    """IEEE half -> float32 little-endian."""
    import struct as _s
    words = _s.unpack(f"<{numel}H", raw[:numel * 2])
    out = bytearray(numel * 4)
    for i, w in enumerate(words):
        _s.pack_into("<I", out, i * 4, w)
    return bytes(out)


def _bf16_to_f32(raw: bytes, numel: int) -> bytes:
    """bfloat16 -> float32 little-endian (top-16 bits placed in fp32)."""
    import struct as _s
    words = _s.unpack(f"<{numel}H", raw[:numel * 2])
    out = bytearray(numel * 4)
    for i, w in enumerate(words):
        _s.pack_into("<I", out, i * 4, w << 16)
    return bytes(out)


def _q4_0_to_f32(raw: bytes, numel: int) -> bytes:
    """Q4_0 superblock (18 bytes per 32 elements) -> float32."""
    blocks = (numel + 31) // 32
    out = bytearray(numel * 4)
    import struct as _s
    for b in range(blocks):
        off = b * 18
        scale = _s.unpack_from("<e", raw, off)[0]
        sub_numel = min(32, numel - b * 32)
        for j in range(sub_numel):
            byte_off = off + 2 + j // 2
            lo = raw[byte_off] & 0x0F if (j & 1) == 0 else (raw[byte_off] >> 4) & 0x0F
            q = lo - 8 if lo >= 8 else -(8 - lo)  # map back: signed 4-bit
            # Stored value: ((q + 8) & 15) so q = (lo - 8) for lo<8, lo-8 for lo>=8
            q = lo - 8
            _s.pack_into("<f", out, (b * 32 + j) * 4, q * scale)
    return bytes(out)


@dataclass
class QwnModel:
    """A loaded .qwn container with mmap access to every tensor."""

    path: Path
    arch: str
    arch_dims: Tuple[int, ...]
    tensors: List[QwnTensor]
    config: Dict[str, object]
    _mmap: mmap.mmap
    _file: object

    @classmethod
    def open(cls, path: Union[str, Path]) -> "QwnModel":
        path = Path(path).resolve()
        f = open(path, "rb")
        try:
            header = f.read(HEADER_PREFIX_SIZE)
        except Exception:
            f.close()
            raise
        if header[:16] != HEADER_MAGIC:
            f.close()
            raise ValueError(f"{path} is not a .qwn file (bad magic)")
        (magic, version, _, arch_code, n_tensors, inline_count, _, n_params,
         *arch_dims8) = struct.unpack(HEADER_PREFIX_FMT, header)
        arch_dims = tuple(int(x) for x in arch_dims8)

        # Tail block (after all inline payloads).
        # We don't need its exact offset to read the inline descriptors;
        # descriptors are placed at header_prefix + i*DESC_SIZE for i
        # in [0, inline_count).  Overflow descriptors live past the
        # tail block; their offsets come from the sorted FNV-1a index.
        size = os.fstat(f.fileno()).st_size
        mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_READ)

        # Inline descriptors
        tensors: List[QwnTensor] = []
        for i in range(inline_count):
            off = HEADER_PREFIX_SIZE + i * DESC_SIZE
            row = struct.unpack_from(DESC_FMT, mm, off)
            # Layout: <64s III 4Q 3Q I>  ->  name, name_len, dtype,
            # n_dims, shape[0..3], numel, byte_offset, byte_size, block_q
            name_raw   = row[0]
            dtype      = row[2]
            n_dims     = row[3]
            shape      = tuple(int(row[4 + j]) for j in range(min(n_dims, 4))) if n_dims > 0 else ()
            numel      = int(row[8])
            byte_off   = int(row[9])
            byte_size  = int(row[10])
            payload_size = byte_size
            name = name_raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
            tensors.append(QwnTensor(
                name=name, shape=shape, dtype=int(dtype),
                byte_offset=byte_off, byte_size=byte_size,
                payload_size=payload_size,
            ))

        # Overflow index: locate the tail block.  The tail is always the
        # last 8 bytes of the file in v1; before it sits the FNV-1a
        # overflow index.  We use the simpler approach of scanning from
        # the end for the last 8-byte offset pair.
        if n_tensors > inline_count:
            overflow_count = n_tensors - inline_count
            # Layout: tail (32 bytes) | overflow descriptors (DESC_SIZE
            # each) | sorted FNV-1a index (16 bytes each).
            tail_offset = struct.unpack_from("<Q", mm, size - 8)[0]
            desc_offset = tail_offset + 32
            for i in range(overflow_count):
                doff = desc_offset + i * DESC_SIZE
                row = struct.unpack_from(DESC_FMT, mm, doff)
                name_raw   = row[0]
                dtype      = row[2]
                n_dims     = row[3]
                shape      = tuple(int(row[4 + j]) for j in range(min(n_dims, 4))) if n_dims > 0 else ()
                numel      = int(row[8])
                byte_off   = int(row[9])
                byte_size  = int(row[10])
                payload_size = byte_size
                name = name_raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
                tensors.append(QwnTensor(
                    name=name, shape=shape, dtype=int(dtype),
                    byte_offset=byte_off, byte_size=byte_size,
                    payload_size=payload_size,
                ))

        # Attach mmap to every tensor (zero-copy view).
        for t in tensors:
            t._mmap = mm
            t._file = f

        # Optional embedded config (for the arch-name + dims metadata).
        arch = arch_name_for(arch_code)
        config: Dict[str, object] = {}
        for t in tensors:
            if t.name == "__qwn.config":
                raw = t.bytes()
                # strip trailing NUL padding (byte_size is 64-byte aligned)
                raw = raw.rstrip(b"\x00")
                try:
                    config = json.loads(raw.decode("utf-8", "replace"))
                except Exception:
                    config = {}
                break

        return cls(path=path, arch=arch, arch_dims=arch_dims,
                   tensors=tensors, config=config, _mmap=mm, _file=f)

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None        # type: ignore
        if self._file is not None:
            self._file.close()
            self._file = None       # type: ignore
        for t in self.tensors:
            t._mmap = None
            t._file = None

    def __enter__(self) -> "QwnModel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self) -> Iterator[QwnTensor]:
        return iter(self.tensors)

    def __getitem__(self, name: str) -> QwnTensor:
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(name)

    def __contains__(self, name: str) -> bool:
        return any(t.name == name for t in self.tensors)


def arch_name_for(arch_code: int) -> str:
    return {0: "qwen", 1: "llama", 2: "moe", 3: "mamba", 4: "hybrid"}.get(
        arch_code, "unknown")


__all__ = [
    "QwnTensor", "QwnModel", "DT_NAME",
    "DT_F32", "DT_F16", "DT_Q4_0", "DT_Q8_0", "DT_BF16",
    "DT_BYTES", "DT_VSQ", "DT_VSQ_ULTRA", "DT_HYPER_VSQ", "DT_HYPER_VSQ2",
]