from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

HEADER_LEN_BYTES = 8

_FLOAT_DTYPES = frozenset({"F16", "F32", "F64", "BF16"})
_NP_LE = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}


@dataclass(frozen=True)
class TensorSlice:
    """Location + type of one tensor within a safetensors byte buffer."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def is_float(self) -> bool:
        return self.dtype in _FLOAT_DTYPES

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    def absolute_range(self, header_len: int) -> tuple[int, int]:
        """Return (absolute_offset, length) in the file's byte space."""
        base = HEADER_LEN_BYTES + header_len
        return base + self.begin, self.end - self.begin


def read_header_len(prefix: bytes) -> int:
    if len(prefix) < HEADER_LEN_BYTES:
        raise ValueError("truncated safetensors file: missing 8-byte length prefix")
    (n,) = struct.unpack("<Q", prefix[:HEADER_LEN_BYTES])
    return int(n)


def parse_header(header_bytes: bytes) -> tuple[int, dict[str, TensorSlice]]:
    """Parse the first ``8 + N`` bytes into (header_len, {name: TensorSlice})."""
    header_len = read_header_len(header_bytes)
    end = HEADER_LEN_BYTES + header_len
    if len(header_bytes) < end:
        raise ValueError(
            f"need {end} header bytes, got {len(header_bytes)}; "
            "read 8 + header_len bytes before calling parse_header"
        )
    try:
        header = json.loads(header_bytes[HEADER_LEN_BYTES:end])
    except json.JSONDecodeError as exc:
        raise ValueError("safetensors header is not valid JSON") from exc

    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")

    slices: dict[str, TensorSlice] = {}
    # Wrap per-entry parsing so a malformed tensor record (missing dtype/shape/
    # data_offsets, wrong types) surfaces as a clean ValueError rather than a
    # raw KeyError/TypeError leaking to callers.
    try:
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, stop = meta["data_offsets"]
            slices[name] = TensorSlice(
                name=name,
                dtype=str(meta["dtype"]),
                shape=tuple(int(d) for d in meta["shape"]),
                begin=int(begin),
                end=int(stop),
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed safetensors header entry: {exc}") from exc
    return header_len, slices


def decode_float_tensor(raw: bytes, sl: TensorSlice) -> NDArray[np.floating]:
    """Interpret ``raw`` bytes as a float ndarray. bf16 is upcast to float32."""
    if sl.dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        arr: NDArray[np.floating] = (u16 << 16).view(np.float32)
    elif sl.dtype in _NP_LE:
        arr = np.frombuffer(raw, dtype=_NP_LE[sl.dtype])
    else:
        raise ValueError(f"not a float dtype: {sl.dtype}")
    return arr.reshape(sl.shape)
