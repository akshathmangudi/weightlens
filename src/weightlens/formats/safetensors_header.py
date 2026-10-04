from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import NDArray

HEADER_LEN_BYTES = 8
MAX_HEADER_LEN = 100 * 1024 * 1024  # 100 MB

_FLOAT_DTYPES = frozenset({"F16", "F32", "F64", "BF16"})
_NP_LE = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}
_DTYPE_BITS = {
    "BOOL": 8,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "C64": 64,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}


@dataclass(frozen=True)
class TensorSlice:
    """Location and type of one tensor within a safetensors byte buffer."""

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
    header_len = int(n)
    if header_len > MAX_HEADER_LEN:
        raise ValueError(
            f"header length {header_len} exceeds maximum {MAX_HEADER_LEN}; "
            "file may be corrupt or malicious"
        )
    return header_len


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safetensors header key: {key!r}")
        result[key] = value
    return result


def _nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def validate_layout(slices: dict[str, TensorSlice], data_size: int) -> None:
    """Require non-overlapping tensor ranges that cover the data buffer."""
    if data_size < 0:
        raise ValueError("safetensors file ends before its data buffer")
    cursor = 0
    for sl in sorted(slices.values(), key=lambda item: (item.begin, item.end)):
        if sl.begin != cursor:
            raise ValueError(f"invalid offset for tensor {sl.name!r}")
        cursor = sl.end
    if cursor != data_size:
        raise ValueError("safetensors tensor offsets do not cover the data buffer")


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
        header = json.loads(
            header_bytes[HEADER_LEN_BYTES:end], object_pairs_hook=_unique_object
        )
    except json.JSONDecodeError as exc:
        raise ValueError("safetensors header is not valid JSON") from exc

    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")

    metadata = header.get("__metadata__")
    if "__metadata__" in header and (
        not isinstance(metadata, dict)
        or any(not isinstance(v, str) for v in metadata.values())
    ):
        raise ValueError("safetensors metadata must be a string map")

    slices: dict[str, TensorSlice] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict) or set(meta) != {
            "dtype", "shape", "data_offsets"
        }:
            raise ValueError(f"malformed safetensors header entry: {name!r}")
        dtype = meta["dtype"]
        shape = meta["shape"]
        offsets = meta["data_offsets"]
        if not isinstance(dtype, str) or dtype not in _DTYPE_BITS:
            raise ValueError(f"unsupported safetensors dtype for {name!r}: {dtype!r}")
        if not isinstance(shape, list):
            raise ValueError(f"invalid shape for tensor {name!r}")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"invalid offsets for tensor {name!r}")
        dimensions = tuple(_nonnegative_int(dim, "shape dimension") for dim in shape)
        begin, stop = (
            _nonnegative_int(offset, "data offset") for offset in offsets
        )
        nbits = prod(dimensions) * _DTYPE_BITS[dtype]
        if nbits % 8 != 0 or stop < begin or stop - begin != nbits // 8:
            raise ValueError(f"invalid shape or byte length for tensor {name!r}")
        slices[name] = TensorSlice(name, dtype, dimensions, begin, stop)
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
