from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from weightlens.formats.safetensors_header import (
    HEADER_LEN_BYTES,
    decode_float_tensor,
    parse_header,
    read_header_len,
)


def _build_st(tensors: dict[str, np.ndarray]) -> bytes:
    """Construct a minimal safetensors byte string by hand."""
    body = b""
    header: dict[str, object] = {}
    _st_dtype = {"float32": "F32", "float16": "F16", "float64": "F64"}
    for name, arr in tensors.items():
        raw = arr.astype(arr.dtype).tobytes()
        header[name] = {
            "dtype": _st_dtype[str(arr.dtype)],
            "shape": list(arr.shape),
            "data_offsets": [len(body), len(body) + len(raw)],
        }
        body += raw
    hjson = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(hjson)) + hjson + body


def test_read_header_len_roundtrips() -> None:
    prefix = struct.pack("<Q", 1234)
    assert read_header_len(prefix) == 1234


def test_read_header_len_rejects_truncated() -> None:
    with pytest.raises(ValueError):
        read_header_len(b"\x00\x00")


def test_parse_header_extracts_slices() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    blob = _build_st({"a.weight": a})
    header_len = read_header_len(blob[:HEADER_LEN_BYTES])
    header_len2, slices = parse_header(blob[: HEADER_LEN_BYTES + header_len])
    assert header_len2 == header_len
    sl = slices["a.weight"]
    assert sl.shape == (2, 3)
    assert sl.dtype == "F32"
    assert sl.is_float is True
    offset, length = sl.absolute_range(header_len)
    assert length == a.nbytes
    assert offset == HEADER_LEN_BYTES + header_len  # first tensor at base


def test_parse_header_skips_metadata_key() -> None:
    a = np.ones(4, dtype=np.float32)
    blob = bytearray(_build_st({"a": a}))
    # Manually inject __metadata__ into the header
    header_len_int = read_header_len(bytes(blob[:HEADER_LEN_BYTES]))
    header_start = HEADER_LEN_BYTES
    header_end = header_start + header_len_int
    header_json = json.loads(blob[header_start:header_end].decode("utf-8"))
    header_json["__metadata__"] = {"format": "pt"}
    new_header_json = json.dumps(header_json).encode("utf-8")
    new_header_len = len(new_header_json)

    # Rebuild blob with new header
    new_blob = struct.pack("<Q", new_header_len) + new_header_json + blob[header_end:]
    new_header_len_read = read_header_len(new_blob[:HEADER_LEN_BYTES])
    _, slices = parse_header(new_blob[: HEADER_LEN_BYTES + new_header_len_read])
    assert "__metadata__" not in slices
    assert "a" in slices  # real tensor should still be present


def test_decode_float_tensor_roundtrips_f32() -> None:
    a = np.linspace(-1, 1, 12, dtype=np.float32).reshape(3, 4)
    blob = _build_st({"w": a})
    header_len = read_header_len(blob[:HEADER_LEN_BYTES])
    _, slices = parse_header(blob[: HEADER_LEN_BYTES + header_len])
    sl = slices["w"]
    off, length = sl.absolute_range(header_len)
    decoded = decode_float_tensor(blob[off : off + length], sl)
    assert decoded.shape == (3, 4)
    assert np.array_equal(decoded, a)


def test_decode_bf16_upcasts_to_float32() -> None:
    # bf16 = top 16 bits of float32. Build raw bf16 bytes from known floats.
    import struct as _s

    from weightlens.formats.safetensors_header import TensorSlice

    vals = [1.0, -2.0, 0.5]
    raw = b"".join(
        _s.pack("<H", _s.unpack("<I", _s.pack("<f", v))[0] >> 16) for v in vals
    )
    sl = TensorSlice(name="b", dtype="BF16", shape=(3,), begin=0, end=len(raw))
    decoded = decode_float_tensor(raw, sl)
    assert decoded.dtype == np.float32
    assert np.allclose(decoded, np.array(vals, dtype=np.float32))
