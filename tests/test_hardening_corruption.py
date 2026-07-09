"""Adversarial tests: corruption & truncation resilience of the safetensors
reader + validator.

Covers, by hand-crafted malformed bytes (struct.pack + json, no real
safetensors writer involved):

1. file shorter than the 8-byte length prefix
2. header-length prefix N larger than the actual file
3. header bytes that are not valid JSON
4. a tensor's data_offsets extending past the file size
5. an empty (0-byte) file
6. a well-formed header but truncated tensor payload

For every case we assert:
  * ``SafetensorsCheckpointValidator(...).validate()`` never raises and
    returns ``loadable=False`` with a non-empty ``corruption_flags`` list
    (bit-exact where the failure is guaranteed detectable from the header
    alone).
  * ``SafetensorsWeightSource(...).iter_layers()`` either raises a clean
    ``ValueError`` (never a raw ``struct.error``/``KeyError``/``json.
    JSONDecodeError``) or completes without crashing the process.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import fsspec
import pytest

from weightlens.sources.safetensors import SafetensorsWeightSource
from weightlens.validators.safetensors import SafetensorsCheckpointValidator

HEADER_LEN_BYTES = 8


def _write_local(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return str(path)


def _write_memory(uri: str, data: bytes) -> str:
    with fsspec.open(uri, "wb") as f:
        f.write(data)
    return uri


def _valid_header_bytes(tensors: dict[str, dict[str, object]]) -> bytes:
    """Build a syntactically valid safetensors prefix (len + JSON header)."""
    hjson = json.dumps(tensors).encode("utf-8")
    return struct.pack("<Q", len(hjson)) + hjson


# --------------------------------------------------------------------------
# 1. File shorter than the 8-byte length prefix.
# --------------------------------------------------------------------------


def test_validator_handles_file_shorter_than_prefix(tmp_path: Path) -> None:
    path = _write_local(tmp_path / "short.safetensors", b"\x01\x02\x03")
    health = SafetensorsCheckpointValidator(path).validate()
    assert health.loadable is False
    assert health.corruption_flags != []
    assert health.tensor_count == 0
    assert health.total_params == 0


def test_source_raises_clean_value_error_for_file_shorter_than_prefix(
    tmp_path: Path,
) -> None:
    path = _write_local(tmp_path / "short.safetensors", b"\x01\x02\x03")
    with pytest.raises(ValueError, match="truncated safetensors file"):
        list(SafetensorsWeightSource(path).iter_layers())


def test_file_shorter_than_prefix_over_memory_uri() -> None:
    uri = _write_memory("memory://hardening_c1.safetensors", b"\xff\xfe")
    health = SafetensorsCheckpointValidator(uri).validate()
    assert health.loadable is False
    assert health.corruption_flags != []
    with pytest.raises(ValueError):
        list(SafetensorsWeightSource(uri).iter_layers())


# --------------------------------------------------------------------------
# 2. Header-length prefix declares N larger than the file.
# --------------------------------------------------------------------------


def test_validator_handles_oversized_header_length_prefix(tmp_path: Path) -> None:
    # Declare a 10_000-byte header but only supply 2 bytes of body ("{}").
    blob = struct.pack("<Q", 10_000) + b"{}"
    path = _write_local(tmp_path / "oversized_header.safetensors", blob)
    health = SafetensorsCheckpointValidator(path).validate()
    assert health.loadable is False
    assert health.corruption_flags != []
    assert "10008" in health.corruption_flags[0] or "header" in (
        health.corruption_flags[0].lower()
    )


def test_source_raises_clean_value_error_for_oversized_header_length(
    tmp_path: Path,
) -> None:
    blob = struct.pack("<Q", 10_000) + b"{}"
    path = _write_local(tmp_path / "oversized_header.safetensors", blob)
    with pytest.raises(ValueError):
        list(SafetensorsWeightSource(path).iter_layers())


# --------------------------------------------------------------------------
# 3. Header bytes are not valid JSON.
# --------------------------------------------------------------------------


def test_validator_handles_invalid_json_header(tmp_path: Path) -> None:
    bad_json = b"{this is not json at all"
    blob = struct.pack("<Q", len(bad_json)) + bad_json
    path = _write_local(tmp_path / "bad_json.safetensors", blob)
    health = SafetensorsCheckpointValidator(path).validate()
    assert health.loadable is False
    assert health.corruption_flags != []
    assert "json" in health.corruption_flags[0].lower()


def test_source_raises_clean_value_error_for_invalid_json_header(
    tmp_path: Path,
) -> None:
    bad_json = b"{this is not json at all"
    blob = struct.pack("<Q", len(bad_json)) + bad_json
    path = _write_local(tmp_path / "bad_json.safetensors", blob)
    with pytest.raises(ValueError, match="not valid JSON"):
        list(SafetensorsWeightSource(path).iter_layers())


# --------------------------------------------------------------------------
# 4. A tensor's data_offsets extend past the file size.
# --------------------------------------------------------------------------


def test_source_raises_clean_value_error_for_offsets_past_eof(
    tmp_path: Path,
) -> None:
    # Header claims a 4x4 F32 tensor (64 bytes) but the file only has 8
    # trailing bytes after the header.
    header: dict[str, dict[str, object]] = {
        "w": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}
    }
    prefix = _valid_header_bytes(header)
    blob = prefix + b"\x00" * 8
    path = _write_local(tmp_path / "offsets_past_eof.safetensors", blob)
    with pytest.raises(ValueError):
        list(SafetensorsWeightSource(path).iter_layers())


def test_validator_flags_offsets_extending_past_file_size(
    tmp_path: Path,
) -> None:
    header: dict[str, dict[str, object]] = {
        "w": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}
    }
    prefix = _valid_header_bytes(header)
    blob = prefix + b"\x00" * 8  # only 8 of the claimed 64 body bytes exist
    path = _write_local(tmp_path / "offsets_past_eof.safetensors", blob)

    health = SafetensorsCheckpointValidator(path).validate()

    assert health.loadable is False
    assert health.corruption_flags != []


# --------------------------------------------------------------------------
# 5. Empty (0-byte) file.
# --------------------------------------------------------------------------


def test_validator_handles_empty_file(tmp_path: Path) -> None:
    path = _write_local(tmp_path / "empty.safetensors", b"")
    health = SafetensorsCheckpointValidator(path).validate()
    assert health.loadable is False
    assert health.is_empty is True
    assert health.corruption_flags != []
    assert health.file_size_bytes == 0
    assert health.tensor_count == 0
    assert health.total_params == 0


def test_source_raises_clean_value_error_for_empty_file(tmp_path: Path) -> None:
    path = _write_local(tmp_path / "empty.safetensors", b"")
    with pytest.raises(ValueError, match="truncated safetensors file"):
        list(SafetensorsWeightSource(path).iter_layers())


def test_empty_file_over_memory_uri() -> None:
    uri = _write_memory("memory://hardening_c5_empty.safetensors", b"")
    health = SafetensorsCheckpointValidator(uri).validate()
    assert health.loadable is False
    assert health.is_empty is True
    assert health.corruption_flags != []


# --------------------------------------------------------------------------
# 6. Well-formed header but truncated tensor data.
# --------------------------------------------------------------------------


def test_source_raises_clean_value_error_for_truncated_tensor_body(
    tmp_path: Path,
) -> None:
    # Header correctly declares a 4-element F32 vector (16 bytes) but the
    # file is chopped off after only 7 body bytes.
    header: dict[str, dict[str, object]] = {
        "w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}
    }
    prefix = _valid_header_bytes(header)
    blob = prefix + b"\x00" * 7
    path = _write_local(tmp_path / "truncated_body.safetensors", blob)
    with pytest.raises(ValueError):
        list(SafetensorsWeightSource(path).iter_layers())


def test_validator_flags_truncated_tensor_body(tmp_path: Path) -> None:
    # The validator cross-checks declared data_offsets against the actual
    # file size, so a truncated tensor body (offsets past EOF) is reported
    # as corruption (loadable=False), consistent with
    # test_validator_flags_offsets_extending_past_file_size.
    header: dict[str, dict[str, object]] = {
        "w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}
    }
    prefix = _valid_header_bytes(header)
    blob = prefix + b"\x00" * 7
    path = _write_local(tmp_path / "truncated_body.safetensors", blob)

    health = SafetensorsCheckpointValidator(path).validate()

    assert health.loadable is False
    assert health.corruption_flags != []


def test_truncated_tensor_body_over_memory_uri() -> None:
    header: dict[str, dict[str, object]] = {
        "w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}
    }
    prefix = _valid_header_bytes(header)
    blob = prefix + b"\x00" * 7
    uri = _write_memory("memory://hardening_c6_truncated.safetensors", blob)
    with pytest.raises(ValueError):
        list(SafetensorsWeightSource(uri).iter_layers())


# --------------------------------------------------------------------------
# Extra: a required tensor-metadata key missing entirely (adjacent
# corruption shape not explicitly enumerated above, but exercised by the
# same "malformed header" family and required by the "never raw
# struct/KeyError" contract).
# --------------------------------------------------------------------------


def test_validator_handles_header_missing_dtype_key(tmp_path: Path) -> None:
    header: dict[str, dict[str, object]] = {
        "w": {"shape": [4], "data_offsets": [0, 16]}
    }
    prefix = _valid_header_bytes(header)
    blob = prefix + b"\x00" * 16
    path = _write_local(tmp_path / "missing_dtype.safetensors", blob)

    health = SafetensorsCheckpointValidator(path).validate()

    assert health.loadable is False
    assert health.corruption_flags != []


def test_source_raises_clean_value_error_for_missing_dtype_key(
    tmp_path: Path,
) -> None:
    header: dict[str, dict[str, object]] = {
        "w": {"shape": [4], "data_offsets": [0, 16]}
    }
    prefix = _valid_header_bytes(header)
    blob = prefix + b"\x00" * 16
    path = _write_local(tmp_path / "missing_dtype.safetensors", blob)

    with pytest.raises(ValueError):
        list(SafetensorsWeightSource(path).iter_layers())


# --------------------------------------------------------------------------
# Sharded (.index.json) corruption: malformed shard referenced by an
# otherwise well-formed index must not crash the validator either.
# --------------------------------------------------------------------------


def test_validator_handles_corrupt_shard_referenced_by_index(
    tmp_path: Path,
) -> None:
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    _write_local(ckpt_dir / shard_name, b"\x01\x02")  # corrupt: too short

    index_path = ckpt_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": {"w": shard_name}})
    )

    health = SafetensorsCheckpointValidator(str(index_path)).validate()

    assert health.loadable is False
    assert health.corruption_flags != []
    assert health.tensor_count == 0
