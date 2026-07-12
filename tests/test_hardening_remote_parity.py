"""Parity tests: local path vs file:// vs memory:// URIs.

Same safetensors bytes read via three URI schemes must produce byte-identical
streamed tensors and identical validator health. Tests ByteRangeReader boundary
reads against plain Python slices.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import fsspec
import numpy as np
import pytest

from tests.fixtures_safetensors import write_sharded, write_single
from weightlens.io.byte_range import ByteRangeReader
from weightlens.models import CheckpointHealth
from weightlens.sources.safetensors import SafetensorsWeightSource
from weightlens.validators.safetensors import SafetensorsCheckpointValidator


def _mem_root() -> str:
    """Unique memory:// directory per test to avoid leaking state."""
    return f"memory://wl-hardening-{uuid.uuid4().hex}"


def _mirror_file_to_memory(local_path: Path, mem_uri: str) -> None:
    """Byte-for-byte copy a local file into the in-memory filesystem."""
    data = local_path.read_bytes()
    with fsspec.open(mem_uri, "wb") as f:
        f.write(data)


def _health_fields(h: CheckpointHealth) -> tuple[int, bool, bool, int, int, list[str]]:
    return (
        h.file_size_bytes,
        h.is_empty,
        h.loadable,
        h.tensor_count,
        h.total_params,
        h.corruption_flags,
    )




def test_single_file_local_vs_file_uri_vs_memory_uri_bit_exact(
    tmp_path: Path,
) -> None:
    tensors: dict[str, np.ndarray] = {
        "model.a.weight": np.random.randn(4, 3).astype(np.float32),
        "model.b.bias": np.random.randn(8).astype(np.float32),
        "model.c.norm": np.random.randn(2, 2, 2).astype(np.float64),
    }
    local_path = tmp_path / "model.safetensors"
    write_single(str(local_path), tensors)

    mem_uri = f"{_mem_root()}/model.safetensors"
    _mirror_file_to_memory(local_path, mem_uri)

    file_uri = f"file://{local_path}"

    local_result = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(str(local_path)).iter_layers()
    }
    file_result = {
        lt.name: lt.values for lt in SafetensorsWeightSource(file_uri).iter_layers()
    }
    mem_result = {
        lt.name: lt.values for lt in SafetensorsWeightSource(mem_uri).iter_layers()
    }

    assert set(local_result) == set(file_result) == set(mem_result) == set(tensors)
    for name in tensors:
        assert np.array_equal(local_result[name], file_result[name])
        assert np.array_equal(local_result[name], mem_result[name])
        assert local_result[name].dtype == file_result[name].dtype
        assert local_result[name].dtype == mem_result[name].dtype
        assert local_result[name].tobytes() == mem_result[name].tobytes()


def test_single_file_validator_health_identical_across_schemes(
    tmp_path: Path,
) -> None:
    tensors: dict[str, np.ndarray] = {
        "w1": np.random.randn(10, 10).astype(np.float32),
        "w2": np.random.randn(5).astype(np.float32),
        "step": np.array([1, 2, 3], dtype=np.int64),
    }
    local_path = tmp_path / "model.safetensors"
    write_single(str(local_path), tensors)

    mem_uri = f"{_mem_root()}/model.safetensors"
    _mirror_file_to_memory(local_path, mem_uri)
    file_uri = f"file://{local_path}"

    h_local = SafetensorsCheckpointValidator(str(local_path)).validate()
    h_file = SafetensorsCheckpointValidator(file_uri).validate()
    h_mem = SafetensorsCheckpointValidator(mem_uri).validate()

    assert _health_fields(h_local) == _health_fields(h_file) == _health_fields(h_mem)
    assert h_local.tensor_count == 2
    assert h_local.total_params == 100 + 5
    assert h_local.loadable is True
    assert h_local.is_empty is False
    assert h_local.corruption_flags == []




def test_byte_range_reader_boundary_reads_match_python_slices() -> None:
    data = bytes(range(256)) * 4  # 1024 bytes, deterministic pattern
    uri = f"{_mem_root()}/blob.bin"
    with fsspec.open(uri, "wb") as f:
        f.write(data)

    reader = ByteRangeReader(uri)
    assert reader.size() == len(data)

    assert reader.read(0, 4) == data[0:4]
    assert reader.read(0, len(data)) == data[0 : len(data)]
    tail_len = 17
    assert (
        reader.read(len(data) - tail_len, tail_len)
        == data[len(data) - tail_len : len(data)]
    )
    assert reader.read(123, 0) == b""
    assert reader.read(0, 0) == b""
    assert reader.read(len(data), 0) == b""
    assert reader.read(500, 1) == data[500:501]
    assert reader.read(37, 200) == data[37:237]


def test_byte_range_reader_sequential_reads_reconstruct_full_blob() -> None:
    data = bytes((i * 7 + 3) % 256 for i in range(999))
    uri = f"{_mem_root()}/blob2.bin"
    with fsspec.open(uri, "wb") as f:
        f.write(data)

    reader = ByteRangeReader(uri)
    chunk = 64
    reassembled = bytearray()
    offset = 0
    while offset < len(data):
        length = min(chunk, len(data) - offset)
        reassembled += reader.read(offset, length)
        offset += length

    assert bytes(reassembled) == data




def test_sharded_checkpoint_memory_uri_bit_exact_parity(tmp_path: Path) -> None:
    shard_a: dict[str, np.ndarray] = {
        "model.layers.0.w": np.random.randn(6, 5).astype(np.float32),
    }
    shard_b: dict[str, np.ndarray] = {
        "model.layers.1.w": np.random.randn(5, 5).astype(np.float32),
        "model.norm.weight": np.random.randn(5).astype(np.float32),
    }
    local_dir = tmp_path / "ckpt"
    local_index_path = write_sharded(str(local_dir), [shard_a, shard_b])

    mem_root = _mem_root()
    mem_dir_name = "ckpt"
    for entry in sorted(Path(local_dir).iterdir()):
        _mirror_file_to_memory(entry, f"{mem_root}/{mem_dir_name}/{entry.name}")
    mem_index_uri = f"{mem_root}/{mem_dir_name}/model.safetensors.index.json"

    local_result = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(local_index_path).iter_layers()
    }
    mem_result = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(mem_index_uri).iter_layers()
    }

    expected_names = {*shard_a, *shard_b}
    assert set(local_result) == set(mem_result) == expected_names
    for name in expected_names:
        assert np.array_equal(local_result[name], mem_result[name])
        assert local_result[name].tobytes() == mem_result[name].tobytes()
        assert local_result[name].dtype == mem_result[name].dtype
        assert local_result[name].shape == mem_result[name].shape


def test_sharded_checkpoint_validator_health_matches_local_and_memory(
    tmp_path: Path,
) -> None:
    shard_a: dict[str, np.ndarray] = {
        "a1": np.random.randn(3, 3).astype(np.float32),
        "a2": np.random.randn(4).astype(np.float32),
    }
    shard_b: dict[str, np.ndarray] = {
        "b1": np.random.randn(2, 2).astype(np.float32),
    }
    local_dir = tmp_path / "ckpt2"
    local_index_path = write_sharded(str(local_dir), [shard_a, shard_b])

    mem_root = _mem_root()
    for entry in sorted(Path(local_dir).iterdir()):
        _mirror_file_to_memory(entry, f"{mem_root}/ckpt2/{entry.name}")
    mem_index_uri = f"{mem_root}/ckpt2/model.safetensors.index.json"

    h_local = SafetensorsCheckpointValidator(local_index_path).validate()
    h_mem = SafetensorsCheckpointValidator(mem_index_uri).validate()

    assert _health_fields(h_local) == _health_fields(h_mem)
    assert h_local.tensor_count == 3
    assert h_local.total_params == 9 + 4 + 4
    assert h_local.loadable is True
    assert h_local.corruption_flags == []


def test_sharded_checkpoint_file_uri_index_bit_exact_parity(tmp_path: Path) -> None:
    shard_a: dict[str, np.ndarray] = {
        "x.weight": np.random.randn(7, 2).astype(np.float32),
    }
    shard_b: dict[str, np.ndarray] = {
        "y.weight": np.random.randn(2, 7).astype(np.float32),
    }
    local_dir = tmp_path / "ckpt3"
    local_index_path = write_sharded(str(local_dir), [shard_a, shard_b])
    file_index_uri = f"file://{local_index_path}"

    local_result = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(local_index_path).iter_layers()
    }
    file_result = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(file_index_uri).iter_layers()
    }

    expected_names = {*shard_a, *shard_b}
    assert set(local_result) == set(file_result) == expected_names
    for name in expected_names:
        assert np.array_equal(local_result[name], file_result[name])
        assert local_result[name].tobytes() == file_result[name].tobytes()




def test_byte_range_reader_local_vs_file_vs_memory_same_offsets(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, size=2048, dtype=np.uint8).tobytes()

    local_path = tmp_path / "raw.bin"
    local_path.write_bytes(data)
    mem_uri = f"{_mem_root()}/raw.bin"
    _mirror_file_to_memory(local_path, mem_uri)
    file_uri = f"file://{local_path}"

    local_reader = ByteRangeReader(str(local_path))
    file_reader = ByteRangeReader(file_uri)
    mem_reader = ByteRangeReader(mem_uri)

    assert local_reader.size() == file_reader.size() == mem_reader.size() == 2048

    probes = [(0, 1), (0, 2048), (2000, 48), (2048, 0), (0, 0), (999, 1)]
    for offset, length in probes:
        expected = data[offset : offset + length]
        assert local_reader.read(offset, length) == expected
        assert file_reader.read(offset, length) == expected
        assert mem_reader.read(offset, length) == expected


def test_byte_range_reader_rejects_negative_over_memory_uri() -> None:
    uri = f"{_mem_root()}/neg.bin"
    with fsspec.open(uri, "wb") as f:
        f.write(b"abcdef")
    reader = ByteRangeReader(uri)
    with pytest.raises(ValueError):
        reader.read(-1, 3)
    with pytest.raises(ValueError):
        reader.read(0, -1)
