from __future__ import annotations

import json
import struct
from math import prod
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures_safetensors import write_sharded, write_single
from weightlens.validators.safetensors import SafetensorsCheckpointValidator


def test_validate_single_counts_float_params(tmp_path: Path) -> None:
    tensors: dict[str, np.ndarray] = {
        "w": np.random.randn(4, 3).astype(np.float32),
        "b": np.random.randn(5).astype(np.float32),
        "step": np.array([1], dtype=np.int64),  # excluded from params
    }
    path = str(tmp_path / "m.safetensors")
    write_single(path, tensors)
    health = SafetensorsCheckpointValidator(path).validate()
    assert health.loadable is True
    assert health.is_empty is False
    assert health.tensor_count == 2  # only float tensors
    assert health.total_params == prod((4, 3)) + 5
    assert health.file_size_bytes > 0


def test_validate_sharded_sums_across_shards(tmp_path: Path) -> None:
    a: dict[str, np.ndarray] = {"x": np.random.randn(10).astype(np.float32)}
    b: dict[str, np.ndarray] = {"y": np.random.randn(20).astype(np.float32)}
    index_path = write_sharded(str(tmp_path / "c"), [a, b])
    health = SafetensorsCheckpointValidator(index_path).validate()
    assert health.tensor_count == 2
    assert health.total_params == 30


def test_validate_only_empty_tensors_marks_checkpoint_empty(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "only-empty.safetensors")
    write_single(path, {"empty.weight": np.empty(0, dtype=np.float32)})

    health = SafetensorsCheckpointValidator(path).validate()

    assert health.loadable is True
    assert health.is_empty is True
    assert health.tensor_count == 1
    assert health.total_params == 0
    assert health.corruption_flags == []


def test_validate_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SafetensorsCheckpointValidator(str(tmp_path / "nope.safetensors")).validate()


@pytest.mark.parametrize(
    ("tensors", "data_size"),
    [
        ({"a.weight": (1, 0, 4), "b.weight": (1, 0, 4)}, 4),
        ({"a.weight": (1, 4, 8)}, 8),
        ({"a.weight": (1, 0, 4)}, 8),
        ({"a.weight": (2, 0, 4)}, 4),
        ({"a.weight": (1, -4, 0)}, 4),
    ],
)
def test_invalid_tensor_layout_is_not_loadable(
    tmp_path: Path, tensors: dict[str, tuple[int, int, int]], data_size: int
) -> None:
    header = {
        name: {
            "dtype": "F32",
            "shape": [count],
            "data_offsets": [begin, end],
        }
        for name, (count, begin, end) in tensors.items()
    }
    payload = json.dumps(header).encode("utf-8")
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + bytes(data_size))

    health = SafetensorsCheckpointValidator(str(path)).validate()

    assert health.loadable is False
    assert health.corruption_flags[0].startswith("unreadable:")


def test_sharded_index_must_match_shard_contents(tmp_path: Path) -> None:
    shard = {
        "a.weight": np.array([1.0], dtype=np.float32),
        "b.weight": np.array([2.0], dtype=np.float32),
    }
    index_path = Path(write_sharded(str(tmp_path / "sharded"), [shard]))
    index = json.loads(index_path.read_text())
    del index["weight_map"]["b.weight"]
    index_path.write_text(json.dumps(index))

    health = SafetensorsCheckpointValidator(str(index_path)).validate()

    assert health.loadable is False
    assert "index disagrees" in health.corruption_flags[0]
