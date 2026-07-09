from __future__ import annotations

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


def test_validate_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SafetensorsCheckpointValidator(str(tmp_path / "nope.safetensors")).validate()
