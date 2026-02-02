from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch

from weightlens.validators.pytorch_checkpoint_validator import (
    PyTorchCheckpointValidator,
)


def _save_checkpoint(tmp_path: Path) -> Path:
    state: OrderedDict[str, torch.Tensor] = OrderedDict(
        layer1=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        layer2=torch.tensor([5.0, 6.0], dtype=torch.float64),
    )
    checkpoint_path = tmp_path / "model.pth"
    torch.save(state, checkpoint_path)
    return checkpoint_path


def test_validator_detects_zero_byte_file(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "empty.pth"
    checkpoint_path.write_bytes(b"")

    validator = PyTorchCheckpointValidator(checkpoint_path)
    health = validator.validate()

    assert health.file_size_bytes == 0
    assert health.is_empty is True
    assert health.loadable is False
    assert health.tensor_count == 0
    assert health.total_params == 0
    assert "empty_file" in health.corruption_flags


def test_validator_flags_load_failure_for_corrupt_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)
    data = checkpoint_path.read_bytes()

    corrupt_path = tmp_path / "corrupt.pth"
    corrupt_path.write_bytes(data[: max(1, len(data) // 2)])

    validator = PyTorchCheckpointValidator(corrupt_path)
    health = validator.validate()

    assert health.loadable is False
    assert "load_failed" in health.corruption_flags


def test_validator_flags_non_tensor_entries(tmp_path: Path) -> None:
    state: OrderedDict[str, object] = OrderedDict(
        layer1=torch.tensor([1.0], dtype=torch.float32),
        meta="not-a-tensor",
    )
    checkpoint_path = tmp_path / "mixed.pth"
    torch.save(state, checkpoint_path)

    validator = PyTorchCheckpointValidator(checkpoint_path)
    health = validator.validate()

    assert health.loadable is True
    assert "non_tensor:meta" in health.corruption_flags
    assert health.tensor_count == 1
    assert health.total_params == 1


def test_validator_flags_empty_tensor(tmp_path: Path) -> None:
    state: OrderedDict[str, torch.Tensor] = OrderedDict(
        empty=torch.empty(0, dtype=torch.float32),
        ok=torch.tensor([1.0], dtype=torch.float32),
    )
    checkpoint_path = tmp_path / "empty_tensor.pth"
    torch.save(state, checkpoint_path)

    validator = PyTorchCheckpointValidator(checkpoint_path)
    health = validator.validate()

    assert "empty_tensor:empty" in health.corruption_flags
    assert health.tensor_count == 2
    assert health.total_params == 1


def test_validator_flags_nan_and_zero_floods(tmp_path: Path) -> None:
    state: OrderedDict[str, torch.Tensor] = OrderedDict(
        nan_layer=torch.tensor([float("nan"), float("nan")], dtype=torch.float32),
        zero_layer=torch.zeros(3, dtype=torch.float32),
    )
    checkpoint_path = tmp_path / "floods.pth"
    torch.save(state, checkpoint_path)

    validator = PyTorchCheckpointValidator(checkpoint_path)
    health = validator.validate()

    assert "nan_flood:nan_layer" in health.corruption_flags
    assert "zero_flood:zero_layer" in health.corruption_flags


def test_validator_counts_tensors_and_params(tmp_path: Path) -> None:
    state: OrderedDict[str, torch.Tensor] = OrderedDict(
        ints=torch.tensor([1, 2, 3], dtype=torch.int64),
        floats=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )
    checkpoint_path = tmp_path / "counts.pth"
    torch.save(state, checkpoint_path)

    validator = PyTorchCheckpointValidator(checkpoint_path)
    health = validator.validate()

    assert health.tensor_count == 2
    assert health.total_params == 5
