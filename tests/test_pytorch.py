from __future__ import annotations

import inspect
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

from weightlens.sources.pytorch import PyTorchWeightSource


def _save_checkpoint(tmp_path: Path) -> Path:
    state: OrderedDict[str, torch.Tensor] = OrderedDict(
        layer1=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        layer2=torch.tensor([5.0, 6.0], dtype=torch.float64),
        int_layer=torch.tensor([1, 2, 3], dtype=torch.int64),
    )
    checkpoint_path = tmp_path / "model.pth"
    torch.save(state, checkpoint_path)
    return checkpoint_path


def test_pytorch_weight_source_streams_layers_one_at_a_time(
    tmp_path: Path,
) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)
    source = PyTorchWeightSource(checkpoint_path)

    layer_iter = source.iter_layers()
    assert inspect.isgenerator(layer_iter)

    first = next(layer_iter)
    assert first.name == "layer1"

    second = next(layer_iter)
    assert second.name == "layer2"

    with pytest.raises(StopIteration):
        next(layer_iter)


def test_pytorch_weight_source_extracts_numeric_values(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)
    source = PyTorchWeightSource(checkpoint_path)

    first = next(source.iter_layers())
    expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    np.testing.assert_allclose(first.values, expected)
    assert isinstance(first.values, np.ndarray)


def test_pytorch_weight_source_reports_shape_and_dtype(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)
    source = PyTorchWeightSource(checkpoint_path)

    first = next(source.iter_layers())

    assert first.shape == (2, 2)
    assert first.dtype == "float32"


def test_pytorch_weight_source_skips_non_float_tensors(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)
    source = PyTorchWeightSource(checkpoint_path)

    names = [layer.name for layer in source.iter_layers()]

    assert names == ["layer1", "layer2"]
