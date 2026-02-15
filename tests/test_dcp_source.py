from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

from weightlens.sources.dcp import DCPWeightSource


def _save_dcp(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    """Save a DCP checkpoint using ``dcp.save(no_dist=True)``."""
    path.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dcp_save(state_dict, checkpoint_id=str(path), no_dist=True)


# ------------------------------------------------------------------
# Happy path
# ------------------------------------------------------------------


def test_streams_correct_names_and_shapes(tmp_path: Path) -> None:
    state = {
        "model.attn.weight": torch.randn(64, 64),
        "model.mlp.weight": torch.randn(128, 64),
        "model.norm.weight": torch.ones(64),
    }
    _save_dcp(tmp_path / "ckpt", state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layers = list(DCPWeightSource(tmp_path / "ckpt").iter_layers())

    names = {layer.name for layer in layers}
    assert names == {"model.attn.weight", "model.mlp.weight", "model.norm.weight"}

    by_name = {layer.name: layer for layer in layers}
    assert by_name["model.attn.weight"].shape == (64, 64)
    assert by_name["model.mlp.weight"].shape == (128, 64)
    assert by_name["model.norm.weight"].shape == (64,)


def test_values_match_original(tmp_path: Path) -> None:
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    state = {"layer": original}
    _save_dcp(tmp_path / "ckpt", state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layers = list(DCPWeightSource(tmp_path / "ckpt").iter_layers())

    assert len(layers) == 1
    np.testing.assert_allclose(layers[0].values, original.numpy().ravel(), rtol=1e-6)


def test_dtype_is_float32_string(tmp_path: Path) -> None:
    state = {"w": torch.randn(4, 4, dtype=torch.float32)}
    _save_dcp(tmp_path / "ckpt", state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layers = list(DCPWeightSource(tmp_path / "ckpt").iter_layers())

    assert layers[0].dtype == "float32"


# ------------------------------------------------------------------
# Non-float skipping
# ------------------------------------------------------------------


def test_skips_non_float_tensors(tmp_path: Path) -> None:
    state = {
        "float_layer": torch.randn(8, 8),
        "int_step": torch.tensor(42, dtype=torch.int32),
        "long_counter": torch.tensor(100, dtype=torch.int64),
    }
    _save_dcp(tmp_path / "ckpt", state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layers = list(DCPWeightSource(tmp_path / "ckpt").iter_layers())

    assert len(layers) == 1
    assert layers[0].name == "float_layer"


# ------------------------------------------------------------------
# Metadata naming conventions
# ------------------------------------------------------------------


def test_handles_dot_metadata(tmp_path: Path) -> None:
    """Standard PyTorch DCP uses ``.metadata``."""
    state = {"w": torch.randn(4, 4)}
    _save_dcp(tmp_path / "ckpt", state)
    # dcp.save creates .metadata by default
    assert (tmp_path / "ckpt" / ".metadata").exists()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layers = list(DCPWeightSource(tmp_path / "ckpt").iter_layers())

    assert len(layers) == 1


def test_handles_no_dot_metadata(tmp_path: Path) -> None:
    """Megatron-LM uses ``metadata`` (no dot prefix)."""
    state = {"w": torch.randn(4, 4)}
    _save_dcp(tmp_path / "ckpt", state)
    # Rename .metadata → metadata
    (tmp_path / "ckpt" / ".metadata").rename(tmp_path / "ckpt" / "metadata")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layers = list(DCPWeightSource(tmp_path / "ckpt").iter_layers())

    assert len(layers) == 1


# ------------------------------------------------------------------
# Error cases
# ------------------------------------------------------------------


def test_raises_on_missing_directory() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(FileNotFoundError):
            list(DCPWeightSource("/nonexistent/path").iter_layers())


def test_raises_on_missing_metadata(tmp_path: Path) -> None:
    state = {"w": torch.randn(4, 4)}
    _save_dcp(tmp_path / "ckpt", state)
    # Remove metadata file
    (tmp_path / "ckpt" / ".metadata").unlink()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(FileNotFoundError):
            list(DCPWeightSource(tmp_path / "ckpt").iter_layers())


# ------------------------------------------------------------------
# Empty checkpoint
# ------------------------------------------------------------------


def test_empty_checkpoint_yields_nothing(tmp_path: Path) -> None:
    """A checkpoint with only non-tensor entries yields no layers."""
    state = {"step": torch.tensor(0, dtype=torch.int64)}
    _save_dcp(tmp_path / "ckpt", state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layers = list(DCPWeightSource(tmp_path / "ckpt").iter_layers())

    assert layers == []
