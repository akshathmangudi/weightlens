from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

from weightlens.validators.dcp_checkpoint import DCPCheckpointValidator


def _save_dcp(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dcp_save(state_dict, checkpoint_id=str(path), no_dist=True)


# ------------------------------------------------------------------
# Healthy checkpoint
# ------------------------------------------------------------------


def test_healthy_checkpoint(tmp_path: Path) -> None:
    state = {
        "model.weight": torch.randn(32, 32),
        "model.bias": torch.randn(32),
    }
    _save_dcp(tmp_path / "ckpt", state)

    health = DCPCheckpointValidator(tmp_path / "ckpt").validate()

    assert health.loadable is True
    assert health.is_empty is False
    assert health.tensor_count == 2
    assert health.total_params == 32 * 32 + 32
    assert health.corruption_flags == []
    assert health.file_size_bytes > 0


# ------------------------------------------------------------------
# Directory / metadata errors
# ------------------------------------------------------------------


def test_not_a_directory(tmp_path: Path) -> None:
    path = tmp_path / "not_a_dir"
    health = DCPCheckpointValidator(path).validate()

    assert health.loadable is False
    assert "not_a_directory" in health.corruption_flags


def test_empty_directory(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.mkdir()

    health = DCPCheckpointValidator(path).validate()

    assert health.loadable is False
    assert "empty_directory" in health.corruption_flags


def test_missing_metadata(tmp_path: Path) -> None:
    path = tmp_path / "ckpt"
    path.mkdir()
    # Write a dummy shard file so directory isn't empty
    (path / "__0_0.distcp").write_bytes(b"fake data")

    health = DCPCheckpointValidator(path).validate()

    assert health.loadable is False
    assert "missing_metadata_file" in health.corruption_flags


def test_empty_metadata_file(tmp_path: Path) -> None:
    state = {"w": torch.randn(4, 4)}
    _save_dcp(tmp_path / "ckpt", state)
    # Truncate metadata to zero bytes
    (tmp_path / "ckpt" / ".metadata").write_bytes(b"")

    health = DCPCheckpointValidator(tmp_path / "ckpt").validate()

    assert health.loadable is False
    assert "empty_metadata_file" in health.corruption_flags


# ------------------------------------------------------------------
# Shard file issues
# ------------------------------------------------------------------


def test_zero_byte_shard(tmp_path: Path) -> None:
    state = {"w": torch.randn(4, 4)}
    _save_dcp(tmp_path / "ckpt", state)
    # Create an additional zero-byte shard
    (tmp_path / "ckpt" / "__1_0.distcp").write_bytes(b"")

    health = DCPCheckpointValidator(tmp_path / "ckpt").validate()

    assert health.loadable is True  # Extra zero-byte shard doesn't block loading
    assert any(f.startswith("zero_byte_shard:") for f in health.corruption_flags)


def test_missing_referenced_shard(tmp_path: Path) -> None:
    state = {"w": torch.randn(4, 4)}
    _save_dcp(tmp_path / "ckpt", state)
    # Remove the shard file that metadata references
    (tmp_path / "ckpt" / "__0_0.distcp").unlink()

    health = DCPCheckpointValidator(tmp_path / "ckpt").validate()

    assert health.loadable is False
    assert any(f.startswith("missing_shard:") for f in health.corruption_flags)


# ------------------------------------------------------------------
# Tensor counting
# ------------------------------------------------------------------


def test_tensor_counting_accuracy(tmp_path: Path) -> None:
    state = {
        "a": torch.randn(10, 10),
        "b": torch.randn(5),
        "c": torch.tensor(42, dtype=torch.int32),  # non-float, still a tensor in DCP
    }
    _save_dcp(tmp_path / "ckpt", state)

    health = DCPCheckpointValidator(tmp_path / "ckpt").validate()

    assert health.tensor_count == 3  # DCP stores all as TensorStorageMetadata
    assert health.total_params == 100 + 5 + 1


def test_corrupt_metadata_pickle(tmp_path: Path) -> None:
    state = {"w": torch.randn(4, 4)}
    _save_dcp(tmp_path / "ckpt", state)
    # Corrupt the metadata file
    (tmp_path / "ckpt" / ".metadata").write_bytes(b"not a valid pickle")

    health = DCPCheckpointValidator(tmp_path / "ckpt").validate()

    assert health.loadable is False
    assert "metadata_read_failed" in health.corruption_flags
