from __future__ import annotations

import logging
import pickle
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import cast

import torch

from weightlens.contracts import CheckpointValidator
from weightlens.models import CheckpointHealth

logger = logging.getLogger(__name__)

_STATE_DICT_KEYS = ("model", "state_dict", "module")
"""Common keys that wrap the actual state dict in training checkpoints."""


def _extract_state_dict(checkpoint: Mapping[str, object]) -> Mapping[str, object]:
    """Unwrap a nested training checkpoint to the tensor mapping.

    If the top-level mapping contains a known wrapper key (e.g. ``model``,
    ``state_dict``), return the inner dict.  Otherwise return as-is.
    """
    for key in _STATE_DICT_KEYS:
        inner = checkpoint.get(key)
        if isinstance(inner, Mapping):
            logger.info("Extracting nested state dict from key %r.", key)
            return cast(Mapping[str, object], inner)
    return checkpoint


def _load_checkpoint(path: Path) -> object:
    """Load a checkpoint with safe fallback.

    Tries ``weights_only=True`` first.  If that fails with an
    ``UnpicklingError`` (e.g. numpy scalar globals), retries with
    ``weights_only=False`` and logs a warning.
    """
    try:
        return torch.load(
            path, weights_only=True, map_location="cpu", mmap=True
        )
    except pickle.UnpicklingError:
        logger.warning(
            "weights_only=True failed for %s, retrying with "
            "weights_only=False.",
            path,
        )
        return torch.load(
            path, weights_only=False, map_location="cpu", mmap=True
        )


class PyTorchCheckpointValidator(CheckpointValidator):
    """Validate PyTorch checkpoints without batch loading."""

    _CHUNK_SIZE = 1_000_000

    def __init__(self, checkpoint_path: str | Path) -> None:
        self._checkpoint_path = Path(checkpoint_path)

    def validate(self) -> CheckpointHealth:
        file_size_bytes = self._checkpoint_path.stat().st_size
        corruption_flags: list[str] = []
        logger.info(
            "Starting validation for checkpoint %s (size=%d bytes).",
            self._checkpoint_path,
            file_size_bytes,
        )

        if file_size_bytes == 0:
            corruption_flags.append("empty_file")
            logger.debug("Flagged empty_file for %s.", self._checkpoint_path)
            health = CheckpointHealth(
                file_size_bytes=file_size_bytes,
                is_empty=True,
                loadable=False,
                tensor_count=0,
                total_params=0,
                corruption_flags=corruption_flags,
            )
            logger.info(
                "Validation completed for %s: %s.",
                self._checkpoint_path,
                health.model_dump(),
            )
            return health

        try:
            checkpoint = _load_checkpoint(self._checkpoint_path)
        except Exception:
            logger.exception("Failed to load checkpoint %s.", self._checkpoint_path)
            corruption_flags.append("load_failed")
            health = CheckpointHealth(
                file_size_bytes=file_size_bytes,
                is_empty=False,
                loadable=False,
                tensor_count=0,
                total_params=0,
                corruption_flags=corruption_flags,
            )
            logger.info(
                "Validation completed for %s: %s.",
                self._checkpoint_path,
                health.model_dump(),
            )
            return health

        if not isinstance(checkpoint, Mapping):
            logger.error("Checkpoint %s is not a mapping.", self._checkpoint_path)
            corruption_flags.append("not_mapping")
            health = CheckpointHealth(
                file_size_bytes=file_size_bytes,
                is_empty=False,
                loadable=False,
                tensor_count=0,
                total_params=0,
                corruption_flags=corruption_flags,
            )
            logger.info(
                "Validation completed for %s: %s.",
                self._checkpoint_path,
                health.model_dump(),
            )
            return health

        typed_checkpoint = _extract_state_dict(
            cast(Mapping[str, object], checkpoint)
        )
        tensor_count = 0
        total_params = 0

        for name, value in typed_checkpoint.items():
            if not isinstance(value, torch.Tensor):
                flag = f"non_tensor:{name}"
                corruption_flags.append(flag)
                logger.debug("Flagged %s.", flag)
                continue

            tensor_count += 1

            try:
                numel = int(value.numel())
            except Exception:
                flag = f"tensor_access_failed:{name}"
                corruption_flags.append(flag)
                logger.exception("Tensor access failed for %s.", name)
                continue

            total_params += numel
            logger.debug(
                "Counted tensor %s (numel=%d). totals: tensors=%d params=%d.",
                name,
                numel,
                tensor_count,
                total_params,
            )

            if numel == 0:
                flag = f"empty_tensor:{name}"
                corruption_flags.append(flag)
                logger.debug("Flagged %s.", flag)
                continue

            try:
                self._check_tensor(name, value, corruption_flags)
            except Exception:
                flag = f"tensor_access_failed:{name}"
                corruption_flags.append(flag)
                logger.exception("Tensor access failed for %s.", name)

        is_empty = tensor_count == 0

        health = CheckpointHealth(
            file_size_bytes=file_size_bytes,
            is_empty=is_empty,
            loadable=True,
            tensor_count=tensor_count,
            total_params=total_params,
            corruption_flags=corruption_flags,
        )
        logger.info(
            "Validation completed for %s: %s.",
            self._checkpoint_path,
            health.model_dump(),
        )
        return health

    @classmethod
    def _iter_chunks(cls, tensor: torch.Tensor) -> Iterator[torch.Tensor]:
        flat = tensor.reshape(-1)
        for start in range(0, flat.numel(), cls._CHUNK_SIZE):
            yield flat[start : start + cls._CHUNK_SIZE]

    @classmethod
    def _is_nan_flood(cls, tensor: torch.Tensor) -> bool:
        if not (tensor.is_floating_point() or tensor.is_complex()):
            return False

        for chunk in cls._iter_chunks(tensor):
            nan_mask = torch.isnan(chunk)
            if not nan_mask.all():
                return False
        return True

    @classmethod
    def _is_zero_flood(cls, tensor: torch.Tensor) -> bool:
        for chunk in cls._iter_chunks(tensor):
            if torch.count_nonzero(chunk).item() != 0:
                return False
        return True

    @classmethod
    def _check_tensor(
        cls, name: str, tensor: torch.Tensor, corruption_flags: list[str]
    ) -> None:
        data = tensor.detach()
        if data.device.type != "cpu":
            data = data.cpu()

        if cls._is_nan_flood(data):
            flag = f"nan_flood:{name}"
            corruption_flags.append(flag)
            logger.debug("Flagged %s.", flag)

        if cls._is_zero_flood(data):
            flag = f"zero_flood:{name}"
            corruption_flags.append(flag)
            logger.debug("Flagged %s.", flag)
