from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import cast

import torch

from weightlens.contracts import CheckpointValidator
from weightlens.models import CheckpointHealth


class PyTorchCheckpointValidator(CheckpointValidator):
    """Validate PyTorch checkpoints without batch loading."""

    _CHUNK_SIZE = 1_000_000

    def __init__(self, checkpoint_path: str | Path) -> None:
        self._checkpoint_path = Path(checkpoint_path)

    def validate(self) -> CheckpointHealth:
        file_size_bytes = self._checkpoint_path.stat().st_size
        corruption_flags: list[str] = []

        if file_size_bytes == 0:
            return CheckpointHealth(
                file_size_bytes=file_size_bytes,
                is_empty=True,
                loadable=False,
                tensor_count=0,
                total_params=0,
                corruption_flags=["empty_file"],
            )

        try:
            checkpoint = torch.load(
                self._checkpoint_path,
                weights_only=True,
                map_location="cpu",
                mmap=True,
            )
        except Exception:
            corruption_flags.append("load_failed")
            return CheckpointHealth(
                file_size_bytes=file_size_bytes,
                is_empty=False,
                loadable=False,
                tensor_count=0,
                total_params=0,
                corruption_flags=corruption_flags,
            )

        if not isinstance(checkpoint, Mapping):
            corruption_flags.append("not_mapping")
            return CheckpointHealth(
                file_size_bytes=file_size_bytes,
                is_empty=False,
                loadable=False,
                tensor_count=0,
                total_params=0,
                corruption_flags=corruption_flags,
            )

        typed_checkpoint = cast(Mapping[str, object], checkpoint)
        tensor_count = 0
        total_params = 0

        for name, value in typed_checkpoint.items():
            if not isinstance(value, torch.Tensor):
                corruption_flags.append(f"non_tensor:{name}")
                continue

            tensor_count += 1

            try:
                numel = int(value.numel())
            except Exception:
                corruption_flags.append(f"tensor_access_failed:{name}")
                continue

            total_params += numel

            if numel == 0:
                corruption_flags.append(f"empty_tensor:{name}")
                continue

            try:
                self._check_tensor(name, value, corruption_flags)
            except Exception:
                corruption_flags.append(f"tensor_access_failed:{name}")

        is_empty = tensor_count == 0

        return CheckpointHealth(
            file_size_bytes=file_size_bytes,
            is_empty=is_empty,
            loadable=True,
            tensor_count=tensor_count,
            total_params=total_params,
            corruption_flags=corruption_flags,
        )

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
            corruption_flags.append(f"nan_flood:{name}")

        if cls._is_zero_flood(data):
            corruption_flags.append(f"zero_flood:{name}")
