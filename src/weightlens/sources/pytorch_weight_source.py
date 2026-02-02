from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import cast

import torch

from weightlens.contracts import WeightSource
from weightlens.models import LayerTensor


class PyTorchWeightSource(WeightSource):
    """Stream PyTorch checkpoints as LayerTensor objects."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self._checkpoint_path = Path(checkpoint_path)

    def iter_layers(self) -> Iterator[LayerTensor]:
        checkpoint = torch.load(
            self._checkpoint_path,
            weights_only=True,
            map_location="cpu",
            mmap=True,
        )

        if not isinstance(checkpoint, Mapping):
            raise TypeError("Expected checkpoint to be a mapping of tensors.")

        typed_checkpoint = cast(Mapping[str, object], checkpoint)

        for name, tensor in typed_checkpoint.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if not tensor.is_floating_point():
                continue

            materialized = tensor.detach().cpu().contiguous()
            values = materialized.reshape(-1).numpy()

            yield LayerTensor(
                name=str(name),
                values=values,
                shape=tuple(tensor.shape),
                dtype=str(values.dtype),
            )
