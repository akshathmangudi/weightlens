from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import cast

import torch

from weightlens.contracts import WeightSource
from weightlens.models import LayerTensor

logger = logging.getLogger(__name__)


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
            logger.error(
                "Checkpoint %s is not a mapping of tensors.", self._checkpoint_path
            )
            raise TypeError("Expected checkpoint to be a mapping of tensors.")

        typed_checkpoint = cast(Mapping[str, object], checkpoint)
        logger.info(
            "Starting streaming for checkpoint %s.", self._checkpoint_path
        )
        layer_count = 0

        try:
            for name, tensor in typed_checkpoint.items():
                if not isinstance(tensor, torch.Tensor):
                    logger.debug("Skipping non-tensor entry %s.", name)
                    continue
                if not tensor.is_floating_point():
                    logger.debug(
                        "Skipping non-float tensor %s (dtype=%s).",
                        name,
                        tensor.dtype,
                    )
                    continue

                materialized = tensor.detach().cpu().contiguous()
                values = materialized.reshape(-1).float().numpy()
                layer_count += 1
                logger.debug(
                    "Yielding layer %s shape=%s dtype=%s param_count=%d.",
                    name,
                    tuple(tensor.shape),
                    values.dtype,
                    int(values.size),
                )

                yield LayerTensor(
                    name=str(name),
                    values=values,
                    shape=tuple(tensor.shape),
                    dtype=str(values.dtype),
                )
        finally:
            logger.info(
                "Completed streaming for checkpoint %s (layers=%d).",
                self._checkpoint_path,
                layer_count,
            )
