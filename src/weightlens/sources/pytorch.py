from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import cast

import torch

from weightlens.contracts import WeightSource
from weightlens.models import LayerTensor, TensorOmission
from weightlens.tensor_utils import tensor_to_numpy
from weightlens.validators.pytorch_checkpoint import (
    _extract_state_dict,
    _load_checkpoint,
)

logger = logging.getLogger(__name__)


class PyTorchWeightSource(WeightSource):
    """Stream PyTorch checkpoints as LayerTensor objects."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self._checkpoint_path = Path(checkpoint_path)

    def iter_layers(self) -> Iterator[LayerTensor]:
        for event in self.iter_events():
            if isinstance(event, LayerTensor):
                yield event

    def iter_events(self) -> Iterator[LayerTensor | TensorOmission]:
        checkpoint = _load_checkpoint(self._checkpoint_path)

        if not isinstance(checkpoint, Mapping):
            logger.error(
                "Checkpoint %s is not a mapping of tensors.", self._checkpoint_path
            )
            raise TypeError("Expected checkpoint to be a mapping of tensors.")

        typed_checkpoint = _extract_state_dict(
            cast(Mapping[str, object], checkpoint)
        )
        logger.info("Starting streaming for checkpoint %s.", self._checkpoint_path)
        layer_count = 0

        try:
            for name, tensor in typed_checkpoint.items():
                if not isinstance(tensor, torch.Tensor):
                    logger.debug("Skipping non-tensor entry %s.", name)
                    continue
                if not tensor.is_floating_point():
                    unsupported = tensor.is_complex()
                    logger.debug(
                        "Skipping non-float tensor %s (dtype=%s).",
                        name,
                        tensor.dtype,
                    )
                    yield TensorOmission(
                        name=str(name),
                        kind="skipped" if unsupported else "excluded",
                        reason=(
                            "unsupported complex dtype" if unsupported
                            else "non-floating tensor"
                        ),
                        dtype=str(tensor.dtype),
                    )
                    continue

                values = tensor_to_numpy(tensor)
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
