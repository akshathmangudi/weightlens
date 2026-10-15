from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weightlens.models import LayerTensor, TensorOmission


class WeightSource(ABC):
    """Stream LayerTensor objects one at a time."""

    @abstractmethod
    def iter_layers(self) -> Iterator[LayerTensor]:
        """Yield LayerTensor objects in a streaming fashion."""
        raise NotImplementedError

    def iter_events(self) -> Iterator[LayerTensor | TensorOmission]:
        """Yield tensors and source exclusions without another data pass."""
        yield from self.iter_layers()
