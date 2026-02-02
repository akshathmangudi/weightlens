from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weightlens.models import LayerTensor


class WeightSource(ABC):
    """Stream LayerTensor objects one at a time."""

    @abstractmethod
    def iter_layers(self) -> Iterator[LayerTensor]:
        """Yield LayerTensor objects in a streaming fashion."""
        raise NotImplementedError
