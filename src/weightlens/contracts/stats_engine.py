from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weightlens.models import LayerStats, LayerTensor


class StatsEngine(ABC):
    """Compute statistics for a single tensor layer."""

    @abstractmethod
    def compute_layer(self, layer: LayerTensor) -> LayerStats:
        """Return a LayerStats summary for the layer."""
        raise NotImplementedError
