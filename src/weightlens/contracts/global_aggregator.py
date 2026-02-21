from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from weightlens.models import GlobalStats


class GlobalAggregator(ABC):
    """Stream global metrics from tensor values."""

    @abstractmethod
    def update(self, values: NDArray[np.number]) -> None:
        """Consume numeric values in a streaming fashion."""
        raise NotImplementedError

    def update_from_summary(
        self,
        values: NDArray[np.number],
        *,
        count: int,
        mean: float,
        variance: float,
    ) -> None:
        """Consume values with pre-computed summary statistics.

        Implementations may use *count*, *mean* and *variance* to skip
        redundant recomputation.  The default falls back to :meth:`update`.
        """
        self.update(values)

    @abstractmethod
    def finalize(self) -> GlobalStats:
        """Return global statistics computed from all updates."""
        raise NotImplementedError
