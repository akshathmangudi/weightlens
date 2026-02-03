from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class FixedRangeHistogramQuantiles:
    """Approximate quantiles using a fixed-range histogram with overflow bins."""

    min_value: float
    max_value: float
    bins: int = 2048
    _counts: NDArray[np.float64] = field(init=False)
    _underflow: int = 0
    _overflow: int = 0
    _total: int = 0

    def __post_init__(self) -> None:
        if self.max_value <= self.min_value:
            raise ValueError("max_value must be greater than min_value.")
        if self.bins <= 0:
            raise ValueError("bins must be positive.")
        self._counts = np.zeros(self.bins, dtype=np.float64)

    def update(self, values: NDArray[np.number]) -> None:
        if values.size == 0:
            return

        self._underflow += int(np.sum(values < self.min_value))
        self._overflow += int(np.sum(values > self.max_value))

        counts, _ = np.histogram(
            values,
            bins=self.bins,
            range=(self.min_value, self.max_value),
        )
        self._counts += counts.astype(np.float64, copy=False)
        self._total += int(values.size)

    def quantile(self, q: float) -> float:
        if self._total == 0:
            raise ValueError("No samples provided for quantile estimation.")
        if not 0.0 < q < 1.0:
            raise ValueError("q must be in (0, 1).")

        target = q * self._total
        if target <= self._underflow:
            return self.min_value
        if target >= self._total - self._overflow:
            return self.max_value

        cumulative = np.cumsum(self._counts)
        idx = int(np.searchsorted(cumulative, target - self._underflow, side="right"))
        idx = min(idx, self.bins - 1)

        prev = cumulative[idx - 1] if idx > 0 else 0.0
        bin_count = self._counts[idx]
        frac = 0.0 if bin_count == 0.0 else (target - self._underflow - prev) / bin_count
        width = (self.max_value - self.min_value) / self.bins
        return self.min_value + (idx + frac) * width
