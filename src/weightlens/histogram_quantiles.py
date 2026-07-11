from __future__ import annotations

import logging
from array import array
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class FixedRangeHistogramQuantiles:
    """Approximate quantiles using a fixed-range histogram with overflow bins."""

    min_value: float
    max_value: float
    bins: int = 2048
    _counts: array[float] = field(init=False)
    _underflow: int = 0
    _overflow: int = 0
    _total: int = 0
    _warned: bool = False

    def __post_init__(self) -> None:
        if self.max_value <= self.min_value:
            raise ValueError("max_value must be greater than min_value.")
        if self.bins <= 0:
            raise ValueError("bins must be positive.")
        self._counts = array("d", [0.0] * self.bins)

    def update(self, values: NDArray[np.number]) -> None:
        if values.size == 0:
            return

        if not np.isfinite(values).all():
            raise ValueError("Histogram input contains NaN or Inf.")

        self._underflow += int(np.sum(values < self.min_value))
        self._overflow += int(np.sum(values > self.max_value))

        counts, _ = np.histogram(
            values,
            bins=self.bins,
            range=(self.min_value, self.max_value),
        )
        counts_view: NDArray[np.float64] = np.frombuffer(
            self._counts, dtype=np.float64
        )
        counts_view += counts
        self._total += int(values.size)

    def merge_histogram(
        self, counts: list[float], underflow: int = 0, overflow: int = 0
    ) -> None:
        counts_view: NDArray[np.float64] = np.frombuffer(
            self._counts, dtype=np.float64
        )
        counts_view += np.array(counts, dtype=np.float64)
        self._underflow += underflow
        self._overflow += overflow
        self._total += int(sum(counts)) + underflow + overflow

    def quantile(self, q: float) -> float:
        if self._total == 0:
            raise ValueError("No samples provided for quantile estimation.")
        if not 0.0 < q < 1.0:
            raise ValueError("q must be in (0, 1).")

        self._check_overflow_ratio()

        target = q * self._total
        if target <= self._underflow:
            return self.min_value
        if target >= self._total - self._overflow:
            return self.max_value

        cumulative = np.cumsum(self._counts, dtype=np.float64)
        idx = int(np.searchsorted(cumulative, target - self._underflow, side="right"))
        idx = min(idx, self.bins - 1)

        prev = float(cumulative[idx - 1]) if idx > 0 else 0.0
        bin_count = float(self._counts[idx])
        frac = (
            0.0
            if bin_count == 0.0
            else (target - self._underflow - prev) / bin_count
        )
        width = (self.max_value - self.min_value) / self.bins
        return self.min_value + (idx + frac) * width

    def _check_overflow_ratio(self) -> None:
        if self._total == 0 or self._warned:
            return
        underflow_pct = (self._underflow / self._total) * 100.0
        overflow_pct = (self._overflow / self._total) * 100.0
        if underflow_pct > 1.0 or overflow_pct > 1.0:
            logger.warning(
                "Histogram range [%.1f, %.1f] may be too narrow: "
                "%.1f%% underflow, %.1f%% overflow of %d total samples.",
                self.min_value,
                self.max_value,
                underflow_pct,
                overflow_pct,
                self._total,
            )
            self._warned = True
