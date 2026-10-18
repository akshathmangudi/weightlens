from __future__ import annotations

import math
import sys

import numpy as np
from numpy.typing import NDArray


class RelativeAbsQuantiles:
    """Estimate quantiles of absolute values with bounded relative value error.

    Positive values use logarithmic buckets. A bucket representative differs
    from every value in that bucket by at most ``relative_error``. Zeros are
    counted exactly. The value-bucket approach follows DDSketch (Masson et al.,
    2019), while the two adjacent order statistics preserve NumPy's linear
    quantile convention.
    """

    def __init__(self, relative_error: float = 0.005) -> None:
        if not 0.0 < relative_error < 1.0:
            raise ValueError("relative_error must be in (0, 1).")
        self._log_gamma = math.log1p(relative_error) - math.log1p(-relative_error)
        self._gamma = (1.0 + relative_error) / (1.0 - relative_error)
        self._counts: dict[int, int] = {}
        self._zeros = 0
        self._total = 0

    def update(self, values: NDArray[np.float64]) -> None:
        magnitudes = np.abs(values)
        positive = magnitudes[magnitudes > 0.0]
        self._zeros += int(magnitudes.size - positive.size)
        self._total += int(magnitudes.size)
        if positive.size == 0:
            return

        indices = np.floor(np.log(positive) / self._log_gamma).astype(np.int64)
        offset = int(indices.min())
        counts = np.bincount(indices - offset)
        for index in np.flatnonzero(counts):
            key = offset + int(index)
            self._counts[key] = self._counts.get(key, 0) + int(counts[index])

    def quantile(self, q: float) -> float:
        if self._total == 0:
            raise ValueError("No samples provided for quantile estimation.")
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1].")

        position = q * (self._total - 1)
        lower_rank = math.floor(position)
        upper_rank = math.ceil(position)
        lower = self._value_at_rank(lower_rank)
        upper = self._value_at_rank(upper_rank)
        return lower + (position - lower_rank) * (upper - lower)

    def _value_at_rank(self, rank: int) -> float:
        if rank < self._zeros:
            return 0.0

        cumulative = self._zeros
        for key in sorted(self._counts):
            cumulative += self._counts[key]
            if rank < cumulative:
                lower = math.exp(key * self._log_gamma)
                estimate = lower * (2.0 * self._gamma / (1.0 + self._gamma))
                return min(estimate, sys.float_info.max)
        raise RuntimeError("Quantile counts do not match the sample count.")
