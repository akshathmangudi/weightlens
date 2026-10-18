from __future__ import annotations

import math
import sys
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from weightlens.models import SignedQuantileSummary


class SignedLogQuantiles:
    """Estimate signed and absolute quantiles with relative value precision."""

    def __init__(self, relative_error: float = 0.005) -> None:
        if not 0.0 < relative_error < 1.0:
            raise ValueError("relative_error must be in (0, 1).")
        self._log_gamma = math.log1p(relative_error) - math.log1p(-relative_error)
        self._gamma = (1.0 + relative_error) / (1.0 - relative_error)
        self._negative: dict[int, int] = {}
        self._positive: dict[int, int] = {}
        self._zeros = 0
        self._total = 0

    def update(self, values: NDArray[np.number]) -> None:
        flat = values.ravel()
        if not np.isfinite(flat).all():
            raise ValueError("Quantile input contains NaN or Inf.")
        self._total += int(flat.size)
        self._zeros += int(np.count_nonzero(flat == 0))
        nonzero = flat[flat != 0]
        if nonzero.size == 0:
            return
        indices = np.floor(np.log(np.abs(nonzero)) / self._log_gamma).astype(np.int64)
        negative_mask = nonzero < 0
        self._add_counts(indices[negative_mask], self._negative)
        self._add_counts(indices[~negative_mask], self._positive)

    @staticmethod
    def _add_counts(indices: NDArray[np.int64], target: dict[int, int]) -> None:
        if indices.size == 0:
            return
        offset = int(indices.min())
        counts = np.bincount(indices - offset)
        for index in np.flatnonzero(counts):
            key = offset + int(index)
            target[key] = target.get(key, 0) + int(counts[index])

    def summary(self) -> SignedQuantileSummary:
        return SignedQuantileSummary(
            negative_counts=self._negative.copy(),
            positive_counts=self._positive.copy(),
            zeros=self._zeros,
            total=self._total,
        )

    def merge(self, summary: SignedQuantileSummary) -> None:
        counted = (
            summary.zeros
            + sum(summary.negative_counts.values())
            + sum(summary.positive_counts.values())
        )
        counts = (*summary.negative_counts.values(), *summary.positive_counts.values())
        if (
            counted != summary.total
            or summary.zeros < 0
            or summary.total < 0
            or any(count < 0 for count in counts)
        ):
            raise ValueError("Quantile summary counts do not match its total.")
        self._zeros += summary.zeros
        self._total += summary.total
        for source, target in (
            (summary.negative_counts, self._negative),
            (summary.positive_counts, self._positive),
        ):
            for key, count in source.items():
                target[key] = target.get(key, 0) + count

    def quantile(self, q: float) -> float:
        return self._interpolate(q, self._signed_value_at_rank)

    def abs_quantile(self, q: float) -> float:
        return self._interpolate(q, self._absolute_value_at_rank)

    def _interpolate(self, q: float, value_at_rank: Callable[[int], float]) -> float:
        if self._total == 0:
            raise ValueError("No samples provided for quantile estimation.")
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1].")
        position = q * (self._total - 1)
        lower_rank = math.floor(position)
        upper_rank = math.ceil(position)
        lower = value_at_rank(lower_rank)
        upper = value_at_rank(upper_rank)
        return lower + (position - lower_rank) * (upper - lower)

    def _representative(self, key: int) -> float:
        lower = math.exp(key * self._log_gamma)
        estimate = lower * (2.0 * self._gamma / (1.0 + self._gamma))
        return min(estimate, sys.float_info.max)

    def _signed_value_at_rank(self, rank: int) -> float:
        cumulative = 0
        for key in sorted(self._negative, reverse=True):
            cumulative += self._negative[key]
            if rank < cumulative:
                return -self._representative(key)
        cumulative += self._zeros
        if rank < cumulative:
            return 0.0
        for key in sorted(self._positive):
            cumulative += self._positive[key]
            if rank < cumulative:
                return self._representative(key)
        raise RuntimeError("Quantile counts do not match the sample count.")

    def _absolute_value_at_rank(self, rank: int) -> float:
        if rank < self._zeros:
            return 0.0
        cumulative = self._zeros
        for key in sorted(self._negative.keys() | self._positive.keys()):
            cumulative += self._negative.get(key, 0) + self._positive.get(key, 0)
            if rank < cumulative:
                return self._representative(key)
        raise RuntimeError("Quantile counts do not match the sample count.")
