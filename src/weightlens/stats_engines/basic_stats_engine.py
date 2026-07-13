from __future__ import annotations

import logging
import math

import numpy as np

from weightlens.contracts import StatsEngine
from weightlens.models import LayerStats, LayerTensor

logger = logging.getLogger(__name__)

_HISTOGRAM_BINS = 4096
_HISTOGRAM_MIN = -100.0
_HISTOGRAM_MAX = 100.0


class BasicStatsEngine(StatsEngine):
    """Compute basic descriptive statistics for a layer."""

    def compute_layer(self, layer: LayerTensor) -> LayerStats:
        values = layer.values
        param_count = int(values.size)
        if param_count == 0:
            logger.error("Layer %s is empty.", layer.name)
            raise ValueError(f"Layer {layer.name} is empty.")

        total = float(np.sum(values, dtype=np.float64))
        mean = total / param_count
        if not math.isfinite(mean):
            logger.error("Layer %s contains NaN or Inf values.", layer.name)
            raise ValueError(f"Layer {layer.name} contains NaN values.")

        variance = float(np.var(values, ddof=0, dtype=np.float64))
        variance = max(0.0, variance)
        sum_sq = (variance + mean**2) * param_count
        std = float(np.sqrt(variance))
        l2_norm = float(np.sqrt(sum_sq))
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        nonzero_count = int(np.count_nonzero(values))
        sparsity = 1.0 - (nonzero_count / param_count)
        p99_abs = self._compute_p99_abs(values)

        try:
            hist, _ = np.histogram(
                values.ravel(),
                bins=_HISTOGRAM_BINS,
                range=(_HISTOGRAM_MIN, _HISTOGRAM_MAX),
            )
        except ValueError:
            data_min = float(np.min(values))
            data_max = float(np.max(values))
            margin = max(1e-6, (data_max - data_min) * 0.01)
            hist, _ = np.histogram(
                values.ravel().astype(np.float32),
                bins=min(_HISTOGRAM_BINS, param_count),
                range=(data_min - margin, data_max + margin),
            )
        histogram_counts = [float(c) for c in hist]
        histogram_underflow = int(np.sum(values < _HISTOGRAM_MIN))
        histogram_overflow = int(np.sum(values > _HISTOGRAM_MAX))

        logger.debug(
            "Computed stats for %s: mean=%.6f std=%.6f min=%.6f max=%.6f "
            "l2_norm=%.6f sparsity=%.6f p99_abs=%.6f.",
            layer.name,
            mean,
            std,
            min_value,
            max_value,
            l2_norm,
            sparsity,
            p99_abs,
        )

        return LayerStats(
            name=layer.name,
            mean=mean,
            std=std,
            min=min_value,
            max=max_value,
            l2_norm=l2_norm,
            sparsity=sparsity,
            param_count=param_count,
            p99_abs=p99_abs,
            histogram_counts=histogram_counts,
            histogram_underflow=histogram_underflow,
            histogram_overflow=histogram_overflow,
        )

    @staticmethod
    def _compute_p99_abs(values: np.ndarray) -> float:
        # O(n log n) via partition; histogram-based p99 is approximated
        # at fixed range and would clip extreme values. For large layers
        # a P² streaming estimator would be faster — see phase spec.
        return float(np.quantile(np.abs(values), 0.99, method="linear"))
