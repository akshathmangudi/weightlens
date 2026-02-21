from __future__ import annotations

import logging
import math

import numpy as np

from weightlens.contracts import StatsEngine
from weightlens.models import LayerStats, LayerTensor

logger = logging.getLogger(__name__)


class BasicStatsEngine(StatsEngine):
    """Compute basic descriptive statistics for a layer."""

    def compute_layer(self, layer: LayerTensor) -> LayerStats:
        values = layer.values
        param_count = int(values.size)
        if param_count == 0:
            logger.error("Layer %s is empty.", layer.name)
            raise ValueError(f"Layer {layer.name} is empty.")

        # Fused: derive mean, std, l2_norm from sum + dot (2 passes)
        total = float(np.sum(values))  # pass 1
        mean = total / param_count
        if not math.isfinite(mean):
            logger.error("Layer %s contains NaN or Inf values.", layer.name)
            raise ValueError(f"Layer {layer.name} contains NaN values.")

        sum_sq = float(np.dot(values, values))  # pass 2
        variance = max(sum_sq / param_count - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        l2_norm = float(np.sqrt(sum_sq))  # free (reuses sum_sq)
        min_value = float(np.min(values))  # pass 3
        max_value = float(np.max(values))  # pass 4
        nonzero_count = int(np.count_nonzero(values))  # pass 5
        sparsity = 1.0 - (nonzero_count / param_count)
        p99_abs = self._compute_p99_abs(values)  # pass 6+7
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
        )

    @staticmethod
    def _compute_p99_abs(values: np.ndarray) -> float:
        return float(np.quantile(np.abs(values), 0.99, method="linear"))
