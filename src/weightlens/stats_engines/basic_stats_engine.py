from __future__ import annotations

import numpy as np

from weightlens.contracts import StatsEngine
from weightlens.models import LayerStats, LayerTensor
from weightlens.p2_quantile import P2QuantileEstimator


class BasicStatsEngine(StatsEngine):
    """Compute basic descriptive statistics for a layer."""

    def compute_layer(self, layer: LayerTensor) -> LayerStats:
        values = layer.values
        param_count = int(values.size)
        if param_count == 0:
            raise ValueError(f"Layer {layer.name} is empty.")

        mean = float(np.mean(values))
        if np.isnan(mean):
            raise ValueError(f"Layer {layer.name} contains NaN values.")

        std = float(np.std(values, ddof=0))
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        l2_norm = float(np.sqrt(np.dot(values, values)))
        nonzero_count = int(np.count_nonzero(values))
        sparsity = 1.0 - (nonzero_count / param_count)
        p99_abs = self._compute_p99_abs(values)

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
        estimator = P2QuantileEstimator(0.99)
        for entry in values.flat:
            estimator.update(abs(float(entry)))
        return estimator.value()
