from __future__ import annotations

import numpy as np

from weightlens.contracts import StatsEngine
from weightlens.models import LayerStats, LayerTensor


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

        return LayerStats(
            name=layer.name,
            mean=mean,
            std=std,
            min=min_value,
            max=max_value,
            l2_norm=l2_norm,
            sparsity=sparsity,
            param_count=param_count,
        )
