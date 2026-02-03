from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from weightlens.aggregators.streaming_layer_metrics import (
    StreamingLayerMetricsAggregator,
)
from weightlens.contracts import GlobalAggregator
from weightlens.models import GlobalStats, LayerStats
from weightlens.p2_quantile import P2QuantileEstimator


class StreamingGlobalAggregator(GlobalAggregator):
    """Streaming global metrics via Welford + P² estimators."""

    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._estimators = {
            "p1": P2QuantileEstimator(0.01),
            "p5": P2QuantileEstimator(0.05),
            "p50": P2QuantileEstimator(0.5),
            "p95": P2QuantileEstimator(0.95),
            "p99": P2QuantileEstimator(0.99),
        }
        self._layer_metrics = StreamingLayerMetricsAggregator()

    def update(self, values: NDArray[np.number]) -> None:
        for entry in values.flat:
            value = float(entry)
            if not math.isfinite(value):
                raise ValueError("Non-finite value encountered in global aggregation.")
            self._update_value(value)

    def update_layer_stats(self, layer_stats: LayerStats) -> None:
        self._layer_metrics.update(layer_stats)

    def finalize(self) -> GlobalStats:
        if self._count == 0:
            raise ValueError("No values provided for global aggregation.")
        variance = self._m2 / self._count
        layer_metrics = self._layer_metrics.finalize()
        return GlobalStats(
            mean=self._mean,
            std=math.sqrt(variance),
            p1=self._estimators["p1"].value(),
            p5=self._estimators["p5"].value(),
            p50=self._estimators["p50"].value(),
            p95=self._estimators["p95"].value(),
            p99=self._estimators["p99"].value(),
            median_layer_variance=layer_metrics.median_layer_variance,
            median_layer_norm=layer_metrics.median_layer_norm,
            iqr_layer_norm=layer_metrics.iqr_layer_norm,
        )

    def _update_value(self, value: float) -> None:
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2
        for estimator in self._estimators.values():
            estimator.update(value)
