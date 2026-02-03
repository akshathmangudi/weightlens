from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from weightlens.models import LayerStats
from weightlens.p2_quantile import P2QuantileEstimator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerMetricsSummary:
    """Streaming summary statistics over per-layer metrics."""

    median_layer_variance: float
    median_layer_norm: float
    iqr_layer_norm: float


class StreamingLayerMetricsAggregator:
    """Aggregate per-layer variance and norm statistics via P² estimators."""

    def __init__(self) -> None:
        self._count = 0
        self._variance_median = P2QuantileEstimator(0.5)
        self._norm_p25 = P2QuantileEstimator(0.25)
        self._norm_p50 = P2QuantileEstimator(0.5)
        self._norm_p75 = P2QuantileEstimator(0.75)

    def update(self, layer_stats: LayerStats) -> None:
        variance = float(layer_stats.std) ** 2
        norm = float(layer_stats.l2_norm)
        if not math.isfinite(variance) or not math.isfinite(norm):
            logger.error(
                "Non-finite layer metrics encountered: layer=%s variance=%s norm=%s.",
                layer_stats.name,
                variance,
                norm,
            )
            raise ValueError("Non-finite layer metrics encountered in aggregation.")

        self._variance_median.update(variance)
        self._norm_p25.update(norm)
        self._norm_p50.update(norm)
        self._norm_p75.update(norm)
        self._count += 1
        logger.debug(
            "Updated layer metrics for %s variance=%.6f norm=%.6f count=%d.",
            layer_stats.name,
            variance,
            norm,
            self._count,
        )

    def finalize(self) -> LayerMetricsSummary:
        if self._count == 0:
            logger.error("Finalize called without any layer stats.")
            raise ValueError("No layer stats provided for layer metrics aggregation.")

        median_variance = self._variance_median.value()
        median_norm = self._norm_p50.value()
        iqr_norm = self._norm_p75.value() - self._norm_p25.value()
        summary = LayerMetricsSummary(
            median_layer_variance=median_variance,
            median_layer_norm=median_norm,
            iqr_layer_norm=iqr_norm,
        )
        logger.debug(
            "Finalized layer metrics: median_variance=%.6f median_norm=%.6f "
            "iqr_norm=%.6f.",
            summary.median_layer_variance,
            summary.median_layer_norm,
            summary.iqr_layer_norm,
        )
        return summary