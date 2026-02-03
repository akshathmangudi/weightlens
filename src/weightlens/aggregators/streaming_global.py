from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from weightlens.aggregators.streaming_layer_metrics import (
    StreamingLayerMetricsAggregator,
)
from weightlens.contracts import GlobalAggregator
from weightlens.histogram_quantiles import FixedRangeHistogramQuantiles
from weightlens.models import GlobalStats, LayerStats

logger = logging.getLogger(__name__)


class StreamingGlobalAggregator(GlobalAggregator):
    """Streaming global metrics via Welford + fixed-range histogram quantiles."""

    def __init__(
        self,
        histogram_bins: int = 2048,
        histogram_min: float = -10.0,
        histogram_max: float = 10.0,
    ) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._quantiles = FixedRangeHistogramQuantiles(
            min_value=histogram_min,
            max_value=histogram_max,
            bins=histogram_bins,
        )
        self._layer_metrics = StreamingLayerMetricsAggregator()

    def update(self, values: NDArray[np.number]) -> None:
        value_count = int(values.size)
        logger.debug(
            "Updating global aggregator with %d values shape=%s.",
            value_count,
            values.shape,
        )
        if value_count == 0:
            return

        batch_mean = float(np.mean(values))
        batch_variance = float(np.var(values, ddof=0))
        if not math.isfinite(batch_mean) or not math.isfinite(batch_variance):
            logger.error("Non-finite value encountered in global aggregation.")
            raise ValueError("Non-finite value encountered in global aggregation.")

        batch_m2 = batch_variance * value_count
        self._merge_batch(value_count, batch_mean, batch_m2)
        self._quantiles.update(values)
        logger.debug("Updated global aggregator count=%d.", self._count)

    def update_layer_stats(self, layer_stats: LayerStats) -> None:
        self._layer_metrics.update(layer_stats)

    def finalize(self) -> GlobalStats:
        if self._count == 0:
            logger.error("Finalize called without any values.")
            raise ValueError("No values provided for global aggregation.")
        variance = self._m2 / self._count
        layer_metrics = self._layer_metrics.finalize()
        stats = GlobalStats(
            mean=self._mean,
            std=math.sqrt(variance),
            p1=self._quantiles.quantile(0.01),
            p5=self._quantiles.quantile(0.05),
            p50=self._quantiles.quantile(0.5),
            p95=self._quantiles.quantile(0.95),
            p99=self._quantiles.quantile(0.99),
            median_layer_variance=layer_metrics.median_layer_variance,
            median_layer_norm=layer_metrics.median_layer_norm,
            iqr_layer_norm=layer_metrics.iqr_layer_norm,
        )
        logger.debug(
            "Finalized global stats: mean=%.6f std=%.6f p1=%.6f p5=%.6f "
            "p50=%.6f p95=%.6f p99=%.6f median_var=%.6f median_norm=%.6f "
            "iqr_norm=%.6f.",
            stats.mean,
            stats.std,
            stats.p1,
            stats.p5,
            stats.p50,
            stats.p95,
            stats.p99,
            stats.median_layer_variance,
            stats.median_layer_norm,
            stats.iqr_layer_norm,
        )
        return stats

    def _merge_batch(self, count: int, mean: float, m2: float) -> None:
        if count == 0:
            return
        if self._count == 0:
            self._count = count
            self._mean = mean
            self._m2 = m2
            return

        n1 = self._count
        n2 = count
        mean1 = self._mean
        mean2 = mean
        delta = mean2 - mean1
        total = n1 + n2
        self._mean = mean1 + (delta * n2 / total)
        self._m2 = self._m2 + m2 + (delta * delta * n1 * n2 / total)
        self._count = total

