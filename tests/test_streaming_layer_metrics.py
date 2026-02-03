from __future__ import annotations

import numpy as np
import pytest

from weightlens.aggregators.streaming_layer_metrics import (
    StreamingLayerMetricsAggregator,
)
from weightlens.models import LayerStats


def _make_layer_stats(std: float, l2_norm: float, name: str) -> LayerStats:
    return LayerStats(
        name=name,
        mean=0.0,
        std=std,
        min=0.0,
        max=0.0,
        l2_norm=l2_norm,
        sparsity=0.0,
        param_count=1,
        p99_abs=0.0,
    )


def test_streaming_layer_metrics_aggregator_computes_expected_metrics() -> None:
    layer_stds = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    layer_norms = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    variances = layer_stds**2

    expected_median_variance = float(np.quantile(variances, 0.5, method="linear"))
    expected_median_norm = float(np.quantile(layer_norms, 0.5, method="linear"))
    expected_iqr_norm = float(
        np.quantile(layer_norms, 0.75, method="linear")
        - np.quantile(layer_norms, 0.25, method="linear")
    )

    aggregator = StreamingLayerMetricsAggregator()
    for index, (std, norm) in enumerate(zip(layer_stds, layer_norms, strict=True)):
        aggregator.update(
            _make_layer_stats(std=std, l2_norm=norm, name=f"layer{index}")
        )

    metrics = aggregator.finalize()

    np.testing.assert_allclose(
        metrics.median_layer_variance, expected_median_variance, atol=1e-12
    )
    np.testing.assert_allclose(
        metrics.median_layer_norm, expected_median_norm, atol=1e-12
    )
    np.testing.assert_allclose(metrics.iqr_layer_norm, expected_iqr_norm, atol=1e-12)


def test_streaming_layer_metrics_aggregator_rejects_empty_finalize() -> None:
    aggregator = StreamingLayerMetricsAggregator()
    with pytest.raises(ValueError, match="No layer stats"):
        aggregator.finalize()
