from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

import numpy as np
import pytest

from weightlens.aggregators.streaming_global import StreamingGlobalAggregator
from weightlens.models import LayerStats


def _expected_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "p1": float(np.quantile(values, 0.01, method="linear")),
        "p5": float(np.quantile(values, 0.05, method="linear")),
        "p50": float(np.quantile(values, 0.5, method="linear")),
        "p95": float(np.quantile(values, 0.95, method="linear")),
        "p99": float(np.quantile(values, 0.99, method="linear")),
    }


def _contains_ndarray(obj: object) -> bool:
    if isinstance(obj, np.ndarray):
        return True
    if isinstance(obj, Mapping):
        mapping = cast(Mapping[object, object], obj)
        return any(_contains_ndarray(value) for value in mapping.values())
    if isinstance(obj, (list, tuple, set)):
        iterable = cast(Iterable[object], obj)
        return any(_contains_ndarray(value) for value in iterable)
    if hasattr(obj, "__dict__"):
        return _contains_ndarray(cast(Mapping[object, object], obj.__dict__))
    return False


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


def test_streaming_global_aggregator_matches_numpy_small_set() -> None:
    values = np.linspace(-1.0, 1.0, 101, dtype=np.float64)
    expected = _expected_stats(values)
    layer_stds = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    layer_norms = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    variances = layer_stds**2
    expected_median_variance = float(np.quantile(variances, 0.5, method="linear"))
    expected_median_norm = float(np.quantile(layer_norms, 0.5, method="linear"))
    expected_iqr_norm = float(
        np.quantile(layer_norms, 0.75, method="linear")
        - np.quantile(layer_norms, 0.25, method="linear")
    )

    aggregator = StreamingGlobalAggregator()
    aggregator.update(values)
    for index, (std, norm) in enumerate(zip(layer_stds, layer_norms, strict=True)):
        aggregator.update_layer_stats(
            _make_layer_stats(std=std, l2_norm=norm, name=f"layer{index}")
        )
    stats = aggregator.finalize()

    np.testing.assert_allclose(stats.mean, expected["mean"], atol=1e-12)
    np.testing.assert_allclose(stats.std, expected["std"], atol=1e-12)
    np.testing.assert_allclose(stats.p1, expected["p1"], atol=0.05)
    np.testing.assert_allclose(stats.p5, expected["p5"], atol=0.05)
    np.testing.assert_allclose(stats.p50, expected["p50"], atol=0.05)
    np.testing.assert_allclose(stats.p95, expected["p95"], atol=0.05)
    np.testing.assert_allclose(stats.p99, expected["p99"], atol=0.05)
    np.testing.assert_allclose(
        stats.median_layer_variance, expected_median_variance, atol=1e-12
    )
    np.testing.assert_allclose(
        stats.median_layer_norm, expected_median_norm, atol=1e-12
    )
    np.testing.assert_allclose(
        stats.iqr_layer_norm, expected_iqr_norm, atol=1e-12
    )


def test_streaming_global_aggregator_is_incremental() -> None:
    values = np.linspace(-5.0, 5.0, 501, dtype=np.float64)
    layer_stds = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    layer_norms = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float64)

    full = StreamingGlobalAggregator()
    full.update(values)
    for index, (std, norm) in enumerate(zip(layer_stds, layer_norms, strict=True)):
        full.update_layer_stats(
            _make_layer_stats(std=std, l2_norm=norm, name=f"layer{index}")
        )
    full_stats = full.finalize()

    chunked = StreamingGlobalAggregator()
    for chunk in np.array_split(values, 10):
        chunked.update(chunk)
    for index, (std, norm) in enumerate(zip(layer_stds, layer_norms, strict=True)):
        chunked.update_layer_stats(
            _make_layer_stats(std=std, l2_norm=norm, name=f"layer{index}")
        )
    chunked_stats = chunked.finalize()

    np.testing.assert_allclose(chunked_stats.mean, full_stats.mean, atol=1e-12)
    np.testing.assert_allclose(chunked_stats.std, full_stats.std, atol=1e-12)
    np.testing.assert_allclose(chunked_stats.p1, full_stats.p1, atol=1e-10)
    np.testing.assert_allclose(chunked_stats.p5, full_stats.p5, atol=1e-10)
    np.testing.assert_allclose(chunked_stats.p50, full_stats.p50, atol=1e-10)
    np.testing.assert_allclose(chunked_stats.p95, full_stats.p95, atol=1e-10)
    np.testing.assert_allclose(chunked_stats.p99, full_stats.p99, atol=1e-10)
    np.testing.assert_allclose(
        chunked_stats.median_layer_variance,
        full_stats.median_layer_variance,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        chunked_stats.median_layer_norm,
        full_stats.median_layer_norm,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        chunked_stats.iqr_layer_norm,
        full_stats.iqr_layer_norm,
        atol=1e-12,
    )


def test_streaming_global_aggregator_is_streaming() -> None:
    values = np.arange(256, dtype=np.float64)
    aggregator = StreamingGlobalAggregator()
    aggregator.update(values)
    aggregator.update_layer_stats(
        _make_layer_stats(std=1.0, l2_norm=2.0, name="layer")
    )

    assert _contains_ndarray(aggregator) is False


def test_streaming_global_aggregator_rejects_empty_finalize() -> None:
    aggregator = StreamingGlobalAggregator()
    with pytest.raises(ValueError, match="No values"):
        aggregator.finalize()
