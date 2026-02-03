from __future__ import annotations

import numpy as np
import pytest

from weightlens.models import LayerTensor
from weightlens.stats_engines.basic_stats_engine import BasicStatsEngine


def _make_layer(values: np.ndarray, name: str = "layer") -> LayerTensor:
    return LayerTensor(
        name=name,
        values=values,
        shape=values.shape,
        dtype=str(values.dtype),
    )


def test_basic_stats_engine_computes_expected_stats() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    engine = BasicStatsEngine()

    stats = engine.compute_layer(_make_layer(values, name="layer1"))
    expected_p99_abs = float(np.quantile(np.abs(values), 0.99, method="linear"))

    assert stats.name == "layer1"
    assert stats.param_count == 4
    np.testing.assert_allclose(stats.mean, 2.5)
    np.testing.assert_allclose(stats.std, np.sqrt(1.25))
    np.testing.assert_allclose(stats.min, 1.0)
    np.testing.assert_allclose(stats.max, 4.0)
    np.testing.assert_allclose(stats.l2_norm, np.sqrt(30.0))
    np.testing.assert_allclose(stats.sparsity, 0.0)
    np.testing.assert_allclose(stats.p99_abs, expected_p99_abs)


def test_basic_stats_engine_reports_zero_sparsity_metrics() -> None:
    values = np.zeros(5, dtype=np.float32)
    engine = BasicStatsEngine()

    stats = engine.compute_layer(_make_layer(values, name="zeros"))

    np.testing.assert_allclose(stats.mean, 0.0)
    np.testing.assert_allclose(stats.std, 0.0)
    np.testing.assert_allclose(stats.min, 0.0)
    np.testing.assert_allclose(stats.max, 0.0)
    np.testing.assert_allclose(stats.l2_norm, 0.0)
    np.testing.assert_allclose(stats.sparsity, 1.0)
    np.testing.assert_allclose(stats.p99_abs, 0.0)
    assert stats.param_count == 5


def test_basic_stats_engine_raises_on_nan_values() -> None:
    values = np.array([1.0, np.nan], dtype=np.float32)
    engine = BasicStatsEngine()

    with pytest.raises(ValueError, match="NaN"):
        engine.compute_layer(_make_layer(values))


def test_basic_stats_engine_rejects_empty_layers() -> None:
    values = np.array([], dtype=np.float32)
    engine = BasicStatsEngine()

    with pytest.raises(ValueError, match="empty"):
        engine.compute_layer(_make_layer(values))


def test_basic_stats_engine_is_deterministic() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    engine = BasicStatsEngine()
    layer = _make_layer(values)

    first = engine.compute_layer(layer)
    second = engine.compute_layer(layer)

    assert first.model_dump() == second.model_dump()
