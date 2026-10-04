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
    np.testing.assert_allclose(stats.p99_abs, expected_p99_abs, rtol=0.006, atol=0)


@pytest.mark.parametrize("scale", [1e-8, 1e-4, 1.0, 1e4, 1e8])
def test_p99_abs_tracks_tensor_scale(scale: float) -> None:
    rng = np.random.default_rng(42)
    values = (rng.normal(size=10_000) * scale).astype(np.float64)
    values[::17] = 0.0
    values[0] = -100.0 * scale

    stats = BasicStatsEngine().compute_layer(_make_layer(values))
    expected = float(np.quantile(np.abs(values), 0.99, method="linear"))

    np.testing.assert_allclose(stats.p99_abs, expected, rtol=0.006, atol=0)


def test_p99_abs_detects_small_scale_spike() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(0.0, 1e-4, size=10_000).astype(np.float32)
    values[0] = 0.1

    stats = BasicStatsEngine().compute_layer(_make_layer(values))
    expected = float(np.quantile(np.abs(values), 0.99, method="linear"))

    np.testing.assert_allclose(stats.p99_abs, expected, rtol=0.006, atol=0)
    assert float(np.max(np.abs(values))) / stats.p99_abs >= 100.0


def test_p99_abs_combines_chunks() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(0.0, 1e-5, size=1_000_123).astype(np.float32)
    values[-1] = -0.1

    stats = BasicStatsEngine().compute_layer(_make_layer(values))
    expected = float(np.quantile(np.abs(values), 0.99, method="linear"))

    np.testing.assert_allclose(stats.p99_abs, expected, rtol=0.006, atol=0)


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
