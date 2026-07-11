from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures_safetensors import write_single
from weightlens.aggregators.streaming_global import StreamingGlobalAggregator
from weightlens.analyzer import Analyzer
from weightlens.classifiers import PyTorchParameterClassifier
from weightlens.contracts import CheckpointValidator
from weightlens.diagnostics import (
    AbnormalNormRule,
    DeadLayerRule,
    ExplodingVarianceRule,
    ExtremeSpikeRule,
)
from weightlens.models import AnalysisResult, CheckpointHealth, LayerStats, LayerTensor
from weightlens.p2_quantile import P2QuantileEstimator
from weightlens.sources.safetensors import SafetensorsWeightSource
from weightlens.stats_engines.basic_stats_engine import BasicStatsEngine
from weightlens.validators.safetensors import SafetensorsCheckpointValidator


class _StaticValidator(CheckpointValidator):
    def __init__(self, health: CheckpointHealth) -> None:
        self._health = health

    def validate(self) -> CheckpointHealth:
        return self._health


def _run_full_analysis(
    uri: str, *, num_workers: int | None = 1
) -> AnalysisResult:
    validator = SafetensorsCheckpointValidator(uri)
    health = validator.validate()
    assert health.loadable
    assert not health.is_empty

    analyzer = Analyzer(
        source=SafetensorsWeightSource(uri),
        validator=_StaticValidator(health),
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=[
            DeadLayerRule(),
            ExplodingVarianceRule(),
            ExtremeSpikeRule(),
            AbnormalNormRule(),
        ],
        classifier=PyTorchParameterClassifier(),
        num_workers=num_workers,
    )
    return analyzer.analyze()


def _make_layer(values: np.ndarray, name: str = "layer") -> LayerTensor:
    return LayerTensor(
        name=name,
        values=values,
        shape=values.shape,
        dtype=str(values.dtype),
    )


def test_all_zeros_layer_sparsity_exact(tmp_path: Path) -> None:
    tensors: dict[str, np.ndarray] = {
        "model.weight": np.zeros((10, 10), dtype=np.float32),
    }
    path = str(tmp_path / "zeros.safetensors")
    write_single(path, tensors)

    result = _run_full_analysis(path)
    got = result.layer_stats[0]

    assert got.sparsity == 1.0
    assert got.std == 0.0


def test_single_element_layer_no_crash() -> None:
    values = np.array([3.14], dtype=np.float32)
    layer = _make_layer(values, name="single")
    engine = BasicStatsEngine()

    stats = engine.compute_layer(layer)

    assert stats.name == "single"
    assert stats.param_count == 1
    np.testing.assert_allclose(stats.mean, 3.14)
    np.testing.assert_allclose(stats.std, 0.0, atol=1e-3)
    np.testing.assert_allclose(stats.min, 3.14)
    np.testing.assert_allclose(stats.max, 3.14)


def test_nan_tensor_raises_valueerror() -> None:
    values = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    layer = _make_layer(values, name="with_nan")
    engine = BasicStatsEngine()

    with pytest.raises(ValueError, match="NaN"):
        engine.compute_layer(layer)


def test_inf_tensor_raises_valueerror() -> None:
    values = np.array([1.0, np.inf, 2.0], dtype=np.float32)
    layer = _make_layer(values, name="with_inf")
    engine = BasicStatsEngine()

    with pytest.raises(ValueError, match="NaN"):
        engine.compute_layer(layer)


def test_float32_large_sum_no_overflow() -> None:
    values = np.full(1_000_000, 1e10, dtype=np.float32)
    layer = _make_layer(values, name="large")
    engine = BasicStatsEngine()

    stats = engine.compute_layer(layer)

    assert math.isfinite(stats.mean)
    assert math.isfinite(stats.std)
    assert math.isfinite(stats.l2_norm)
    np.testing.assert_allclose(stats.mean, 1e10, rtol=1e-6)


def test_variance_guards_against_negative() -> None:
    base = np.array([0.5, 0.5, 0.5 + 1e-7, 0.5 - 1e-7, 0.5], dtype=np.float32)
    layer = _make_layer(base, name="near_constant")
    engine = BasicStatsEngine()

    stats = engine.compute_layer(layer)

    true_std = float(np.std(base.astype(np.float64), ddof=0))
    np.testing.assert_allclose(stats.std, true_std, rtol=1e-2, atol=1e-12)
    assert stats.std >= 0.0


def test_histogram_overflow_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    rng = np.random.default_rng(42)
    big = (rng.normal(0, 1, size=(5, 5)) * 1e4).astype(np.float32)
    small = (rng.normal(0, 1, size=(5, 5)) * 1e-4).astype(np.float32)

    with caplog.at_level(logging.WARNING):
        agg = StreamingGlobalAggregator()
        agg.update(big)
        agg.update(small)
        agg.update_layer_stats(
            LayerStats(
                name="layer",
                mean=0.0,
                std=1.0,
                min=0.0,
                max=0.0,
                l2_norm=1.0,
                sparsity=0.0,
                param_count=1,
                p99_abs=0.0,
            )
        )
        agg.finalize()

    overflow_terms = ["overflow", "underflow"]
    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any(
        any(term in msg.lower() for term in overflow_terms)
        for msg in warning_messages
    )


def test_welford_matches_full_pass() -> None:
    rng = np.random.default_rng(99)
    big = (rng.normal(0, 1, size=(1000,)) * 1e4).astype(np.float64)
    small = (rng.normal(0, 1, size=(1000,)) * 1e-4).astype(np.float64)
    mid = rng.normal(0, 1, size=(200,)).astype(np.float64)

    agg = StreamingGlobalAggregator()
    agg.update(big)
    agg.update(small)
    agg.update(mid)
    agg.update_layer_stats(
        LayerStats(
            name="layer",
            mean=0.0,
            std=1.0,
            min=0.0,
            max=0.0,
            l2_norm=1.0,
            sparsity=0.0,
            param_count=1,
            p99_abs=0.0,
        )
    )
    welford = agg.finalize()

    all_values = np.concatenate([big, small, mid])
    oracle_mean = float(np.mean(all_values))
    oracle_std = float(np.std(all_values, ddof=0))

    np.testing.assert_allclose(welford.mean, oracle_mean, rtol=1e-12)
    np.testing.assert_allclose(welford.std, oracle_std, rtol=1e-8)


def test_p2_median_of_1000_normal_samples() -> None:
    rng = np.random.default_rng(1234)
    samples = rng.normal(0, 1, size=1000).astype(np.float64)

    p2 = P2QuantileEstimator(0.5)
    for v in samples:
        p2.update(float(v))

    estimated = p2.value()
    true_median = float(np.median(samples))

    assert abs(estimated - true_median) < 0.05 * abs(true_median) + 0.1


def test_p2_stable_on_extreme_values() -> None:
    p2 = P2QuantileEstimator(0.5)

    for i in range(100):
        p2.update(float(1e30 if i % 2 == 0 else -1e30))
        p2.update(0.0)

    result = p2.value()
    assert math.isfinite(result)


def test_mixed_sign_tensor_stats_correct() -> None:
    values = np.array([-3.0, -1.0, 0.5, 2.0, 5.0], dtype=np.float32)
    layer = _make_layer(values, name="mixed")
    engine = BasicStatsEngine()

    stats = engine.compute_layer(layer)

    assert stats.min < 0.0
    assert stats.max > 0.0
    assert stats.param_count == 5
    assert stats.std > 0.0
    assert stats.sparsity == 0.0


def test_bf16_roundtrip_matches_oracle(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, size=(20, 16)).astype(np.float32)
    bf16_tensor = torch.from_numpy(base).to(torch.bfloat16)
    path = str(tmp_path / "bf16.safetensors")
    save_file({"layer.weight": bf16_tensor}, path)

    result = _run_full_analysis(path)
    got = result.layer_stats[0]

    oracle_values = bf16_tensor.float().numpy()
    flat = oracle_values.astype(np.float64).ravel()
    oracle_mean = float(np.mean(flat))
    oracle_std = float(np.std(flat, ddof=0))
    oracle_min = float(np.min(flat))
    oracle_max = float(np.max(flat))
    oracle_l2 = float(np.sqrt(np.sum(flat * flat)))
    oracle_sparsity = 1.0 - (int(np.count_nonzero(flat)) / flat.size)
    oracle_p99 = float(np.quantile(np.abs(flat), 0.99, method="linear"))

    np.testing.assert_allclose(got.mean, oracle_mean, rtol=1e-5)
    np.testing.assert_allclose(got.std, oracle_std, rtol=1e-5)
    np.testing.assert_allclose(got.min, oracle_min, rtol=1e-5)
    np.testing.assert_allclose(got.max, oracle_max, rtol=1e-5)
    np.testing.assert_allclose(got.l2_norm, oracle_l2, rtol=1e-5)
    assert got.sparsity == pytest.approx(oracle_sparsity, abs=1e-12)
    np.testing.assert_allclose(got.p99_abs, oracle_p99, rtol=1e-5)
    assert got.param_count == int(flat.size)
