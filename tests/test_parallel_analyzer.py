from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import numpy as np

from weightlens.aggregators.streaming_global import StreamingGlobalAggregator
from weightlens.analyzer import Analyzer
from weightlens.classifiers import PyTorchParameterClassifier
from weightlens.contracts import CheckpointValidator, WeightSource
from weightlens.diagnostics import (
    AbnormalNormRule,
    DeadLayerRule,
    ExplodingVarianceRule,
    ExtremeSpikeRule,
)
from weightlens.models import CheckpointHealth, LayerTensor
from weightlens.stats_engines.basic_stats_engine import BasicStatsEngine


class _DummyValidator(CheckpointValidator):
    def __init__(self, count: int, size: int) -> None:
        self._count = count
        self._size = size

    def validate(self) -> CheckpointHealth:
        return CheckpointHealth(
            file_size_bytes=1,
            is_empty=False,
            loadable=True,
            tensor_count=self._count,
            total_params=self._count * self._size,
            corruption_flags=[],
        )


class _SyntheticSource(WeightSource):
    def __init__(self, layer_count: int, layer_size: int) -> None:
        self._layer_count = layer_count
        self._layer_size = layer_size

    def iter_layers(self) -> Iterator[LayerTensor]:
        rng = np.random.default_rng(42)
        for i in range(self._layer_count):
            values = rng.standard_normal(self._layer_size).astype(np.float32)
            yield LayerTensor(
                name=f"layer.{i}.weight",
                values=values,
                shape=(self._layer_size,),
                dtype="float32",
            )


def _build_analyzer(
    layer_count: int, layer_size: int, num_workers: int | None
) -> Analyzer:
    return Analyzer(
        source=_SyntheticSource(layer_count, layer_size),
        validator=_DummyValidator(layer_count, layer_size),
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=[
            DeadLayerRule(),
            ExplodingVarianceRule(),
            ExtremeSpikeRule(),
            AbnormalNormRule(),
        ],
        classifier=PyTorchParameterClassifier(),
        prefetch=False,  # deterministic ordering for comparison
        num_workers=num_workers,
    )


def test_parallel_matches_sequential() -> None:
    """num_workers=1 and num_workers=4 must produce identical results."""
    seq = _build_analyzer(50, 256, num_workers=1).analyze()
    par = _build_analyzer(50, 256, num_workers=4).analyze()

    assert len(seq.layer_stats) == len(par.layer_stats)
    for s, p in zip(seq.layer_stats, par.layer_stats, strict=True):
        assert s.name == p.name
        np.testing.assert_allclose(s.mean, p.mean, atol=1e-6)
        np.testing.assert_allclose(s.std, p.std, atol=1e-6)

    np.testing.assert_allclose(seq.global_stats.mean, par.global_stats.mean, atol=1e-6)
    np.testing.assert_allclose(seq.global_stats.std, par.global_stats.std, atol=1e-6)


def test_low_memory_reduces_workers() -> None:
    """When memory is low, compute_max_workers returns 1."""
    with patch(
        "weightlens.memory.available_memory_bytes", return_value=1024 * 1024 * 512
    ):
        from weightlens.memory import compute_max_workers

        workers = compute_max_workers(avg_tensor_bytes=325 * 1024 * 1024)
        assert workers == 1


def test_stress_200_layers() -> None:
    result = _build_analyzer(200, 64, num_workers=4).analyze()
    assert len(result.layer_stats) == 200
