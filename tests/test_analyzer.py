from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from weightlens.aggregators.streaming_global import StreamingGlobalAggregator
from weightlens.analyzer import Analyzer
from weightlens.contracts import CheckpointValidator, WeightSource
from weightlens.diagnostics import (
    AbnormalNormRule,
    DeadLayerRule,
    ExplodingVarianceRule,
    ExtremeSpikeRule,
)
from weightlens.models import CheckpointHealth, LayerTensor
from weightlens.sources.pytorch import PyTorchWeightSource
from weightlens.stats_engines.basic_stats_engine import BasicStatsEngine
from weightlens.validators.pytorch_checkpoint import PyTorchCheckpointValidator


def _save_checkpoint(tmp_path: Path) -> Path:
    state: OrderedDict[str, torch.Tensor] = OrderedDict(
        layer1=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        layer2=torch.tensor([5.0, 6.0], dtype=torch.float64),
        int_layer=torch.tensor([1, 2, 3], dtype=torch.int64),
    )
    checkpoint_path = tmp_path / "model.pth"
    torch.save(state, checkpoint_path)
    return checkpoint_path


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


class DummyValidator(CheckpointValidator):
    def __init__(self, health: CheckpointHealth) -> None:
        self._health = health

    def validate(self) -> CheckpointHealth:
        return self._health


class ExplodingSource(WeightSource):
    def iter_layers(self) -> Iterator[LayerTensor]:
        raise AssertionError("WeightSource should not be iterated.")


class SyntheticSource(WeightSource):
    def __init__(self, layer_count: int, layer_size: int) -> None:
        self._layer_count = layer_count
        self._layer_size = layer_size

    def iter_layers(self) -> Iterator[LayerTensor]:
        for index in range(self._layer_count):
            values = np.full(self._layer_size, float(index), dtype=np.float32)
            yield LayerTensor(
                name=f"layer{index}",
                values=values,
                shape=values.shape,
                dtype=str(values.dtype),
            )


def test_analyzer_happy_path(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)
    analyzer = Analyzer(
        source=PyTorchWeightSource(checkpoint_path),
        validator=PyTorchCheckpointValidator(checkpoint_path),
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=[
            DeadLayerRule(),
            ExplodingVarianceRule(),
            ExtremeSpikeRule(),
            AbnormalNormRule(),
        ],
    )

    result = analyzer.analyze()

    assert result.health.loadable is True
    assert result.health.is_empty is False
    assert len(result.layer_stats) == 2
    assert isinstance(result.diagnostics, list)
    for field in (
        "mean",
        "std",
        "p1",
        "p5",
        "p50",
        "p95",
        "p99",
        "median_layer_variance",
        "median_layer_norm",
        "iqr_layer_norm",
    ):
        assert math.isfinite(float(getattr(result.global_stats, field)))


def test_analyzer_aborts_on_unhealthy_checkpoint() -> None:
    health = CheckpointHealth(
        file_size_bytes=1,
        is_empty=False,
        loadable=False,
        tensor_count=0,
        total_params=0,
        corruption_flags=["load_failed"],
    )
    analyzer = Analyzer(
        source=ExplodingSource(),
        validator=DummyValidator(health),
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=[],
    )

    with pytest.raises(ValueError, match="Checkpoint is not loadable or is empty"):
        analyzer.analyze()


def test_analyzer_streams_large_synthetic_model() -> None:
    layer_count = 1000
    layer_size = 8
    health = CheckpointHealth(
        file_size_bytes=1,
        is_empty=False,
        loadable=True,
        tensor_count=layer_count,
        total_params=layer_count * layer_size,
        corruption_flags=[],
    )
    analyzer = Analyzer(
        source=SyntheticSource(layer_count=layer_count, layer_size=layer_size),
        validator=DummyValidator(health),
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=[],
    )

    result = analyzer.analyze()

    assert len(result.layer_stats) == layer_count
    assert _contains_ndarray(result) is False
