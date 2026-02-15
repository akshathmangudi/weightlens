from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from weightlens.aggregators import StreamingGlobalAggregator
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
from weightlens.stats_engines import BasicStatsEngine

_TOTAL_PARAMS = 64 * 64 + 64 + 64 + 64 + 1

_ALL_RULES = [
    DeadLayerRule(),
    ExplodingVarianceRule(),
    ExtremeSpikeRule(),
    AbnormalNormRule(),
]


class _DummyValidator(CheckpointValidator):
    def __init__(
        self, tensor_count: int, total_params: int
    ) -> None:
        self._count = tensor_count
        self._total = total_params

    def validate(self) -> CheckpointHealth:
        return CheckpointHealth(
            file_size_bytes=1,
            is_empty=False,
            loadable=True,
            tensor_count=self._count,
            total_params=self._total,
            corruption_flags=[],
        )


class _MixedParamSource(WeightSource):
    """Synthetic source with weight, bias, norm, buffer params."""

    def iter_layers(self) -> Iterator[LayerTensor]:
        # A normal kernel weight
        rng = np.random.default_rng(42)
        raw = rng.normal(0, 0.02, size=(64, 64))
        kernel = raw.astype(np.float32).ravel()
        yield LayerTensor(
            name="layer.weight",
            values=kernel,
            shape=(64, 64),
            dtype="float32",
        )

        # An all-zero bias -- should NOT trigger dead-layer
        bias = np.zeros(64, dtype=np.float32)
        yield LayerTensor(
            name="layer.bias",
            values=bias,
            shape=(64,),
            dtype="float32",
        )

        # A norm gamma near 1.0
        gamma = np.ones(64, dtype=np.float32) + 0.001
        yield LayerTensor(
            name="layer_norm.weight",
            values=gamma,
            shape=(64,),
            dtype="float32",
        )

        # A running_mean buffer
        buf = np.zeros(64, dtype=np.float32)
        yield LayerTensor(
            name="bn.running_mean",
            values=buf,
            shape=(64,),
            dtype="float32",
        )

        # A num_batches_tracked -- should be skipped entirely
        tracked = np.array([100], dtype=np.float32)
        yield LayerTensor(
            name="bn.num_batches_tracked",
            values=tracked,
            shape=(1,),
            dtype="float32",
        )


def _run_analysis() -> tuple[list[str], list[str]]:
    """Run analysis and return (layer_names, diagnostic_rules)."""
    source = _MixedParamSource()
    validator = _DummyValidator(
        tensor_count=5, total_params=_TOTAL_PARAMS
    )
    analyzer = Analyzer(
        source=source,
        validator=validator,
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=_ALL_RULES,
        classifier=PyTorchParameterClassifier(),
    )
    result = analyzer.analyze()
    layer_names = [s.name for s in result.layer_stats]
    triggered = [d.rule for d in result.diagnostics]
    return layer_names, triggered


def test_skip_parameters_excluded() -> None:
    """Only 'skip' (num_batches_tracked) is excluded."""
    layer_names, _ = _run_analysis()
    assert "bn.num_batches_tracked" not in layer_names
    # buffer params are included, just bucketed separately
    assert "bn.running_mean" in layer_names


def test_dead_layer_not_triggered_on_zero_bias() -> None:
    _, triggered_rules = _run_analysis()
    assert "dead-layer" not in triggered_rules


def test_exploding_variance_compares_within_bucket() -> None:
    """Norm gamma (std~0) not compared against kernel variance."""
    _, triggered_rules = _run_analysis()
    assert "exploding-variance" not in triggered_rules


def test_bucket_stats_present() -> None:
    source = _MixedParamSource()
    validator = _DummyValidator(
        tensor_count=5, total_params=_TOTAL_PARAMS
    )
    analyzer = Analyzer(
        source=source,
        validator=validator,
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=[],
        classifier=PyTorchParameterClassifier(),
    )
    result = analyzer.analyze()
    assert "kernel" in result.bucket_stats
    assert "bias" in result.bucket_stats
    assert "norm_scale" in result.bucket_stats
    assert "buffer" in result.bucket_stats
    # skip should NOT appear in bucket_stats
    assert "skip" not in result.bucket_stats


def test_layer_stats_have_category() -> None:
    source = _MixedParamSource()
    validator = _DummyValidator(
        tensor_count=5, total_params=_TOTAL_PARAMS
    )
    analyzer = Analyzer(
        source=source,
        validator=validator,
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=[],
        classifier=PyTorchParameterClassifier(),
    )
    result = analyzer.analyze()
    cats = {s.name: s.category for s in result.layer_stats}
    assert cats["layer.weight"] == "kernel"
    assert cats["layer.bias"] == "bias"
    assert cats["layer_norm.weight"] == "norm_scale"


def test_real_checkpoint_with_classifier(
    tmp_path: Path,
) -> None:
    """End-to-end with a real .pth file."""
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    state["encoder.weight"] = torch.randn(32, 32)
    state["encoder.bias"] = torch.zeros(32)
    state["norm.weight"] = torch.ones(32)
    state["norm.bias"] = torch.zeros(32)
    state["bn.running_mean"] = torch.zeros(32)
    state["bn.num_batches_tracked"] = torch.tensor(100)

    path = tmp_path / "mixed.pth"
    torch.save(state, path)

    from weightlens.sources import PyTorchWeightSource
    from weightlens.validators import PyTorchCheckpointValidator

    validator = PyTorchCheckpointValidator(path)
    health = validator.validate()

    analyzer = Analyzer(
        source=PyTorchWeightSource(path),
        validator=_DummyValidator(
            tensor_count=health.tensor_count,
            total_params=health.total_params,
        ),
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=_ALL_RULES,
        classifier=PyTorchParameterClassifier(),
    )
    result = analyzer.analyze()

    names = [s.name for s in result.layer_stats]
    assert "bn.num_batches_tracked" not in names
    assert "encoder.weight" in names
    assert "bn.running_mean" in names

    # All-zero biases should not trigger dead-layer
    dead = [
        d.layer
        for d in result.diagnostics
        if d.rule == "dead-layer"
    ]
    assert "encoder.bias" not in dead
    assert "norm.bias" not in dead
