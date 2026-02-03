from __future__ import annotations

import math

from weightlens.diagnostics.rules import (
    AbnormalNormRule,
    DeadLayerRule,
    ExplodingVarianceRule,
    ExtremeSpikeRule,
)
from weightlens.models import GlobalStats, LayerStats


def _make_layer_stats(**overrides: float | int | str) -> LayerStats:
    payload: dict[str, object] = {
        "name": "layer",
        "mean": 0.0,
        "std": 1.0,
        "min": -1.0,
        "max": 1.0,
        "l2_norm": 10.0,
        "sparsity": 0.0,
        "param_count": 100,
        "p99_abs": 1.0,
    }
    payload.update(overrides)
    return LayerStats.model_validate(payload)


def _make_global_stats(**overrides: float) -> GlobalStats:
    payload: dict[str, object] = {
        "mean": 0.0,
        "std": 1.0,
        "p1": -1.0,
        "p5": -0.5,
        "p50": 0.0,
        "p95": 0.5,
        "p99": 1.0,
        "median_layer_variance": 1.0,
        "median_layer_norm": 10.0,
        "iqr_layer_norm": 2.0,
    }
    payload.update(overrides)
    return GlobalStats.model_validate(payload)


def test_dead_layer_rule_triggers_error() -> None:
    rule = DeadLayerRule()
    layer = _make_layer_stats(sparsity=0.9999)
    global_stats = _make_global_stats()

    flag = rule.check(layer, global_stats)

    assert flag is not None
    assert flag.rule == "dead-layer"
    assert flag.severity == "error"


def test_dead_layer_rule_ignores_non_dead_layers() -> None:
    rule = DeadLayerRule()
    layer = _make_layer_stats(sparsity=0.5)
    global_stats = _make_global_stats()

    assert rule.check(layer, global_stats) is None


def test_exploding_variance_rule_triggers_warn() -> None:
    rule = ExplodingVarianceRule()
    layer = _make_layer_stats(std=math.sqrt(10.0))
    global_stats = _make_global_stats(median_layer_variance=1.0)

    flag = rule.check(layer, global_stats)

    assert flag is not None
    assert flag.rule == "exploding-variance"
    assert flag.severity == "warn"


def test_exploding_variance_rule_ignores_below_threshold() -> None:
    rule = ExplodingVarianceRule()
    layer = _make_layer_stats(std=math.sqrt(9.0))
    global_stats = _make_global_stats(median_layer_variance=1.0)

    assert rule.check(layer, global_stats) is None


def test_exploding_variance_rule_guards_non_positive_denominator() -> None:
    rule = ExplodingVarianceRule()
    layer = _make_layer_stats(std=10.0)
    global_stats = _make_global_stats(median_layer_variance=0.0)

    assert rule.check(layer, global_stats) is None


def test_extreme_spike_rule_triggers_error() -> None:
    rule = ExtremeSpikeRule()
    layer = _make_layer_stats(min=-2.0, max=1000.0, p99_abs=5.0)
    global_stats = _make_global_stats()

    flag = rule.check(layer, global_stats)

    assert flag is not None
    assert flag.rule == "extreme-spike"
    assert flag.severity == "error"


def test_extreme_spike_rule_ignores_below_threshold() -> None:
    rule = ExtremeSpikeRule()
    layer = _make_layer_stats(min=-2.0, max=50.0, p99_abs=2.0)
    global_stats = _make_global_stats()

    assert rule.check(layer, global_stats) is None


def test_extreme_spike_rule_guards_non_positive_denominator() -> None:
    rule = ExtremeSpikeRule()
    layer = _make_layer_stats(min=-2.0, max=50.0, p99_abs=0.0)
    global_stats = _make_global_stats()

    assert rule.check(layer, global_stats) is None


def test_abnormal_norm_rule_triggers_warn() -> None:
    rule = AbnormalNormRule()
    layer = _make_layer_stats(l2_norm=20.0)
    global_stats = _make_global_stats(median_layer_norm=10.0, iqr_layer_norm=2.0)

    flag = rule.check(layer, global_stats)

    assert flag is not None
    assert flag.rule == "abnormal-norm"
    assert flag.severity == "warn"


def test_abnormal_norm_rule_ignores_below_threshold() -> None:
    rule = AbnormalNormRule()
    layer = _make_layer_stats(l2_norm=18.0)
    global_stats = _make_global_stats(median_layer_norm=10.0, iqr_layer_norm=2.0)

    assert rule.check(layer, global_stats) is None


def test_abnormal_norm_rule_guards_non_positive_denominator() -> None:
    rule = AbnormalNormRule()
    layer = _make_layer_stats(l2_norm=20.0)
    global_stats = _make_global_stats(median_layer_norm=10.0, iqr_layer_norm=0.0)

    assert rule.check(layer, global_stats) is None
