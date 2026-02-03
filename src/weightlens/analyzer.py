from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast, runtime_checkable

from weightlens.contracts import (
    CheckpointValidator,
    DiagnosticRule,
    GlobalAggregator,
    StatsEngine,
    WeightSource,
)
from weightlens.models import AnalysisResult, DiagnosticFlag, LayerStats


@runtime_checkable
class LayerStatsConsumer(Protocol):
    """Protocol for aggregators that consume per-layer stats."""

    def update_layer_stats(self, layer_stats: LayerStats) -> None:
        """Consume per-layer statistics for global aggregation."""


class Analyzer:
    """Orchestrate streaming analysis across validation, stats, and diagnostics."""

    def __init__(
        self,
        source: WeightSource,
        validator: CheckpointValidator,
        stats_engine: StatsEngine,
        aggregator: GlobalAggregator,
        rules: Sequence[DiagnosticRule],
    ) -> None:
        self._source = source
        self._validator = validator
        self._stats_engine = stats_engine
        self._aggregator = aggregator
        self._rules = list(rules)

    def analyze(self) -> AnalysisResult:
        health = self._validator.validate()
        if not health.loadable or health.is_empty:
            raise ValueError("Checkpoint is not loadable or is empty.")

        if not isinstance(self._aggregator, LayerStatsConsumer):
            raise TypeError("GlobalAggregator does not accept layer stats updates.")
        layer_stats_consumer = cast(LayerStatsConsumer, self._aggregator)

        layer_stats: list[LayerStats] = []
        for layer in self._source.iter_layers():
            stats = self._stats_engine.compute_layer(layer)
            layer_stats.append(stats)
            self._aggregator.update(layer.values)
            layer_stats_consumer.update_layer_stats(stats)

        global_stats = self._aggregator.finalize()

        diagnostics: list[DiagnosticFlag] = []
        for stats in layer_stats:
            for rule in self._rules:
                flag = rule.check(stats, global_stats)
                if flag is not None:
                    diagnostics.append(flag)

        return AnalysisResult(
            layer_stats=layer_stats,
            global_stats=global_stats,
            diagnostics=diagnostics,
            health=health,
        )
