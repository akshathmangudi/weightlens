from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast, runtime_checkable

from weightlens.aggregators import StreamingGlobalAggregator
from weightlens.contracts import (
    CheckpointValidator,
    DiagnosticRule,
    GlobalAggregator,
    ParameterClassifier,
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
        classifier: ParameterClassifier | None = None,
    ) -> None:
        self._source = source
        self._validator = validator
        self._stats_engine = stats_engine
        self._aggregator = aggregator
        self._rules = list(rules)
        self._classifier = classifier

    def _create_bucket_aggregator(self) -> StreamingGlobalAggregator:
        """Create a fresh aggregator for a parameter bucket."""
        return StreamingGlobalAggregator()

    def analyze(self) -> AnalysisResult:
        health = self._validator.validate()
        if not health.loadable or health.is_empty:
            raise ValueError("Checkpoint is not loadable or is empty.")

        if not isinstance(self._aggregator, LayerStatsConsumer):
            raise TypeError("GlobalAggregator does not accept layer stats updates.")
        layer_stats_consumer = cast(LayerStatsConsumer, self._aggregator)

        # Per-bucket aggregators (lazy, created on first encounter)
        bucket_aggregators: dict[str, StreamingGlobalAggregator] = {}

        layer_stats: list[LayerStats] = []
        for layer in self._source.iter_layers():
            # Classify this parameter
            if self._classifier is not None:
                category = self._classifier.classify(
                    layer.name, layer.shape, layer.dtype
                )
            else:
                category = "kernel"

            if category == "skip":
                continue

            stats = self._stats_engine.compute_layer(layer)
            stats = stats.model_copy(update={"category": category})
            layer_stats.append(stats)

            # Feed the overall aggregator (preserves existing behaviour)
            self._aggregator.update(layer.values)
            layer_stats_consumer.update_layer_stats(stats)

            # Feed the per-bucket aggregator
            if category not in bucket_aggregators:
                bucket_aggregators[category] = self._create_bucket_aggregator()
            bucket_agg = bucket_aggregators[category]
            bucket_agg.update(layer.values)
            bucket_agg.update_layer_stats(stats)

        global_stats = self._aggregator.finalize()

        # Finalize per-bucket global stats
        bucket_stats = {
            cat: agg.finalize() for cat, agg in bucket_aggregators.items()
        }

        # Run diagnostics using per-bucket global stats
        diagnostics: list[DiagnosticFlag] = []
        for stats in layer_stats:
            cat = stats.category
            bucket_global = bucket_stats.get(cat)
            if bucket_global is None:
                continue
            for rule in self._rules:
                if cat not in rule.applicable_categories:
                    continue
                flag = rule.check(stats, bucket_global)
                if flag is not None:
                    diagnostics.append(flag)

        return AnalysisResult(
            layer_stats=layer_stats,
            global_stats=global_stats,
            bucket_stats=bucket_stats,
            diagnostics=diagnostics,
            health=health,
        )
