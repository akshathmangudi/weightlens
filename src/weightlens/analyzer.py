from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from weightlens.aggregators import StreamingGlobalAggregator
from weightlens.contracts import (
    CheckpointValidator,
    DiagnosticRule,
    GlobalAggregator,
    ParameterClassifier,
    StatsEngine,
    WeightSource,
)
from weightlens.memory import compute_max_workers
from weightlens.models import (
    AnalysisResult,
    DiagnosticFlag,
    LayerStats,
    LayerTensor,
)
from weightlens.prefetch import PrefetchIterator

logger = logging.getLogger(__name__)


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
        prefetch: bool = True,
        num_workers: int | None = None,
    ) -> None:
        self._source = source
        self._validator = validator
        self._stats_engine = stats_engine
        self._aggregator = aggregator
        self._rules = list(rules)
        self._classifier = classifier
        self._prefetch = prefetch
        self._num_workers = num_workers

    def _create_bucket_aggregator(self) -> StreamingGlobalAggregator:
        """Create a fresh aggregator for a parameter bucket."""
        return StreamingGlobalAggregator()

    def _resolve_workers(self) -> int:
        if self._num_workers is not None:
            return max(1, self._num_workers)
        return compute_max_workers()

    def analyze(self) -> AnalysisResult:
        health = self._validator.validate()
        if not health.loadable or health.is_empty:
            raise ValueError("Checkpoint is not loadable or is empty.")

        if not isinstance(self._aggregator, LayerStatsConsumer):
            raise TypeError("GlobalAggregator does not accept layer stats updates.")
        layer_stats_consumer = cast(LayerStatsConsumer, self._aggregator)

        num_workers = self._resolve_workers()

        if num_workers > 1:
            return self._analyze_parallel(layer_stats_consumer, health, num_workers)
        return self._analyze_sequential(layer_stats_consumer, health)

    def _classify(self, layer: LayerTensor) -> str:
        if self._classifier is not None:
            return self._classifier.classify(layer.name, layer.shape, layer.dtype)
        return "kernel"

    def _analyze_sequential(
        self,
        layer_stats_consumer: LayerStatsConsumer,
        health: object,
    ) -> AnalysisResult:
        from weightlens.models import CheckpointHealth

        typed_health = cast(CheckpointHealth, health)
        bucket_aggregators: dict[str, StreamingGlobalAggregator] = {}
        diagnostics: list[DiagnosticFlag] = []
        layer_stats: list[LayerStats] = []

        layer_iter = self._source.iter_layers()
        if self._prefetch:
            layer_iter = PrefetchIterator(layer_iter)

        for layer in layer_iter:
            category = self._classify(layer)
            if category == "skip":
                continue

            try:
                stats = self._stats_engine.compute_layer(layer)
            except ValueError:
                logger.warning(
                    "Skipping layer %s: non-finite or invalid values.", layer.name
                )
                diagnostics.append(
                    DiagnosticFlag(
                        layer=layer.name,
                        rule="non-finite-values",
                        message="Layer contains NaN or Inf values",
                        severity="error",
                    )
                )
                continue

            stats = stats.model_copy(update={"category": category})
            layer_stats.append(stats)

            count = stats.param_count
            mean = stats.mean
            variance = stats.std**2

            try:
                self._aggregator.update_from_summary(
                    layer.values, count=count, mean=mean, variance=variance
                )
            except ValueError:
                logger.warning(
                    "Skipping layer %s in global aggregation: non-finite stats.",
                    layer.name,
                )
                layer_stats.pop()
                diagnostics.append(
                    DiagnosticFlag(
                        layer=layer.name,
                        rule="non-finite-values",
                        message="Layer produces non-finite global statistics",
                        severity="error",
                    )
                )
                continue
            layer_stats_consumer.update_layer_stats(stats)

            if category not in bucket_aggregators:
                bucket_aggregators[category] = self._create_bucket_aggregator()
            bucket_agg = bucket_aggregators[category]
            bucket_agg.update_from_summary(
                layer.values, count=count, mean=mean, variance=variance
            )
            bucket_agg.update_layer_stats(stats)

        return self._finalize(
            layer_stats, diagnostics, bucket_aggregators, typed_health
        )

    def _analyze_parallel(
        self,
        layer_stats_consumer: LayerStatsConsumer,
        health: object,
        num_workers: int,
    ) -> AnalysisResult:
        from weightlens.models import CheckpointHealth

        typed_health = cast(CheckpointHealth, health)
        bucket_aggregators: dict[str, StreamingGlobalAggregator] = {}
        diagnostics: list[DiagnosticFlag] = []
        layer_stats: list[LayerStats] = []

        # Bounded window of futures to cap memory usage
        pending: deque[tuple[str, str, NDArray[np.number], Future[LayerStats]]] = (
            deque()
        )

        layer_iter = self._source.iter_layers()
        if self._prefetch:
            layer_iter = PrefetchIterator(layer_iter)

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for layer in layer_iter:
                category = self._classify(layer)
                if category == "skip":
                    continue

                future = pool.submit(self._stats_engine.compute_layer, layer)
                pending.append((layer.name, category, layer.values, future))

                # Drain completed futures from the front to bound memory
                while len(pending) > num_workers and pending[0][3].done():
                    self._drain_one(
                        pending,
                        layer_stats,
                        diagnostics,
                        bucket_aggregators,
                        layer_stats_consumer,
                    )

            # Drain remaining futures in submission order
            while pending:
                self._drain_one(
                    pending,
                    layer_stats,
                    diagnostics,
                    bucket_aggregators,
                    layer_stats_consumer,
                )

        return self._finalize(
            layer_stats, diagnostics, bucket_aggregators, typed_health
        )

    def _drain_one(
        self,
        pending: deque[tuple[str, str, NDArray[np.number], Future[LayerStats]]],
        layer_stats: list[LayerStats],
        diagnostics: list[DiagnosticFlag],
        bucket_aggregators: dict[str, StreamingGlobalAggregator],
        layer_stats_consumer: LayerStatsConsumer,
    ) -> None:
        name, category, values, future = pending.popleft()
        try:
            stats = future.result()
        except ValueError:
            logger.warning("Skipping layer %s: non-finite or invalid values.", name)
            diagnostics.append(
                DiagnosticFlag(
                    layer=name,
                    rule="non-finite-values",
                    message="Layer contains NaN or Inf values",
                    severity="error",
                )
            )
            return

        stats = stats.model_copy(update={"category": category})
        layer_stats.append(stats)

        count = stats.param_count
        mean = stats.mean
        variance = stats.std**2

        try:
            self._aggregator.update_from_summary(
                values, count=count, mean=mean, variance=variance
            )
        except ValueError:
            logger.warning(
                "Skipping layer %s in global aggregation: non-finite stats.",
                name,
            )
            layer_stats.pop()
            diagnostics.append(
                DiagnosticFlag(
                    layer=name,
                    rule="non-finite-values",
                    message="Layer produces non-finite global statistics",
                    severity="error",
                )
            )
            return
        layer_stats_consumer.update_layer_stats(stats)

        if category not in bucket_aggregators:
            bucket_aggregators[category] = self._create_bucket_aggregator()
        bucket_agg = bucket_aggregators[category]
        bucket_agg.update_from_summary(
            values, count=count, mean=mean, variance=variance
        )
        bucket_agg.update_layer_stats(stats)

    def _finalize(
        self,
        layer_stats: list[LayerStats],
        diagnostics: list[DiagnosticFlag],
        bucket_aggregators: dict[str, StreamingGlobalAggregator],
        health: object,
    ) -> AnalysisResult:
        from weightlens.models import CheckpointHealth

        typed_health = cast(CheckpointHealth, health)
        global_stats = self._aggregator.finalize()

        bucket_stats = {cat: agg.finalize() for cat, agg in bucket_aggregators.items()}

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
            health=typed_health,
        )
