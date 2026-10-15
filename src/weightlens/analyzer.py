from __future__ import annotations

import logging
import math
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
    AnalysisCoverage,
    AnalysisResult,
    CheckpointHealth,
    DiagnosticFlag,
    LayerStats,
    LayerTensor,
    TensorOmission,
)
from weightlens.prefetch import PrefetchIterator
from weightlens.stats_engines.errors import EmptyTensorError, NonFiniteTensorError

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

    @staticmethod
    def _record_unanalyzed(
        name: str,
        dtype: str,
        reason: str,
        rule: str,
        message: str,
        diagnostics: list[DiagnosticFlag],
        omissions: list[TensorOmission],
        severity: str = "error",
    ) -> None:
        omissions.append(
            TensorOmission(name=name, kind="skipped", reason=reason, dtype=dtype)
        )
        diagnostics.append(
            DiagnosticFlag(layer=name, rule=rule, message=message, severity=severity)
        )

    @classmethod
    def _record_stats_error(
        cls,
        name: str,
        dtype: str,
        error: EmptyTensorError | NonFiniteTensorError,
        diagnostics: list[DiagnosticFlag],
        omissions: list[TensorOmission],
    ) -> None:
        if isinstance(error, EmptyTensorError):
            reason = "empty tensor"
            rule = "empty-tensor"
            message = "Tensor is empty. Statistics skipped."
        else:
            reason = "NaN or Inf values"
            rule = "non-finite-values"
            message = "Tensor contains NaN or Inf values. Statistics skipped."
        logger.warning("Skipping layer %s: %s.", name, reason)
        cls._record_unanalyzed(
            name, dtype, reason, rule, message, diagnostics, omissions,
            severity="warn" if isinstance(error, EmptyTensorError) else "error",
        )

    def _analyze_sequential(
        self,
        layer_stats_consumer: LayerStatsConsumer,
        health: CheckpointHealth,
    ) -> AnalysisResult:
        bucket_aggregators: dict[str, StreamingGlobalAggregator] = {}
        diagnostics: list[DiagnosticFlag] = []
        layer_stats: list[LayerStats] = []
        omissions: list[TensorOmission] = []

        layer_iter = self._source.iter_events()
        if self._prefetch:
            layer_iter = PrefetchIterator(layer_iter)

        for event in layer_iter:
            if isinstance(event, TensorOmission):
                omissions.append(event)
                continue
            layer = event
            category = self._classify(layer)
            if category == "skip":
                omissions.append(
                    TensorOmission(
                        name=layer.name, kind="excluded",
                        reason="excluded by parameter classifier", dtype=layer.dtype,
                    )
                )
                continue

            try:
                stats = self._stats_engine.compute_layer(layer)
            except (EmptyTensorError, NonFiniteTensorError) as error:
                self._record_stats_error(
                    layer.name, layer.dtype, error, diagnostics, omissions
                )
                continue

            stats = stats.model_copy(update={"category": category})
            count = stats.param_count
            mean = stats.mean
            variance = stats.std**2
            if not math.isfinite(mean) or not math.isfinite(variance):
                self._record_unanalyzed(
                    layer.name, layer.dtype, "non-finite computed statistics",
                    "non-finite-statistics",
                    "Computed statistics are non-finite. Statistics skipped.",
                    diagnostics, omissions,
                )
                continue
            self._aggregator.update_from_summary(
                count=count,
                mean=mean,
                variance=variance,
                quantile_summary=stats.quantile_summary,
                values=layer.values,
            )
            layer_stats.append(stats)
            layer_stats_consumer.update_layer_stats(stats)

            if category not in bucket_aggregators:
                bucket_aggregators[category] = self._create_bucket_aggregator()
            bucket_agg = bucket_aggregators[category]
            bucket_agg.update_from_summary(
                count=count,
                mean=mean,
                variance=variance,
                quantile_summary=stats.quantile_summary,
                values=layer.values,
            )
            bucket_agg.update_layer_stats(stats)

        return self._finalize(
            layer_stats, diagnostics, omissions, bucket_aggregators, health
        )

    def _analyze_parallel(
        self,
        layer_stats_consumer: LayerStatsConsumer,
        health: CheckpointHealth,
        num_workers: int,
    ) -> AnalysisResult:
        bucket_aggregators: dict[str, StreamingGlobalAggregator] = {}
        diagnostics: list[DiagnosticFlag] = []
        layer_stats: list[LayerStats] = []
        omissions: list[TensorOmission] = []

        # Bounded window of futures to cap memory usage
        pending: deque[tuple[str, str, NDArray[np.number], Future[LayerStats]]] = (
            deque()
        )

        layer_iter = self._source.iter_events()
        if self._prefetch:
            layer_iter = PrefetchIterator(layer_iter)

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for event in layer_iter:
                if isinstance(event, TensorOmission):
                    omissions.append(event)
                    continue
                layer = event
                category = self._classify(layer)
                if category == "skip":
                    omissions.append(
                        TensorOmission(
                            name=layer.name, kind="excluded",
                            reason="excluded by parameter classifier",
                            dtype=layer.dtype,
                        )
                    )
                    continue

                future = pool.submit(self._stats_engine.compute_layer, layer)
                pending.append((layer.name, category, layer.values, future))

                # Drain completed futures from the front to bound memory
                while len(pending) > num_workers and pending[0][3].done():
                    self._drain_one(
                        pending,
                        layer_stats,
                        diagnostics,
                        omissions,
                        bucket_aggregators,
                        layer_stats_consumer,
                    )

            # Drain remaining futures in submission order
            while pending:
                self._drain_one(
                    pending,
                    layer_stats,
                    diagnostics,
                    omissions,
                    bucket_aggregators,
                    layer_stats_consumer,
                )

        return self._finalize(
            layer_stats, diagnostics, omissions, bucket_aggregators, health
        )

    def _drain_one(
        self,
        pending: deque[tuple[str, str, NDArray[np.number], Future[LayerStats]]],
        layer_stats: list[LayerStats],
        diagnostics: list[DiagnosticFlag],
        omissions: list[TensorOmission],
        bucket_aggregators: dict[str, StreamingGlobalAggregator],
        layer_stats_consumer: LayerStatsConsumer,
    ) -> None:
        name, category, values, future = pending.popleft()
        try:
            stats = future.result()
        except (EmptyTensorError, NonFiniteTensorError) as error:
            self._record_stats_error(
                name, str(values.dtype), error, diagnostics, omissions
            )
            return

        stats = stats.model_copy(update={"category": category})

        count = stats.param_count
        mean = stats.mean
        variance = stats.std**2
        if not math.isfinite(mean) or not math.isfinite(variance):
            self._record_unanalyzed(
                name, str(values.dtype), "non-finite computed statistics",
                "non-finite-statistics",
                "Computed statistics are non-finite. Statistics skipped.",
                diagnostics, omissions,
            )
            return
        self._aggregator.update_from_summary(
            count=count,
            mean=mean,
            variance=variance,
            quantile_summary=stats.quantile_summary,
            values=values,
        )
        layer_stats.append(stats)
        layer_stats_consumer.update_layer_stats(stats)

        if category not in bucket_aggregators:
            bucket_aggregators[category] = self._create_bucket_aggregator()
        bucket_agg = bucket_aggregators[category]
        bucket_agg.update_from_summary(
            count=count,
            mean=mean,
            variance=variance,
            quantile_summary=stats.quantile_summary,
            values=values,
        )
        bucket_agg.update_layer_stats(stats)

    def _finalize(
        self,
        layer_stats: list[LayerStats],
        diagnostics: list[DiagnosticFlag],
        omissions: list[TensorOmission],
        bucket_aggregators: dict[str, StreamingGlobalAggregator],
        health: CheckpointHealth,
    ) -> AnalysisResult:
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
            health=health,
            coverage=AnalysisCoverage(
                analyzed=len(layer_stats),
                eligible=len(layer_stats)
                + sum(item.kind == "skipped" for item in omissions),
                omissions=omissions,
            ),
        )
