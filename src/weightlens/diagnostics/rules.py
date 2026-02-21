from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Literal

from weightlens.contracts import DiagnosticRule
from weightlens.models import DiagnosticFlag, GlobalStats, LayerStats

if TYPE_CHECKING:
    from weightlens.categories import ParameterCategory

logger = logging.getLogger(__name__)


class DeadLayerRule(DiagnosticRule):
    """Detect layers with near-total exact zeros."""

    @property
    def severity(self) -> Literal["info", "warn", "error"]:
        return "error"

    @property
    def applicable_categories(self) -> frozenset[ParameterCategory]:
        return frozenset(["kernel", "embedding"])

    def check(
        self, layer: LayerStats, global_stats: GlobalStats
    ) -> DiagnosticFlag | None:
        _ = global_stats
        dead_fraction = float(layer.sparsity)
        if dead_fraction >= 0.9999:
            flag = DiagnosticFlag(
                layer=layer.name,
                rule="dead-layer",
                message=f"dead_fraction={dead_fraction:.6f} >= 0.9999",
                severity=self.severity,
            )
            logger.debug(
                "Diagnostic rule %s triggered for %s: %s.",
                flag.rule,
                flag.layer,
                flag.message,
            )
            return flag
        return None


class ExplodingVarianceRule(DiagnosticRule):
    """Detect layers with variance far above the median variance."""

    @property
    def severity(self) -> Literal["info", "warn", "error"]:
        return "warn"

    @property
    def applicable_categories(self) -> frozenset[ParameterCategory]:
        return frozenset(["kernel", "embedding", "adapter"])

    def check(
        self, layer: LayerStats, global_stats: GlobalStats
    ) -> DiagnosticFlag | None:
        variance = float(layer.std) ** 2
        median_variance = float(global_stats.median_layer_variance)
        if (
            not math.isfinite(variance)
            or not math.isfinite(median_variance)
            or median_variance <= 0.0
        ):
            logger.debug(
                "Skipping exploding-variance for %s due to invalid denominator "
                "(variance=%s median_variance=%s).",
                layer.name,
                variance,
                median_variance,
            )
            return None

        ratio = variance / median_variance
        if ratio >= 10.0:
            flag = DiagnosticFlag(
                layer=layer.name,
                rule="exploding-variance",
                message=f"variance_ratio={ratio:.3f} >= 10.0",
                severity=self.severity,
            )
            logger.debug(
                "Diagnostic rule %s triggered for %s: %s.",
                flag.rule,
                flag.layer,
                flag.message,
            )
            return flag
        return None


class ExtremeSpikeRule(DiagnosticRule):
    """Detect extreme max spikes relative to the 99th percentile scale."""

    @property
    def severity(self) -> Literal["info", "warn", "error"]:
        return "error"

    @property
    def applicable_categories(self) -> frozenset[ParameterCategory]:
        return frozenset(["kernel", "embedding"])

    def check(
        self, layer: LayerStats, global_stats: GlobalStats
    ) -> DiagnosticFlag | None:
        _ = global_stats
        max_abs = max(abs(float(layer.min)), abs(float(layer.max)))
        p99_abs = float(layer.p99_abs)
        if (
            not math.isfinite(max_abs)
            or not math.isfinite(p99_abs)
            or p99_abs <= 0.0
        ):
            logger.debug(
                "Skipping extreme-spike for %s due to invalid denominator "
                "(max_abs=%s p99_abs=%s).",
                layer.name,
                max_abs,
                p99_abs,
            )
            return None

        ratio = max_abs / p99_abs
        if ratio >= 100.0:
            flag = DiagnosticFlag(
                layer=layer.name,
                rule="extreme-spike",
                message=f"spike_ratio={ratio:.3f} >= 100.0",
                severity=self.severity,
            )
            logger.debug(
                "Diagnostic rule %s triggered for %s: %s.",
                flag.rule,
                flag.layer,
                flag.message,
            )
            return flag
        return None


class AbnormalNormRule(DiagnosticRule):
    """Detect norms far from the median using IQR scaling."""

    @property
    def severity(self) -> Literal["info", "warn", "error"]:
        return "warn"

    @property
    def applicable_categories(self) -> frozenset[ParameterCategory]:
        return frozenset(["kernel", "embedding", "adapter"])

    def check(
        self, layer: LayerStats, global_stats: GlobalStats
    ) -> DiagnosticFlag | None:
        norm = float(layer.l2_norm)
        median_norm = float(global_stats.median_layer_norm)
        iqr_norm = float(global_stats.iqr_layer_norm)
        if (
            not math.isfinite(norm)
            or not math.isfinite(median_norm)
            or not math.isfinite(iqr_norm)
            or iqr_norm <= 0.0
        ):
            logger.debug(
                "Skipping abnormal-norm for %s due to invalid denominator "
                "(norm=%s median_norm=%s iqr_norm=%s).",
                layer.name,
                norm,
                median_norm,
                iqr_norm,
            )
            return None

        z_score = (norm - median_norm) / iqr_norm
        if abs(z_score) >= 5.0:
            flag = DiagnosticFlag(
                layer=layer.name,
                rule="abnormal-norm",
                message=f"norm_z_score={z_score:.3f} exceeds +/-5.0",
                severity=self.severity,
            )
            logger.debug(
                "Diagnostic rule %s triggered for %s: %s.",
                flag.rule,
                flag.layer,
                flag.message,
            )
            return flag
        return None
