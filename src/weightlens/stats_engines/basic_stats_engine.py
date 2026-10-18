from __future__ import annotations

import logging
import math

import numpy as np

from weightlens.contracts import StatsEngine
from weightlens.models import LayerStats, LayerTensor
from weightlens.stats_engines.errors import EmptyTensorError, NonFiniteTensorError
from weightlens.stats_engines.signed_log_quantiles import SignedLogQuantiles

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1_000_000


class BasicStatsEngine(StatsEngine):
    """Compute per-layer statistics using chunked streaming algorithms.

    Mean and variance are computed via Welford's online algorithm.
    Signed and absolute quantiles use relative-error logarithmic buckets.
    """

    def compute_layer(self, layer: LayerTensor) -> LayerStats:
        values = layer.values
        param_count = int(values.size)
        if param_count == 0:
            logger.error("Layer %s is empty.", layer.name)
            raise EmptyTensorError(f"Layer {layer.name} is empty.")

        flat = values.ravel()

        if not np.isfinite(flat).all():
            logger.error("Layer %s contains NaN or Inf values.", layer.name)
            raise NonFiniteTensorError(
                f"Layer {layer.name} contains NaN or Inf values."
            )

        # --- single fused pass over chunks ---
        welford_count = 0
        welford_mean = 0.0
        welford_m2 = 0.0
        quantiles = SignedLogQuantiles()
        min_val = float("inf")
        max_val = float("-inf")
        nonzero = 0

        input_is_f16 = values.dtype == np.float16

        for start in range(0, param_count, _CHUNK_SIZE):
            chunk = flat[start : start + _CHUNK_SIZE]
            n = int(chunk.size)

            if input_is_f16:
                chunk_w = chunk.astype(np.float64)
            else:
                chunk_w = chunk.astype(np.float64, copy=False)

            # Welford update (float64)
            delta = chunk_w - welford_mean
            new_total = welford_count + n
            welford_mean += float(np.sum(delta, dtype=np.float64)) / new_total
            welford_m2 += float(
                np.sum(delta * (chunk_w - welford_mean), dtype=np.float64)
            )
            welford_count += n

            quantiles.update(chunk_w)

            # Min / max / nonzero tracking
            cmin = float(chunk_w.min())
            cmax = float(chunk_w.max())
            if cmin < min_val:
                min_val = cmin
            if cmax > max_val:
                max_val = cmax
            nonzero += int(np.count_nonzero(chunk))

        # --- derive scalar stats ---
        variance = max(0.0, welford_m2 / welford_count)
        std = math.sqrt(variance)
        mean = welford_mean
        sum_sq = (variance + mean**2) * param_count
        l2_norm = math.sqrt(max(0.0, sum_sq))
        sparsity = 1.0 - (nonzero / param_count)
        p99_abs = quantiles.abs_quantile(0.99)

        logger.debug(
            "Computed stats for %s: mean=%.6f std=%.6f min=%.6f max=%.6f "
            "l2_norm=%.6f sparsity=%.6f p99_abs=%.6f.",
            layer.name,
            mean,
            std,
            min_val,
            max_val,
            l2_norm,
            sparsity,
            p99_abs,
        )

        return LayerStats(
            name=layer.name,
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            l2_norm=l2_norm,
            sparsity=sparsity,
            param_count=param_count,
            p99_abs=p99_abs,
            quantile_summary=quantiles.summary(),
        )
