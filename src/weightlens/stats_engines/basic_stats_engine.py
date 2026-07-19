from __future__ import annotations

import logging
import math

import numpy as np

from weightlens.contracts import StatsEngine
from weightlens.models import LayerStats, LayerTensor

logger = logging.getLogger(__name__)

_HISTOGRAM_BINS = 4096
_HISTOGRAM_MIN = -100.0
_HISTOGRAM_MAX = 100.0
_CHUNK_SIZE = 1_000_000


class BasicStatsEngine(StatsEngine):
    """Compute per-layer statistics using chunked streaming algorithms.

    Mean and variance are computed via Welford's online algorithm.
    Histogram bins are accumulated incrementally across chunks.
    P99 is estimated from the histogram (avoids O(n log n) quantile).
    """

    def compute_layer(self, layer: LayerTensor) -> LayerStats:
        values = layer.values
        param_count = int(values.size)
        if param_count == 0:
            logger.error("Layer %s is empty.", layer.name)
            raise ValueError(f"Layer {layer.name} is empty.")

        flat = values.ravel()

        if not np.isfinite(flat).all():
            logger.error("Layer %s contains NaN or Inf values.", layer.name)
            raise ValueError(f"Layer {layer.name} contains NaN values.")

        # --- single fused pass over chunks ---
        welford_count = 0
        welford_mean = 0.0
        welford_m2 = 0.0
        hist_acc = np.zeros(_HISTOGRAM_BINS, dtype=np.float64)
        hist_under = 0
        hist_over = 0
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

            # Histogram accumulation
            h32 = chunk_w if not input_is_f16 else chunk.astype(np.float32)
            try:
                h, _ = np.histogram(
                    h32, bins=_HISTOGRAM_BINS, range=(_HISTOGRAM_MIN, _HISTOGRAM_MAX)
                )
            except ValueError:
                data_min = float(h32.min())
                data_max = float(h32.max())
                margin = max(1e-6, (data_max - data_min) * 0.01)
                h, _ = np.histogram(
                    h32,
                    bins=min(_HISTOGRAM_BINS, n),
                    range=(data_min - margin, data_max + margin),
                )
            hist_acc += h
            hist_under += int(np.sum(chunk_w < _HISTOGRAM_MIN))
            hist_over += int(np.sum(chunk_w > _HISTOGRAM_MAX))

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
        max_abs = max(abs(min_val), abs(max_val))
        p99_abs = self._histogram_p99(hist_acc, hist_under + hist_over, max_abs)

        histogram_counts = [float(c) for c in hist_acc]

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
            histogram_counts=histogram_counts,
            histogram_underflow=hist_under,
            histogram_overflow=hist_over,
        )

    @staticmethod
    def _histogram_p99(
        hist_counts: np.ndarray, overflow: int, max_abs_val: float
    ) -> float:
        """Estimate p99 of |x| from signed histogram + overflow count.

        Approximates ``np.quantile(np.abs(tensor), 0.99)`` using the
        signed histogram bins.  Values beyond ±100 (overflow) are
        assumed uniformly distributed up to 5× the overflow threshold.
        """
        total = float(np.sum(hist_counts))
        if total == 0:
            return 0.0

        # Fold signed histogram into |x| histogram
        half = len(hist_counts) // 2
        abs_hist = np.zeros(half + 1, dtype=np.float64)
        bin_width = (_HISTOGRAM_MAX - _HISTOGRAM_MIN) / len(hist_counts)

        for i in range(len(hist_counts)):
            center = abs(_HISTOGRAM_MIN + (i + 0.5) * bin_width)
            idx = min(int(center / bin_width), half)
            abs_hist[idx] += hist_counts[i]

        # Add overflow as an extra bin
        total_with_overflow = total + overflow
        target = 0.99 * total_with_overflow
        cumsum = 0.0
        for i in range(len(abs_hist)):
            cumsum += abs_hist[i]
            if cumsum >= target:
                if abs_hist[i] == 0:
                    frac = 0.5
                else:
                    frac = (target - (cumsum - abs_hist[i])) / abs_hist[i]
                return min((i + frac) * bin_width, max_abs_val)

        # Fallback: target is in overflow tail
        return max_abs_val
