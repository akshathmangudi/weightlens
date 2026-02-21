from __future__ import annotations

import numpy as np

from weightlens.aggregators.streaming_global import StreamingGlobalAggregator
from weightlens.models import LayerStats


def _make_layer_stats(std: float, l2_norm: float, name: str) -> LayerStats:
    return LayerStats(
        name=name,
        mean=0.0,
        std=std,
        min=0.0,
        max=0.0,
        l2_norm=l2_norm,
        sparsity=0.0,
        param_count=1,
        p99_abs=0.0,
    )


def test_update_from_summary_matches_update() -> None:
    """update() and update_from_summary() must produce identical GlobalStats."""
    rng = np.random.default_rng(42)
    arrays = [rng.standard_normal(500).astype(np.float64) for _ in range(5)]
    layer_stds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    layer_norms = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    # Reference: use update()
    ref = StreamingGlobalAggregator()
    for arr in arrays:
        ref.update(arr)
    for i, (s, n) in enumerate(zip(layer_stds, layer_norms, strict=True)):
        ref.update_layer_stats(_make_layer_stats(std=s, l2_norm=n, name=f"l{i}"))
    ref_stats = ref.finalize()

    # Test: use update_from_summary()
    opt = StreamingGlobalAggregator()
    for arr in arrays:
        count = int(arr.size)
        mean = float(np.mean(arr))
        variance = float(np.var(arr, ddof=0))
        opt.update_from_summary(arr, count=count, mean=mean, variance=variance)
    for i, (s, n) in enumerate(zip(layer_stds, layer_norms, strict=True)):
        opt.update_layer_stats(_make_layer_stats(std=s, l2_norm=n, name=f"l{i}"))
    opt_stats = opt.finalize()

    np.testing.assert_allclose(opt_stats.mean, ref_stats.mean, atol=1e-12)
    np.testing.assert_allclose(opt_stats.std, ref_stats.std, atol=1e-12)
    np.testing.assert_allclose(opt_stats.p1, ref_stats.p1, atol=1e-10)
    np.testing.assert_allclose(opt_stats.p50, ref_stats.p50, atol=1e-10)
    np.testing.assert_allclose(opt_stats.p99, ref_stats.p99, atol=1e-10)
    np.testing.assert_allclose(
        opt_stats.median_layer_variance,
        ref_stats.median_layer_variance,
        atol=1e-12,
    )


def test_update_from_summary_skips_empty() -> None:
    agg = StreamingGlobalAggregator()
    empty = np.array([], dtype=np.float64)
    agg.update_from_summary(empty, count=0, mean=0.0, variance=0.0)
    assert agg._count == 0
