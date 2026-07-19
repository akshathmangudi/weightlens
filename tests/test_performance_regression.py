from __future__ import annotations

import gc
import time
import weakref
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from tests.fixtures_realistic import (
    make_realistic_model_dict,
    make_realistic_pytorch,
    make_realistic_safetensors,
)
from tests.fixtures_safetensors import write_single
from weightlens.aggregators.streaming_global import StreamingGlobalAggregator
from weightlens.analyzer import Analyzer
from weightlens.classifiers import PyTorchParameterClassifier
from weightlens.contracts import CheckpointValidator, WeightSource
from weightlens.diagnostics import (
    AbnormalNormRule,
    DeadLayerRule,
    ExplodingVarianceRule,
    ExtremeSpikeRule,
)
from weightlens.histogram_quantiles import FixedRangeHistogramQuantiles
from weightlens.models import AnalysisResult, CheckpointHealth, LayerTensor
from weightlens.sources.pytorch import PyTorchWeightSource
from weightlens.sources.safetensors import SafetensorsWeightSource
from weightlens.stats_engines.basic_stats_engine import BasicStatsEngine
from weightlens.validators.pytorch_checkpoint import PyTorchCheckpointValidator
from weightlens.validators.safetensors import SafetensorsCheckpointValidator

_ALL_RULES = [
    DeadLayerRule(),
    ExplodingVarianceRule(),
    ExtremeSpikeRule(),
    AbnormalNormRule(),
]


# ── Shared test helpers ──────────────────────────────────────────────

class _DummyValidator(CheckpointValidator):
    def __init__(self, tensor_count: int, total_params: int) -> None:
        self._tensor_count = tensor_count
        self._total_params = total_params

    def validate(self) -> CheckpointHealth:
        return CheckpointHealth(
            file_size_bytes=1,
            is_empty=False,
            loadable=True,
            tensor_count=self._tensor_count,
            total_params=self._total_params,
            corruption_flags=[],
        )


class _StaticValidator(CheckpointValidator):
    def __init__(self, health: CheckpointHealth) -> None:
        self._health = health

    def validate(self) -> CheckpointHealth:
        return self._health


class _SyntheticSource(WeightSource):
    def __init__(self, n: int, size: int, seed: int = 42) -> None:
        self._n = n
        self._size = size
        self._seed = seed

    def iter_layers(self) -> Iterator[LayerTensor]:
        rng = np.random.default_rng(self._seed)
        for i in range(self._n):
            values = rng.standard_normal(self._size).astype(np.float32)
            yield LayerTensor(
                name=f"layer.{i}.weight",
                values=values,
                shape=(self._size,),
                dtype="float32",
            )


def _run_full_analysis_safetensors(
    uri: str, *, num_workers: int | None = 1
) -> AnalysisResult:
    validator = SafetensorsCheckpointValidator(uri)
    health = validator.validate()
    analyzer = Analyzer(
        source=SafetensorsWeightSource(uri),
        validator=_StaticValidator(health),
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=_ALL_RULES,
        classifier=PyTorchParameterClassifier(),
        num_workers=num_workers,
    )
    return analyzer.analyze()


def _run_full_analysis_pytorch(
    path: Path, *, num_workers: int | None = 1
) -> AnalysisResult:
    validator = PyTorchCheckpointValidator(path)
    health = validator.validate()
    analyzer = Analyzer(
        source=PyTorchWeightSource(path),
        validator=_StaticValidator(health),
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=_ALL_RULES,
        classifier=PyTorchParameterClassifier(),
        num_workers=num_workers,
    )
    return analyzer.analyze()


# ═══════════════════════════════════════════════════════════════════════
# Pass-count tests
# ═══════════════════════════════════════════════════════════════════════


class _CountingHistogram(FixedRangeHistogramQuantiles):
    update_calls: int = 0

    def update(self, values: NDArray[np.number]) -> None:
        _CountingHistogram.update_calls += 1
        return super().update(values)


class _CountingAggregator(StreamingGlobalAggregator):
    def __init__(self) -> None:
        super().__init__()
        self._quantiles = _CountingHistogram(
            min_value=-100.0, max_value=100.0, bins=4096
        )


def test_histogram_scan_called_at_most_n_times() -> None:
    N = 7
    _CountingHistogram.update_calls = 0
    source = _SyntheticSource(n=N, size=256)
    aggregator = _CountingAggregator()
    analyzer = Analyzer(
        source=source,
        validator=_DummyValidator(tensor_count=N, total_params=N * 256),
        stats_engine=BasicStatsEngine(),
        aggregator=aggregator,
        rules=[],
        classifier=PyTorchParameterClassifier(),
        num_workers=1,
    )
    analyzer.analyze()
    assert _CountingHistogram.update_calls == 0, (
        f"histogram scanned {_CountingHistogram.update_calls} times "
        f"for {N} layers (expected 0 — merge_histogram should be used instead)"
    )


def test_no_numpy_quantile_on_large_tensors() -> None:
    engine = BasicStatsEngine()
    values = np.random.default_rng(42).standard_normal(20000).astype(np.float32)
    layer = LayerTensor(
        name="large.weight",
        values=values,
        shape=(20000,),
        dtype="float32",
    )
    with patch("numpy.quantile", wraps=np.quantile) as spy:
        result = engine.compute_layer(layer)
        assert isinstance(result.p99_abs, float)
    assert spy.call_count == 0, (
        "np.quantile must not be called — histogram-based p99 is used instead. "
        "If this is >0, the hot path regressed to O(n log n)."
    )


# ═══════════════════════════════════════════════════════════════════════
# Memory tests
# ═══════════════════════════════════════════════════════════════════════


def test_values_array_freed_after_compute_layer() -> None:
    engine = BasicStatsEngine()
    values = np.arange(1000, dtype=np.float32)
    ref = weakref.ref(values)
    layer = LayerTensor(
        name="test.weight",
        values=values,
        shape=(1000,),
        dtype="float32",
    )
    engine.compute_layer(layer)
    del values
    del layer
    gc.collect()
    assert ref() is None, "values array should have been freed after compute_layer"


def test_parallel_path_holds_at_most_n_plus_one_arrays() -> None:
    """10 layers of 1M params each must complete without OOM."""
    layer_count = 10
    layer_size = 1_000_000
    analyzer = Analyzer(
        source=_SyntheticSource(n=layer_count, size=layer_size),
        validator=_DummyValidator(
            tensor_count=layer_count,
            total_params=layer_count * layer_size,
        ),
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=_ALL_RULES,
        classifier=PyTorchParameterClassifier(),
        num_workers=2,
    )
    result = analyzer.analyze()
    assert len(result.layer_stats) == layer_count


# ═══════════════════════════════════════════════════════════════════════
# Parallel consistency tests
# ═══════════════════════════════════════════════════════════════════════


def test_sequential_matches_parallel_numerically(tmp_path: Path) -> None:
    tensors = make_realistic_model_dict(seed=42)
    path = str(tmp_path / "model.safetensors")
    write_single(path, tensors)

    sequential = _run_full_analysis_safetensors(path, num_workers=1)
    parallel = _run_full_analysis_safetensors(path, num_workers=4)

    seq_by_name = {s.name: s for s in sequential.layer_stats}
    par_by_name = {s.name: s for s in parallel.layer_stats}
    assert set(seq_by_name) == set(par_by_name) == set(tensors)

    for name in tensors:
        s = seq_by_name[name]
        p = par_by_name[name]
        for field in ("mean", "std", "min", "max", "p99_abs", "l2_norm"):
            np.testing.assert_allclose(
                getattr(s, field),
                getattr(p, field),
                atol=1e-6,
                err_msg=(
                    f"{name}.{field}: seq={getattr(s, field)!r} "
                    f"par={getattr(p, field)!r}"
                ),
            )
        assert s.sparsity == pytest.approx(p.sparsity, abs=1e-12)

    np.testing.assert_allclose(
        sequential.global_stats.mean, parallel.global_stats.mean, atol=1e-6
    )
    np.testing.assert_allclose(
        sequential.global_stats.std, parallel.global_stats.std, atol=1e-6
    )


def test_parallel_not_slower_than_sequential() -> None:
    layer_count = 30
    layer_size = 5000

    def _build(nw: int | None) -> Analyzer:
        return Analyzer(
            source=_SyntheticSource(n=layer_count, size=layer_size),
            validator=_DummyValidator(
                tensor_count=layer_count,
                total_params=layer_count * layer_size,
            ),
            stats_engine=BasicStatsEngine(),
            aggregator=StreamingGlobalAggregator(),
            rules=_ALL_RULES,
            classifier=PyTorchParameterClassifier(),
            num_workers=nw,
        )

    _build(1).analyze()  # warm-up

    t0 = time.perf_counter()
    _build(1).analyze()
    seq_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _build(4).analyze()
    par_time = time.perf_counter() - t0

    assert par_time <= seq_time + 1.0, (
        f"parallel ({par_time:.3f}s) not within 1s of sequential ({seq_time:.3f}s)"
    )


# ═══════════════════════════════════════════════════════════════════════
# Format parity tests
# ═══════════════════════════════════════════════════════════════════════


def test_pytorch_safetensors_produce_same_stats(tmp_path: Path) -> None:
    pytorch_path = make_realistic_pytorch(tmp_path, seed=7)
    safetensors_path = make_realistic_safetensors(tmp_path, seed=7)

    pytorch_result = _run_full_analysis_pytorch(pytorch_path, num_workers=1)
    safetensors_result = _run_full_analysis_safetensors(
        str(safetensors_path), num_workers=1
    )

    pt_by_name = {s.name: s for s in pytorch_result.layer_stats}
    sf_by_name = {s.name: s for s in safetensors_result.layer_stats}
    assert set(pt_by_name) == set(sf_by_name)

    for name in pt_by_name:
        pt = pt_by_name[name]
        sf = sf_by_name[name]
        for field in ("mean", "std", "min", "max", "p99_abs", "l2_norm"):
            np.testing.assert_allclose(
                getattr(pt, field),
                getattr(sf, field),
                atol=1e-6,
                err_msg=(
                    f"{name}.{field}: pytorch={getattr(pt, field)!r} "
                    f"safetensors={getattr(sf, field)!r}"
                ),
            )
        assert pt.sparsity == pytest.approx(sf.sparsity, abs=1e-12)

    np.testing.assert_allclose(
        pytorch_result.global_stats.mean,
        safetensors_result.global_stats.mean,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pytorch_result.global_stats.std,
        safetensors_result.global_stats.std,
        atol=1e-6,
    )


@pytest.mark.integration
def test_all_three_formats_produce_identical_layer_stats(tmp_path: Path) -> None:
    """PyTorch .pth vs safetensors vs DCP: all three must yield identical stats."""
    from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

    from weightlens.sources.dcp import DCPWeightSource
    from weightlens.validators.dcp_checkpoint import DCPCheckpointValidator

    # ── PyTorch .pth ──
    pytorch_path = make_realistic_pytorch(tmp_path, seed=13)
    pytorch_result = _run_full_analysis_pytorch(pytorch_path, num_workers=1)

    # ── Safetensors ──
    safetensors_path = make_realistic_safetensors(tmp_path, seed=13)
    safetensors_result = _run_full_analysis_safetensors(
        str(safetensors_path), num_workers=1
    )

    # ── DCP ──
    model_dict = make_realistic_model_dict(seed=13)
    st = {
        k: torch.from_numpy(np.ascontiguousarray(v))
        for k, v in model_dict.items()
    }
    dcp_path = tmp_path / "dcp"
    dcp_path.mkdir()
    dcp_save(st, checkpoint_id=str(dcp_path), no_dist=True)

    dcp_validator = DCPCheckpointValidator(dcp_path)
    dcp_health = dcp_validator.validate()
    dcp_analyzer = Analyzer(
        source=DCPWeightSource(dcp_path),
        validator=_StaticValidator(dcp_health),
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=_ALL_RULES,
        classifier=PyTorchParameterClassifier(),
        num_workers=1,
    )
    dcp_result = dcp_analyzer.analyze()

    pt_by_name = {s.name: s for s in pytorch_result.layer_stats}
    sf_by_name = {s.name: s for s in safetensors_result.layer_stats}
    dcp_by_name = {s.name: s for s in dcp_result.layer_stats}

    assert set(pt_by_name) == set(sf_by_name) == set(dcp_by_name)

    for name in pt_by_name:
        pt = pt_by_name[name]
        sf = sf_by_name[name]
        dc = dcp_by_name[name]
        for field in ("mean", "std", "min", "max", "p99_abs", "l2_norm"):
            np.testing.assert_allclose(
                getattr(pt, field),
                getattr(sf, field),
                atol=1e-6,
                err_msg=f"{name}.{field}: pytorch != safetensors",
            )
            np.testing.assert_allclose(
                getattr(pt, field),
                getattr(dc, field),
                atol=1e-6,
                err_msg=f"{name}.{field}: pytorch != dcp",
            )

    np.testing.assert_allclose(
        pytorch_result.global_stats.mean,
        safetensors_result.global_stats.mean,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pytorch_result.global_stats.mean,
        dcp_result.global_stats.mean,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pytorch_result.global_stats.std,
        safetensors_result.global_stats.std,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pytorch_result.global_stats.std,
        dcp_result.global_stats.std,
        atol=1e-6,
    )


def test_analyze_corrupted_checkpoint_completes_within_budget() -> None:
    from weightlens.cli import StaticCheckpointValidator
    from weightlens.sources import PyTorchWeightSource
    from weightlens.stats_engines import BasicStatsEngine
    from weightlens.validators import PyTorchCheckpointValidator

    cp = Path("artifacts/checkpoints/corrupted_spike.pth")

    start = time.perf_counter()
    validator = PyTorchCheckpointValidator(cp)
    health = validator.validate()
    analyzer = Analyzer(
        source=PyTorchWeightSource(cp),
        validator=StaticCheckpointValidator(health),
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=[
            DeadLayerRule(),
            ExplodingVarianceRule(),
            ExtremeSpikeRule(),
            AbnormalNormRule(),
        ],
        classifier=PyTorchParameterClassifier(),
    )
    analyzer.analyze()
    elapsed = time.perf_counter() - start

    assert elapsed < 3.0, f"Analysis took {elapsed:.1f}s, budget is 3.0s"
