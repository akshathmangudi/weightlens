from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from tests.fixtures_safetensors import write_sharded, write_single
from weightlens.aggregators import StreamingGlobalAggregator
from weightlens.analyzer import Analyzer
from weightlens.classifiers import PyTorchParameterClassifier
from weightlens.contracts import CheckpointValidator
from weightlens.diagnostics import (
    AbnormalNormRule,
    DeadLayerRule,
    ExplodingVarianceRule,
    ExtremeSpikeRule,
)
from weightlens.models import AnalysisResult, CheckpointHealth, LayerStats
from weightlens.sources.safetensors import SafetensorsWeightSource
from weightlens.stats_engines import BasicStatsEngine
from weightlens.validators.safetensors import SafetensorsCheckpointValidator

# Golden-oracle suite: streamed analysis == full-local-load analysis.
# Tests that streaming map-reduce pipeline produces numerically identical
# results to naive "load everything + crunch with plain numpy" approach.

MEAN_STD_RTOL = 1e-5
MEAN_STD_ATOL = 1e-8
LAYER_FIELD_RTOL = 1e-5
LAYER_FIELD_ATOL = 1e-8


class _StaticValidator(CheckpointValidator):
    """CheckpointValidator that reuses a precomputed health result."""

    def __init__(self, health: CheckpointHealth) -> None:
        self._health = health

    def validate(self) -> CheckpointHealth:
        return self._health


def _run_full_analysis(uri: str, *, num_workers: int | None = 1) -> AnalysisResult:
    """Run the real Analyzer wired exactly like ``_run_analyze_safetensors``."""
    validator = SafetensorsCheckpointValidator(uri)
    health = validator.validate()
    assert health.loadable
    assert not health.is_empty

    analyzer = Analyzer(
        source=SafetensorsWeightSource(uri),
        validator=_StaticValidator(health),
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=[
            DeadLayerRule(),
            ExplodingVarianceRule(),
            ExtremeSpikeRule(),
            AbnormalNormRule(),
        ],
        classifier=PyTorchParameterClassifier(),
        num_workers=num_workers,
    )
    return analyzer.analyze()


def _oracle_layer_stats(name: str, values: np.ndarray) -> dict[str, float]:
    """Compute per-layer stats independently with plain numpy."""
    _ = name
    flat = values.astype(np.float64).ravel()
    count = flat.size
    mean = float(np.mean(flat))
    std = float(np.std(flat, ddof=0))
    minimum = float(np.min(flat))
    maximum = float(np.max(flat))
    l2_norm = float(np.sqrt(np.sum(flat * flat)))
    sparsity = 1.0 - (int(np.count_nonzero(flat)) / count)
    p99_abs = float(np.quantile(np.abs(flat), 0.99, method="linear"))
    return {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "l2_norm": l2_norm,
        "sparsity": sparsity,
        "param_count": float(count),
        "p99_abs": p99_abs,
    }


def _oracle_global_mean_std(all_values: list[np.ndarray]) -> tuple[float, float]:
    """Compute global mean/std over the full concatenation of every tensor."""
    flat = np.concatenate([v.astype(np.float64).ravel() for v in all_values])
    return float(np.mean(flat)), float(np.std(flat, ddof=0))


def _assert_global_mean_std_matches(
    global_mean: float,
    global_std: float,
    oracle_mean: float,
    oracle_std: float,
    *,
    abs_tol: float = MEAN_STD_ATOL,
) -> None:
    assert math.isclose(
        global_mean, oracle_mean, rel_tol=MEAN_STD_RTOL, abs_tol=abs_tol
    )
    assert math.isclose(
        global_std, oracle_std, rel_tol=MEAN_STD_RTOL, abs_tol=abs_tol
    )


def _assert_layer_matches_oracle(layer: LayerStats, oracle: dict[str, float]) -> None:
    assert layer.param_count == int(oracle["param_count"])
    for field in ("mean", "std", "min", "max", "l2_norm", "p99_abs"):
        got = getattr(layer, field)
        want = oracle[field]
        assert math.isclose(
            got, want, rel_tol=LAYER_FIELD_RTOL, abs_tol=LAYER_FIELD_ATOL
        ), f"{layer.name}.{field}: streamed={got!r} oracle={want!r}"
    assert layer.sparsity == pytest.approx(oracle["sparsity"], abs=1e-12)


def _make_realistic_tensors(seed: int) -> dict[str, np.ndarray]:
    """A realistic multi-layer, multi-category tensor set (all float32)."""
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {
        "embed.tok_embeddings.weight": rng.normal(0, 0.02, size=(64, 32)).astype(
            np.float32
        ),
        "layers.0.attn.q_proj.weight": rng.normal(0, 0.05, size=(32, 32)).astype(
            np.float32
        ),
        "layers.0.attn.q_proj.bias": rng.normal(0, 0.01, size=(32,)).astype(
            np.float32
        ),
        "layers.0.attn.norm.weight": np.ones((32,), dtype=np.float32),
        "layers.0.mlp.fc1.weight": rng.normal(0, 0.1, size=(128, 32)).astype(
            np.float32
        ),
        "layers.0.mlp.fc1.bias": rng.normal(0, 0.01, size=(128,)).astype(np.float32),
        "layers.1.attn.q_proj.weight": rng.uniform(-1, 1, size=(32, 32)).astype(
            np.float32
        ),
        "layers.1.mlp.fc2.weight": rng.normal(0, 0.5, size=(32, 128)).astype(
            np.float32
        ),
        "output.norm.weight": np.full((32,), 0.5, dtype=np.float32),
    }
    return tensors


def test_single_file_streamed_matches_full_load_oracle(tmp_path: Path) -> None:
    """Multi-tensor single-file checkpoint: streamed stats == numpy oracle."""
    tensors = _make_realistic_tensors(seed=1)
    path = str(tmp_path / "model.safetensors")
    write_single(path, tensors)

    result = _run_full_analysis(path)

    oracle_files = load_file(path)
    assert set(oracle_files) == set(tensors)

    by_name = {stats.name: stats for stats in result.layer_stats}
    assert set(by_name) == set(tensors)

    for name, arr in tensors.items():
        _assert_layer_matches_oracle(by_name[name], _oracle_layer_stats(name, arr))

    oracle_mean, oracle_std = _oracle_global_mean_std(list(tensors.values()))
    _assert_global_mean_std_matches(
        result.global_stats.mean, result.global_stats.std, oracle_mean, oracle_std
    )


def test_sharded_checkpoint_streamed_matches_full_load_oracle(
    tmp_path: Path,
) -> None:
    """Multi-shard HF-style checkpoint: streamed stats == numpy oracle."""
    rng = np.random.default_rng(2)
    shard_a: dict[str, np.ndarray] = {
        "layers.0.attn.q_proj.weight": rng.normal(0, 0.05, size=(40, 20)).astype(
            np.float32
        ),
        "layers.0.attn.q_proj.bias": rng.normal(0, 0.02, size=(40,)).astype(
            np.float32
        ),
        "layers.0.norm.weight": np.ones((40,), dtype=np.float32),
    }
    shard_b: dict[str, np.ndarray] = {
        "layers.1.mlp.fc1.weight": rng.normal(0, 0.3, size=(80, 40)).astype(
            np.float32
        ),
        "layers.1.mlp.fc1.bias": rng.normal(0, 0.01, size=(80,)).astype(np.float32),
    }
    shard_c: dict[str, np.ndarray] = {
        "embed.tok_embeddings.weight": rng.normal(0, 0.02, size=(100, 20)).astype(
            np.float32
        ),
        "output.norm.weight": np.full((20,), 0.7, dtype=np.float32),
    }
    all_shards = [shard_a, shard_b, shard_c]
    index_path = write_sharded(str(tmp_path / "ckpt"), all_shards)

    result = _run_full_analysis(index_path)

    merged: dict[str, np.ndarray] = {}
    for shard in all_shards:
        merged.update(shard)

    by_name = {stats.name: stats for stats in result.layer_stats}
    assert set(by_name) == set(merged)

    for name, arr in merged.items():
        _assert_layer_matches_oracle(by_name[name], _oracle_layer_stats(name, arr))

    oracle_mean, oracle_std = _oracle_global_mean_std(list(merged.values()))
    _assert_global_mean_std_matches(
        result.global_stats.mean, result.global_stats.std, oracle_mean, oracle_std
    )


def test_parallel_workers_match_sequential_and_oracle(tmp_path: Path) -> None:
    """num_workers > 1 must not change the numeric result vs. the oracle."""
    tensors = _make_realistic_tensors(seed=3)
    path = str(tmp_path / "model.safetensors")
    write_single(path, tensors)

    sequential = _run_full_analysis(path, num_workers=1)
    parallel = _run_full_analysis(path, num_workers=4)

    seq_by_name = {stats.name: stats for stats in sequential.layer_stats}
    par_by_name = {stats.name: stats for stats in parallel.layer_stats}
    assert set(seq_by_name) == set(par_by_name) == set(tensors)

    for name, arr in tensors.items():
        oracle = _oracle_layer_stats(name, arr)
        _assert_layer_matches_oracle(seq_by_name[name], oracle)
        _assert_layer_matches_oracle(par_by_name[name], oracle)

    assert math.isclose(
        sequential.global_stats.mean,
        parallel.global_stats.mean,
        rel_tol=MEAN_STD_RTOL,
        abs_tol=MEAN_STD_ATOL,
    )
    assert math.isclose(
        sequential.global_stats.std,
        parallel.global_stats.std,
        rel_tol=MEAN_STD_RTOL,
        abs_tol=MEAN_STD_ATOL,
    )


def test_extreme_magnitude_spread_matches_oracle(tmp_path: Path) -> None:
    """Tensors spanning many orders of magnitude must not desync Welford
    merge from the naive full-population computation."""
    rng = np.random.default_rng(4)
    tensors: dict[str, np.ndarray] = {
        "big.weight": (rng.normal(0, 1, size=(50, 50)) * 1e4).astype(np.float32),
        "tiny.weight": (rng.normal(0, 1, size=(50, 50)) * 1e-4).astype(np.float32),
        "mid.bias": rng.normal(0, 1, size=(200,)).astype(np.float32),
    }
    path = str(tmp_path / "extreme.safetensors")
    write_single(path, tensors)

    result = _run_full_analysis(path)
    by_name = {stats.name: stats for stats in result.layer_stats}

    for name, arr in tensors.items():
        _assert_layer_matches_oracle(by_name[name], _oracle_layer_stats(name, arr))

    oracle_mean, oracle_std = _oracle_global_mean_std(list(tensors.values()))
    _assert_global_mean_std_matches(
        result.global_stats.mean,
        result.global_stats.std,
        oracle_mean,
        oracle_std,
        abs_tol=1e-2,
    )


def test_sparse_layer_with_exact_zeros_matches_oracle(tmp_path: Path) -> None:
    """A layer with a known, exact sparsity fraction must match bit-for-bit."""
    values = np.zeros((10, 10), dtype=np.float32)
    # Set exactly 17 of the 100 entries nonzero -> sparsity is an exact ratio.
    rng = np.random.default_rng(5)
    flat_view = values.reshape(-1)
    nonzero_idx = rng.choice(100, size=17, replace=False)
    flat_view[nonzero_idx] = rng.normal(0, 1, size=17).astype(np.float32)

    tensors: dict[str, np.ndarray] = {
        "sparse.weight": values,
        "dense.bias": rng.normal(0, 1, size=(64,)).astype(np.float32),
    }
    path = str(tmp_path / "sparse.safetensors")
    write_single(path, tensors)

    result = _run_full_analysis(path)
    by_name = {stats.name: stats for stats in result.layer_stats}

    assert by_name["sparse.weight"].sparsity == pytest.approx(0.83, abs=1e-12)
    for name, arr in tensors.items():
        _assert_layer_matches_oracle(by_name[name], _oracle_layer_stats(name, arr))


def test_bf16_tensor_matches_upcast_oracle(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    rng = np.random.default_rng(6)
    base = rng.normal(0, 1, size=(20, 16)).astype(np.float32)
    bf16_tensor = torch.from_numpy(base).to(torch.bfloat16)
    path = str(tmp_path / "bf16.safetensors")
    save_file({"layer.weight": bf16_tensor}, path)

    result = _run_full_analysis(path)
    assert len(result.layer_stats) == 1
    got = result.layer_stats[0]

    # Independent oracle: reinterpret raw bf16 bytes, upcast to float32.
    oracle_values = bf16_tensor.float().numpy()
    oracle = _oracle_layer_stats("layer.weight", oracle_values)
    _assert_layer_matches_oracle(got, oracle)


def test_global_stats_reflects_full_checkpoint_not_partial(tmp_path: Path) -> None:
    tensors = _make_realistic_tensors(seed=7)
    path = str(tmp_path / "model.safetensors")
    write_single(path, tensors)

    result = _run_full_analysis(path)

    full_mean, _full_std = _oracle_global_mean_std(list(tensors.values()))
    assert math.isclose(
        result.global_stats.mean,
        full_mean,
        rel_tol=MEAN_STD_RTOL,
        abs_tol=MEAN_STD_ATOL,
    )
    names = list(tensors)
    partial_values = [tensors[n] for n in names[1:]]
    partial_mean, _ = _oracle_global_mean_std(partial_values)
    assert not math.isclose(
        result.global_stats.mean, partial_mean, rel_tol=1e-9, abs_tol=1e-9
    )


def test_layer_count_and_total_params_match_oracle(tmp_path: Path) -> None:
    tensors = _make_realistic_tensors(seed=8)
    path = str(tmp_path / "model.safetensors")
    write_single(path, tensors)

    result = _run_full_analysis(path)

    oracle_files = load_file(path)
    oracle_total_params = sum(int(v.size) for v in oracle_files.values())

    assert result.health.tensor_count == len(oracle_files)
    assert result.health.total_params == oracle_total_params
    assert sum(stats.param_count for stats in result.layer_stats) == (
        oracle_total_params
    )


def test_many_small_shards_global_stats_match_oracle(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    shards: list[dict[str, np.ndarray]] = [
        {f"layers.{i}.weight": rng.normal(i, 1.0, size=(8, 8)).astype(np.float32)}
        for i in range(15)
    ]
    index_path = write_sharded(str(tmp_path / "many_shards"), shards)

    result = _run_full_analysis(index_path)

    merged: dict[str, np.ndarray] = {}
    for shard in shards:
        merged.update(shard)

    by_name = {stats.name: stats for stats in result.layer_stats}
    for name, arr in merged.items():
        _assert_layer_matches_oracle(by_name[name], _oracle_layer_stats(name, arr))

    oracle_mean, oracle_std = _oracle_global_mean_std(list(merged.values()))
    _assert_global_mean_std_matches(
        result.global_stats.mean, result.global_stats.std, oracle_mean, oracle_std
    )
