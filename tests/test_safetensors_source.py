from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from tests.fixtures_safetensors import write_sharded, write_single
from weightlens.sources.safetensors import SafetensorsWeightSource


def test_single_file_yields_float_tensors_bit_exact(tmp_path: Path) -> None:
    tensors = {
        "model.a.weight": np.random.randn(4, 3).astype(np.float32),
        "model.b.bias": np.random.randn(8).astype(np.float32),
    }
    path = str(tmp_path / "model.safetensors")
    write_single(path, tensors)

    got = {lt.name: lt.values for lt in SafetensorsWeightSource(path).iter_layers()}
    oracle = load_file(path)

    assert set(got) == set(oracle)
    for name in oracle:
        assert np.array_equal(got[name], oracle[name])


def test_single_file_skips_integer_tensors(tmp_path: Path) -> None:
    tensors: dict[str, np.ndarray] = {
        "w": np.random.randn(4).astype(np.float32),
        "step": np.array([1, 2, 3], dtype=np.int64),
    }
    path = str(tmp_path / "m.safetensors")
    write_single(path, tensors)
    names = [lt.name for lt in SafetensorsWeightSource(path).iter_layers()]
    assert names == ["w"]


def test_single_file_over_memory_matches_local(tmp_path: Path) -> None:
    import fsspec

    t = {"w": np.arange(12, dtype=np.float32).reshape(3, 4)}
    local = str(tmp_path / "m.safetensors")
    write_single(local, t)
    with open(local, "rb") as f, fsspec.open("memory://m.safetensors", "wb") as g:
        g.write(f.read())

    got = next(SafetensorsWeightSource("memory://m.safetensors").iter_layers())
    assert np.array_equal(got.values, t["w"])


def test_sharded_index_streams_all_shards_bit_exact(tmp_path: Path) -> None:
    import fsspec  # noqa: F401

    shard_a: dict[str, np.ndarray] = {
        "model.layers.0.w": np.random.randn(6, 5).astype(np.float32),
    }
    shard_b: dict[str, np.ndarray] = {
        "model.layers.1.w": np.random.randn(5, 5).astype(np.float32),
        "model.norm.weight": np.random.randn(5).astype(np.float32),
    }
    index_path = write_sharded(str(tmp_path / "ckpt"), [shard_a, shard_b])

    got = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(index_path).iter_layers()
    }
    oracle: dict[str, np.ndarray] = {}
    shard1_path = str(tmp_path / "ckpt" / "model-00001-of-00002.safetensors")
    shard2_path = str(tmp_path / "ckpt" / "model-00002-of-00002.safetensors")
    oracle.update(load_file(shard1_path))
    oracle.update(load_file(shard2_path))

    assert set(got) == set(oracle)
    for name in oracle:
        assert np.array_equal(got[name], oracle[name])


def test_sharded_many_tiny_shards(tmp_path: Path) -> None:
    # FSDP-like pathological case: many shards with one tensor each.
    shards: list[dict[str, np.ndarray]] = [
        {f"t{i}": np.full((2,), float(i), dtype=np.float32)} for i in range(25)
    ]
    index_path = write_sharded(str(tmp_path / "many"), shards)
    got = {
        lt.name: float(lt.values[0])
        for lt in SafetensorsWeightSource(index_path).iter_layers()
    }
    assert got == {f"t{i}": float(i) for i in range(25)}
