from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from tests.fixtures_safetensors import write_single
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
