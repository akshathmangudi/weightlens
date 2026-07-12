"""Adversarial coverage for sharded safetensors topology edge cases.

Dimension: ``model.safetensors.index.json`` sharding topology, including:
missing shard files, pathological tensor/shard cardinality ratios, mixed
float/non-float tensors within a shard, non-sorted ``weight_map`` shard
ordering, and ``__metadata__`` headers on shard files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file
from safetensors.torch import save_file

from tests.fixtures_safetensors import write_sharded, write_single
from weightlens.sources.safetensors import SafetensorsWeightSource
from weightlens.validators.safetensors import SafetensorsCheckpointValidator


def _write_index(dir_path: str, weight_map: dict[str, str]) -> str:
    """Hand-build index.json with arbitrary weight_map (no shard files)."""
    index_path = os.path.join(dir_path, "model.safetensors.index.json")
    doc = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    with open(index_path, "w") as f:
        json.dump(doc, f)
    return index_path


# --- (1) Shard referenced by index but missing on disk ---------------------


def test_missing_shard_source_raises_file_not_found(tmp_path: Path) -> None:
    shard_a: dict[str, np.ndarray] = {
        "model.layers.0.w": np.random.randn(4, 4).astype(np.float32),
    }
    ckpt_dir = tmp_path / "ckpt"
    index_path = write_sharded(str(ckpt_dir), [shard_a])

    with open(index_path) as f:
        doc = json.load(f)
    doc["weight_map"]["model.layers.1.w"] = "model-00002-of-00002.safetensors"
    with open(index_path, "w") as f:
        json.dump(doc, f)

    source = SafetensorsWeightSource(index_path)
    with pytest.raises(FileNotFoundError):
        list(source.iter_layers())


def test_missing_shard_validator_raises_or_degrades(tmp_path: Path) -> None:
    shard_a: dict[str, np.ndarray] = {
        "w": np.random.randn(3, 3).astype(np.float32),
    }
    ckpt_dir = tmp_path / "ckpt2"
    index_path = write_sharded(str(ckpt_dir), [shard_a])

    with open(index_path) as f:
        doc = json.load(f)
    doc["weight_map"]["ghost.w"] = "model-00002-of-00002.safetensors"
    with open(index_path, "w") as f:
        json.dump(doc, f)

    validator = SafetensorsCheckpointValidator(index_path)
    try:
        health = validator.validate()
    except FileNotFoundError:
        return
    assert health.loadable is False
    assert health.corruption_flags != []


def test_missing_shard_does_not_hang_and_leaves_no_garbage_state(
    tmp_path: Path,
) -> None:
    shard_a: dict[str, np.ndarray] = {
        "keep.me": np.arange(6, dtype=np.float32),
    }
    ckpt_dir = tmp_path / "ckpt3"
    index_path = write_sharded(str(ckpt_dir), [shard_a])
    with open(index_path) as f:
        doc = json.load(f)
    doc["weight_map"]["gone.w"] = "model-00099-of-00002.safetensors"
    with open(index_path, "w") as f:
        json.dump(doc, f)

    collected: list[str] = []
    with pytest.raises(FileNotFoundError):
        for lt in SafetensorsWeightSource(index_path).iter_layers():
            collected.append(lt.name)
    assert "gone.w" not in collected


# --- (2) Many tensors -> few shards, and few tensors -> many shards --------


def test_many_tensors_into_two_shards_bit_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    shard_a: dict[str, np.ndarray] = {
        f"a.{i}.w": rng.standard_normal((3, 2)).astype(np.float32) for i in range(40)
    }
    shard_b: dict[str, np.ndarray] = {
        f"b.{i}.w": rng.standard_normal((2, 3)).astype(np.float32) for i in range(40)
    }
    ckpt_dir = tmp_path / "many_to_few"
    index_path = write_sharded(str(ckpt_dir), [shard_a, shard_b])

    got = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(index_path).iter_layers()
    }
    oracle: dict[str, np.ndarray] = {}
    oracle.update(load_file(str(ckpt_dir / "model-00001-of-00002.safetensors")))
    oracle.update(load_file(str(ckpt_dir / "model-00002-of-00002.safetensors")))

    assert len(got) == 80
    assert set(got) == set(oracle)
    for name in oracle:
        assert np.array_equal(got[name], oracle[name])


def test_few_tensors_into_many_shards_bit_exact(tmp_path: Path) -> None:
    shards: list[dict[str, np.ndarray]] = [
        {f"solo.{i}": np.full((5,), float(i) * 0.5, dtype=np.float32)}
        for i in range(30)
    ]
    ckpt_dir = tmp_path / "few_to_many"
    index_path = write_sharded(str(ckpt_dir), shards)

    got = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(index_path).iter_layers()
    }
    assert len(got) == 30
    for i in range(30):
        expected = np.full((5,), float(i) * 0.5, dtype=np.float32)
        assert np.array_equal(got[f"solo.{i}"], expected)

    health = SafetensorsCheckpointValidator(index_path).validate()
    assert health.tensor_count == 30
    assert health.total_params == 30 * 5


# --- (3) Shard mixing float and non-float tensors --------------------------


def test_shard_with_mixed_float_and_int_tensors_skips_int_only(
    tmp_path: Path,
) -> None:
    shard_a: dict[str, np.ndarray] = {
        "float.w": np.random.randn(4, 4).astype(np.float32),
        "step": np.array([1, 2, 3, 4], dtype=np.int64),
        "mask": np.array([True, False, True], dtype=np.bool_),
    }
    shard_b: dict[str, np.ndarray] = {
        "other.float.w": np.random.randn(2, 2).astype(np.float32),
    }
    ckpt_dir = tmp_path / "mixed"
    index_path = write_sharded(str(ckpt_dir), [shard_a, shard_b])

    got = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(index_path).iter_layers()
    }

    # Only float tensors are streamed; int64/bool are skipped entirely.
    assert set(got) == {"float.w", "other.float.w"}
    oracle = load_file(str(ckpt_dir / "model-00001-of-00002.safetensors"))
    assert np.array_equal(got["float.w"], oracle["float.w"])

    health = SafetensorsCheckpointValidator(index_path).validate()
    assert health.tensor_count == 2
    assert health.total_params == 16 + 4


# --- (4) weight_map lists shards in non-sorted order ------------------------


def test_weight_map_non_sorted_shard_order_still_streams_correctly(
    tmp_path: Path,
) -> None:
    ckpt_dir = tmp_path / "unsorted"
    os.makedirs(ckpt_dir, exist_ok=True)

    shard_z: dict[str, np.ndarray] = {"z.w": np.full((3,), 9.0, dtype=np.float32)}
    shard_a: dict[str, np.ndarray] = {"a.w": np.full((3,), 1.0, dtype=np.float32)}
    shard_m: dict[str, np.ndarray] = {"m.w": np.full((3,), 5.0, dtype=np.float32)}

    save_file(
        {k: torch.from_numpy(v) for k, v in shard_z.items()},
        str(ckpt_dir / "shard-zulu.safetensors"),
    )
    save_file(
        {k: torch.from_numpy(v) for k, v in shard_a.items()},
        str(ckpt_dir / "shard-alpha.safetensors"),
    )
    save_file(
        {k: torch.from_numpy(v) for k, v in shard_m.items()},
        str(ckpt_dir / "shard-mike.safetensors"),
    )

    weight_map = {
        "z.w": "shard-zulu.safetensors",
        "m.w": "shard-mike.safetensors",
        "a.w": "shard-alpha.safetensors",
    }
    index_path = _write_index(str(ckpt_dir), weight_map)

    got = {
        lt.name: float(lt.values[0])
        for lt in SafetensorsWeightSource(index_path).iter_layers()
    }
    assert got == {"z.w": 9.0, "m.w": 5.0, "a.w": 1.0}

    health = SafetensorsCheckpointValidator(index_path).validate()
    assert health.tensor_count == 3
    assert health.total_params == 9


def test_weight_map_reverse_sorted_shard_order_bit_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    shard_1: dict[str, np.ndarray] = {
        "layer.0.w": rng.standard_normal((4, 4)).astype(np.float32)
    }
    shard_2: dict[str, np.ndarray] = {
        "layer.1.w": rng.standard_normal((4, 4)).astype(np.float32)
    }
    ckpt_dir = tmp_path / "reverse"
    write_sharded(str(ckpt_dir), [shard_1, shard_2])
    weight_map = {
        "layer.1.w": "model-00002-of-00002.safetensors",
        "layer.0.w": "model-00001-of-00002.safetensors",
    }
    index_path = _write_index(str(ckpt_dir), weight_map)

    got = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(index_path).iter_layers()
    }
    oracle: dict[str, np.ndarray] = {}
    oracle.update(load_file(str(ckpt_dir / "model-00001-of-00002.safetensors")))
    oracle.update(load_file(str(ckpt_dir / "model-00002-of-00002.safetensors")))

    assert set(got) == set(oracle)
    for name in oracle:
        assert np.array_equal(got[name], oracle[name])


# --- (5) Header with a __metadata__ key present -----------------------------


def test_shard_header_with_metadata_key_is_ignored_as_tensor(
    tmp_path: Path,
) -> None:
    shard_a: dict[str, np.ndarray] = {
        "w": np.random.randn(3, 3).astype(np.float32),
    }
    ckpt_dir = tmp_path / "meta"
    os.makedirs(ckpt_dir, exist_ok=True)
    fname = "model-00001-of-00001.safetensors"
    save_file(
        {k: torch.from_numpy(v) for k, v in shard_a.items()},
        str(ckpt_dir / fname),
        metadata={"format": "pt"},
    )
    weight_map = {"w": fname}
    index_path = _write_index(str(ckpt_dir), weight_map)

    got = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(index_path).iter_layers()
    }
    assert set(got) == {"w"}
    oracle = load_file(str(ckpt_dir / fname))
    assert np.array_equal(got["w"], oracle["w"])

    health = SafetensorsCheckpointValidator(index_path).validate()
    assert health.loadable is True
    assert health.tensor_count == 1
    assert health.total_params == 9


def test_multi_shard_with_metadata_on_one_shard_only(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "meta_mixed"
    os.makedirs(ckpt_dir, exist_ok=True)

    shard_with_meta: dict[str, np.ndarray] = {
        "has.meta.w": np.random.randn(2, 5).astype(np.float32),
    }
    shard_without_meta: dict[str, np.ndarray] = {
        "no.meta.w": np.random.randn(5, 2).astype(np.float32),
    }
    save_file(
        {k: torch.from_numpy(v) for k, v in shard_with_meta.items()},
        str(ckpt_dir / "model-00001-of-00002.safetensors"),
        metadata={"format": "pt"},
    )
    save_file(
        {k: torch.from_numpy(v) for k, v in shard_without_meta.items()},
        str(ckpt_dir / "model-00002-of-00002.safetensors"),
    )
    weight_map = {
        "has.meta.w": "model-00001-of-00002.safetensors",
        "no.meta.w": "model-00002-of-00002.safetensors",
    }
    index_path = _write_index(str(ckpt_dir), weight_map)

    got = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(index_path).iter_layers()
    }
    assert set(got) == {"has.meta.w", "no.meta.w"}

    health = SafetensorsCheckpointValidator(index_path).validate()
    assert health.tensor_count == 2
    assert health.total_params == 10 + 10


def test_write_single_and_sharded_agree_on_full_tensor_set(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    tensors: dict[str, np.ndarray] = {
        f"t{i}": rng.standard_normal((3,)).astype(np.float32) for i in range(6)
    }

    single_path = str(tmp_path / "single.safetensors")
    write_single(single_path, tensors)

    names = list(tensors)
    shards: list[dict[str, np.ndarray]] = [
        {names[0]: tensors[names[0]]},
        {names[1]: tensors[names[1]], names[2]: tensors[names[2]]},
        {
            names[3]: tensors[names[3]],
            names[4]: tensors[names[4]],
            names[5]: tensors[names[5]],
        },
    ]
    index_path = write_sharded(str(tmp_path / "sharded"), shards)

    single_got = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(single_path).iter_layers()
    }
    sharded_got = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(index_path).iter_layers()
    }

    assert set(single_got) == set(sharded_got) == set(tensors)
    for name in tensors:
        assert np.array_equal(single_got[name], tensors[name])
        assert np.array_equal(sharded_got[name], tensors[name])
