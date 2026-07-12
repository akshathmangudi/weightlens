from __future__ import annotations

import json
import os

import numpy as np
import torch
from safetensors.torch import save_file


def write_single(path: str, tensors: dict[str, np.ndarray]) -> None:
    """Write a single-file safetensors checkpoint to a local path."""
    st = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in tensors.items()}
    save_file(st, path)


def write_sharded(dir_path: str, shards: list[dict[str, np.ndarray]]) -> str:
    """Write N shard files + a model.safetensors.index.json. Return index path."""
    os.makedirs(dir_path, exist_ok=True)
    n = len(shards)
    weight_map: dict[str, str] = {}
    total = 0
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        st = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in shard.items()}
        save_file(st, os.path.join(dir_path, fname))
        for name, arr in shard.items():
            weight_map[name] = fname
            total += arr.nbytes
    index_path = os.path.join(dir_path, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f)
    return index_path
