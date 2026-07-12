from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import (
    load_file as safetensors_load_file,
    save_file as safetensors_save_file,
)
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

from tests.fixtures_safetensors import write_sharded, write_single


def make_realistic_model_dict(
    seed: int = 42,
    num_layers: int = 6,
    hidden_dim: int = 128,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}

    xavier_std = 0.02 / np.sqrt(hidden_dim)

    result["model.embed_tokens.weight"] = (
        rng.normal(0.0, 0.02, (64, hidden_dim)).astype(np.float32)
    )

    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            result[f"{prefix}.self_attn.{proj}.weight"] = (
                rng.normal(0.0, xavier_std, (hidden_dim, hidden_dim)).astype(
                    np.float32
                )
            )
            result[f"{prefix}.self_attn.{proj}.bias"] = (
                rng.normal(0.0, 0.01, (hidden_dim,)).astype(np.float32)
            )

        for ln_name in ("input_layernorm", "post_attention_layernorm"):
            result[f"{prefix}.{ln_name}.weight"] = (
                np.ones(hidden_dim, dtype=np.float32)
                + rng.normal(0.0, 1e-6, (hidden_dim,)).astype(np.float32)
            )
            result[f"{prefix}.{ln_name}.bias"] = np.zeros(
                hidden_dim, dtype=np.float32
            )

        for proj in ("down_proj", "up_proj"):
            result[f"{prefix}.mlp.{proj}.weight"] = (
                rng.normal(0.0, xavier_std, (4 * hidden_dim, hidden_dim)).astype(
                    np.float32
                )
            )
            result[f"{prefix}.mlp.{proj}.bias"] = (
                rng.normal(0.0, 0.01, (4 * hidden_dim,)).astype(np.float32)
            )

    result["model.norm.weight"] = (
        np.ones(hidden_dim, dtype=np.float32)
        + rng.normal(0.0, 1e-6, (hidden_dim,)).astype(np.float32)
    )

    return result


def make_realistic_model_dict_medium(
    seed: int = 42,
    num_layers: int = 24,
    hidden_dim: int = 512,
) -> dict[str, np.ndarray]:
    return make_realistic_model_dict(
        seed=seed, num_layers=num_layers, hidden_dim=hidden_dim
    )


def make_realistic_pytorch(tmp_path: Path, **kwargs: Any) -> Path:
    model_dict = make_realistic_model_dict(**kwargs)
    st = {
        k: torch.from_numpy(np.ascontiguousarray(v))
        for k, v in model_dict.items()
    }
    path = tmp_path / "model.pth"
    torch.save(st, path)
    return path


def make_realistic_safetensors(tmp_path: Path, **kwargs: Any) -> Path:
    model_dict = make_realistic_model_dict(**kwargs)
    path = tmp_path / "model.safetensors"
    write_single(str(path), model_dict)
    return path


def make_realistic_sharded_safetensors(
    tmp_path: Path, num_shards: int = 3, **kwargs: Any
) -> Path:
    model_dict = make_realistic_model_dict(**kwargs)
    dest_dir = tmp_path / "sharded"
    shards: list[dict[str, np.ndarray]] = [{} for _ in range(num_shards)]
    for i, (name, arr) in enumerate(model_dict.items()):
        shards[i % num_shards][name] = arr
    return Path(write_sharded(str(dest_dir), shards))


def make_realistic_dcp(tmp_path: Path, **kwargs: Any) -> Path:
    model_dict = make_realistic_model_dict(**kwargs)
    st = {
        k: torch.from_numpy(np.ascontiguousarray(v))
        for k, v in model_dict.items()
    }
    path = tmp_path / "dcp"
    path.mkdir(parents=True, exist_ok=True)
    dcp_save(st, checkpoint_id=str(path), no_dist=True)
    return path


def _find_largest_tensor(
    tensors: Mapping[str, np.ndarray | torch.Tensor],
) -> str:
    best_name = ""
    best_size = -1
    for name, arr in tensors.items():
        if isinstance(arr, torch.Tensor):
            if arr.dtype != torch.float32:
                continue
            n_elements = arr.numel()
        else:
            if arr.dtype != np.float32:
                continue
            n_elements = arr.size
        if n_elements > best_size:
            best_size = n_elements
            best_name = name
    if not best_name:
        raise ValueError("No float32 tensor found in checkpoint.")
    return best_name


def corrupt_with_zeros(
    ckpt_path: Path, fmt: str = "pytorch"
) -> None:
    if fmt == "pytorch":
        ckpt: dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
        st = ckpt
        largest = _find_largest_tensor(
            {k: v.numpy() for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        )
        st[largest] = torch.zeros_like(st[largest])
        torch.save(st, ckpt_path)
    elif fmt == "safetensors":
        st = safetensors_load_file(str(ckpt_path))
        largest = _find_largest_tensor(st)
        st[largest] = torch.zeros_like(st[largest])
        safetensors_save_file(st, str(ckpt_path))
    else:
        raise ValueError(f"Unsupported corruption format: {fmt}")


def corrupt_with_spike(
    ckpt_path: Path, fmt: str = "pytorch"
) -> None:
    if fmt == "pytorch":
        ckpt: dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
        st = ckpt
        largest = _find_largest_tensor(
            {k: v.numpy() for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        )
        st[largest] = st[largest].clone()
        st[largest].view(-1)[0] = 1e6
        torch.save(st, ckpt_path)
    elif fmt == "safetensors":
        st = safetensors_load_file(str(ckpt_path))
        largest = _find_largest_tensor(st)
        flat = st[largest].ravel()
        flat[0] = 1e6
        safetensors_save_file(st, str(ckpt_path))
    else:
        raise ValueError(f"Unsupported corruption format: {fmt}")


def corrupt_with_nans(
    ckpt_path: Path, fmt: str = "pytorch"
) -> None:
    if fmt == "pytorch":
        ckpt: dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
        st = ckpt
        largest = _find_largest_tensor(
            {k: v.numpy() for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        )
        st[largest] = torch.full_like(st[largest], float("nan"))
        torch.save(st, ckpt_path)
    elif fmt == "safetensors":
        st = safetensors_load_file(str(ckpt_path))
        largest = _find_largest_tensor(st)
        st[largest] = torch.full_like(st[largest], float("nan"))
        safetensors_save_file(st, str(ckpt_path))
    else:
        raise ValueError(f"Unsupported corruption format: {fmt}")


def corrupt_truncated(
    ckpt_path: Path, fmt: str = "pytorch", fraction: float = 0.5
) -> Path:
    original = ckpt_path.read_bytes()
    trunc_len = max(1, int(len(original) * fraction))
    truncated = original[:trunc_len]
    new_path = ckpt_path.with_name(
        f"{ckpt_path.stem}_truncated{ckpt_path.suffix}"
    )
    new_path.write_bytes(truncated)
    return new_path


def make_extreme_range_model(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    path = tmp_path / "extreme.safetensors"
    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": rng.uniform(-100.0, 100.0, (64, 64)).astype(
            np.float32
        ),
        "model.layers.0.self_attn.q_proj.weight": rng.uniform(
            -100.0, 100.0, (64, 64)
        ).astype(np.float32),
        "model.layers.0.self_attn.q_proj.bias": rng.uniform(
            -100.0, 100.0, (64,)
        ).astype(np.float32),
        "model.layers.0.mlp.down_proj.weight": rng.uniform(
            -100.0, 100.0, (256, 64)
        ).astype(np.float32),
    }
    write_single(str(path), tensors)
    return path
