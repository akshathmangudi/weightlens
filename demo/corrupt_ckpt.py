from __future__ import annotations

from pathlib import Path

import torch


def load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("Expected a state_dict mapping.")

    tensors: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Expected tensor for key {name}.")
        tensors[str(name)] = value
    return tensors


def corrupt_zero_flood(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    corrupted = False
    for name, tensor in state.items():
        if (
            not corrupted
            and tensor.is_floating_point()
            and tensor.numel() > 1000
        ):
            out[name] = torch.zeros_like(tensor)
            corrupted = True
        else:
            out[name] = tensor
    if not corrupted:
        raise RuntimeError("No suitable tensor found for zero flood.")
    return out


def corrupt_spike(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    corrupted = False
    for name, tensor in state.items():
        if (
            not corrupted
            and tensor.is_floating_point()
            and tensor.numel() > 1000
        ):
            spike = tensor.clone()
            spike.view(-1)[0] = 1e6
            out[name] = spike
            corrupted = True
        else:
            out[name] = tensor
    if not corrupted:
        raise RuntimeError("No suitable tensor found for spike corruption.")
    return out


def main() -> None:
    demo_dir = Path("demo")
    clean_path = demo_dir / "clean.pth"
    if not clean_path.exists():
        raise FileNotFoundError("Missing demo/clean.pth. Run make_clean_ckpt.py")

    state = load_state(clean_path)

    torch.save(corrupt_zero_flood(state), demo_dir / "corrupted_zero.pth")
    torch.save(corrupt_spike(state), demo_dir / "corrupted_spike.pth")
    print("Saved demo/corrupted_zero.pth")
    print("Saved demo/corrupted_spike.pth")


if __name__ == "__main__":
    main()
