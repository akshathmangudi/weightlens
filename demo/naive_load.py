from __future__ import annotations

import torch
from demo.make_clean_ckpt import ToyNet


def load_model(path: str) -> ToyNet:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = ToyNet()
    result = model.load_state_dict(state)
    print(f"{path}: {result}")
    model.eval()
    return model


def main() -> None:
    x = torch.randn(4, 3, 8, 8)

    clean = load_model("demo/checkpoints/clean.pth")
    zero = load_model("demo/checkpoints/corrupted_zero.pth")
    spike = load_model("demo/checkpoints/corrupted_spike.pth")

    with torch.no_grad():
        y_clean = clean(x)
        y_zero = zero(x)
        y_spike = spike(x)

    print("max|delta| clean vs zero:", (y_clean - y_zero).abs().max().item())
    print("max|delta| clean vs spike:", (y_clean - y_spike).abs().max().item())


if __name__ == "__main__":
    main()
