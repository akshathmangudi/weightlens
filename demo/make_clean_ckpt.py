from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


class ToyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return self.fc2(x)


def main() -> None:
    model = ToyNet()

    for param in model.parameters():
        if param.dim() > 1:
            scaled_std = 0.02 / math.sqrt(param.numel())
            nn.init.normal_(param, mean=0.0, std=scaled_std)
        else:
            nn.init.normal_(param, mean=0.0, std=0.001)

    out_dir = Path("demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "clean.pth")
    print("Saved demo/clean.pth")


if __name__ == "__main__":
    main()
