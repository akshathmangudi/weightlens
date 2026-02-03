from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from rich.console import Console

from weightlens.cli import run_cli


def _save_checkpoint(tmp_path: Path) -> Path:
    state: OrderedDict[str, torch.Tensor] = OrderedDict(
        layer1=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        layer2=torch.tensor([5.0, 6.0], dtype=torch.float64),
    )
    checkpoint_path = tmp_path / "model.pth"
    torch.save(state, checkpoint_path)
    return checkpoint_path


def test_cli_analyze_happy_path(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)
    console = Console(
        record=True,
        force_terminal=False,
        color_system=None,
        width=120,
    )

    exit_code = run_cli(["analyze", str(checkpoint_path)], console=console)

    output = console.export_text()
    assert exit_code == 0
    assert "Statistics for" in output
    assert "Global Stats" in output
    assert "Diagnostics (" in output
    assert "Summary: layers=" not in output  # Removed summary line in new format


def test_cli_analyze_empty_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "empty.pth"
    checkpoint_path.write_bytes(b"")
    console = Console(
        record=True,
        force_terminal=False,
        color_system=None,
        width=120,
    )

    exit_code = run_cli(["analyze", str(checkpoint_path)], console=console)

    output = console.export_text()
    assert exit_code == 1
    assert "Checkpoint Health" in output
    assert "empty_file" in output
    assert "Analysis aborted" in output
