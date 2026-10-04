from __future__ import annotations

import json
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
from rich.console import Console
from safetensors.torch import save_file

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


@pytest.mark.parametrize("format_name", ["pt", "safetensors"])
def test_cli_reports_small_scale_spike(
    tmp_path: Path, format_name: str
) -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(0.0, 1e-4, size=10_000).astype(np.float32)
    values[0] = 0.1
    checkpoint_path = tmp_path / f"spike.{format_name}"
    state = {"layer.weight": torch.from_numpy(values)}
    if format_name == "pt":
        torch.save(state, checkpoint_path)
    else:
        save_file(state, str(checkpoint_path))

    console = Console(
        record=True, force_terminal=False, color_system=None, width=120
    )
    exit_code = run_cli(["analyze", str(checkpoint_path)], console=console)

    output = console.export_text()
    assert "extreme-spike" in output
    assert "FAILED" in output
    assert exit_code == 1


def test_cli_does_not_pass_when_local_bf16_contains_nan(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "mixed-bf16.safetensors"
    save_file(
        {
            "f32.weight": torch.tensor([1.0], dtype=torch.float32),
            "bf16.weight": torch.tensor([float("nan")], dtype=torch.bfloat16),
        },
        str(checkpoint_path),
    )
    console = Console(record=True, force_terminal=False, color_system=None)

    exit_code = run_cli(["analyze", str(checkpoint_path)], console=console)

    output = console.export_text()
    assert "Layer contains NaN or" in output
    assert "FAILED" in output
    assert "PASSED" not in output
    assert exit_code == 1


def test_cli_shows_active_thresholds_on_pass(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path)
    console = Console(record=True, force_terminal=False, color_system=None, width=120)

    exit_code = run_cli(
        [
            "analyze",
            str(checkpoint_path),
            "--variance-threshold",
            "12.5",
            "--spike-threshold",
            "250",
            "--norm-threshold",
            "6.25",
            "--sparsity-threshold",
            "0.95",
        ],
        console=console,
    )

    output = console.export_text()
    assert exit_code == 0
    assert "Diagnostic thresholds" in output
    for flag, value in (
        ("--variance-threshold", "12.5"),
        ("--spike-threshold", "250.0"),
        ("--norm-threshold", "6.25"),
        ("--sparsity-threshold", "0.95"),
    ):
        assert flag in output
        assert value in output


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--variance-threshold", "nan"),
        ("--spike-threshold", "inf"),
        ("--norm-threshold", "0"),
        ("--sparsity-threshold", "-0.1"),
        ("--sparsity-threshold", "1.1"),
        ("--sparsity-threshold", "nan"),
    ],
)
def test_cli_rejects_invalid_threshold(flag: str, value: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_cli(["analyze", "unused.pt", flag, value])

    assert exc_info.value.code == 2


def test_cli_analyzes_bf16_only_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "only-bf16.safetensors"
    save_file(
        {"bf16.weight": torch.tensor([0.25, -0.125], dtype=torch.bfloat16)},
        str(checkpoint_path),
    )
    console = Console(record=True, force_terminal=False, color_system=None)

    exit_code = run_cli(["analyze", str(checkpoint_path)], console=console)

    assert exit_code == 0
    assert "PASSED" in console.export_text()


def test_cli_rejects_overlapping_safetensors(tmp_path: Path) -> None:
    entry = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
    header = json.dumps({"a.weight": entry, "b.weight": entry}).encode()
    path = tmp_path / "overlap.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))
    console = Console(record=True, force_terminal=False, color_system=None)

    exit_code = run_cli(["analyze", str(path)], console=console)

    assert exit_code == 1
    assert "Analysis aborted" in console.export_text()
    assert "PASSED" not in console.export_text()
