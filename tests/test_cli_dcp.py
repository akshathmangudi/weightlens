from __future__ import annotations

import warnings
from pathlib import Path

import torch
from rich.console import Console
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

from weightlens.cli import run_cli


def _save_dcp(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dcp_save(state_dict, checkpoint_id=str(path), no_dist=True)


def _make_console() -> Console:
    return Console(
        record=True,
        force_terminal=False,
        color_system=None,
        width=120,
    )


# ------------------------------------------------------------------
# Auto-detection
# ------------------------------------------------------------------


def test_auto_detects_dcp_directory(tmp_path: Path) -> None:
    state = {"model.weight": torch.randn(16, 16)}
    _save_dcp(tmp_path / "ckpt", state)
    console = _make_console()

    exit_code = run_cli(["analyze", str(tmp_path / "ckpt")], console=console)

    output = console.export_text()
    assert exit_code == 0
    assert "EXPERIMENTAL" in output
    assert "Statistics for" in output


# ------------------------------------------------------------------
# Explicit --format dcp
# ------------------------------------------------------------------


def test_explicit_format_dcp(tmp_path: Path) -> None:
    state = {"model.weight": torch.randn(16, 16)}
    _save_dcp(tmp_path / "ckpt", state)
    console = _make_console()

    exit_code = run_cli(
        ["analyze", str(tmp_path / "ckpt"), "--format", "dcp"],
        console=console,
    )

    output = console.export_text()
    assert exit_code == 0
    assert "EXPERIMENTAL" in output


# ------------------------------------------------------------------
# Error cases
# ------------------------------------------------------------------


def test_nonexistent_path(tmp_path: Path) -> None:
    console = _make_console()

    exit_code = run_cli(
        ["analyze", str(tmp_path / "nope")],
        console=console,
    )

    assert exit_code == 2


def test_directory_without_metadata(tmp_path: Path) -> None:
    path = tmp_path / "empty_dir"
    path.mkdir()
    # Make it non-empty so it's not detected as "empty path"
    (path / "random.txt").write_text("not a checkpoint")
    console = _make_console()

    exit_code = run_cli(
        ["analyze", str(path)],
        console=console,
    )

    output = console.export_text()
    assert exit_code == 2
    assert "metadata" in output.lower()


def test_include_optimizer_flag(tmp_path: Path) -> None:
    """The --include-optimizer flag is accepted and analysis succeeds."""
    state = {"model.weight": torch.randn(16, 16)}
    _save_dcp(tmp_path / "ckpt", state)
    console = _make_console()

    exit_code = run_cli(
        ["analyze", str(tmp_path / "ckpt"), "--format", "dcp", "--include-optimizer"],
        console=console,
    )

    output = console.export_text()
    assert exit_code == 0
    assert "EXPERIMENTAL" in output


def test_dcp_not_loadable_shows_health(tmp_path: Path) -> None:
    """A DCP directory with missing shard shows health table and aborts."""
    state = {"w": torch.randn(8, 8)}
    _save_dcp(tmp_path / "ckpt", state)
    # Remove the shard to make it not loadable
    (tmp_path / "ckpt" / "__0_0.distcp").unlink()
    console = _make_console()

    exit_code = run_cli(
        ["analyze", str(tmp_path / "ckpt"), "--format", "dcp"],
        console=console,
    )

    output = console.export_text()
    assert exit_code == 1
    assert "Checkpoint Health" in output
    assert "missing_shard" in output
    assert "Analysis aborted" in output
