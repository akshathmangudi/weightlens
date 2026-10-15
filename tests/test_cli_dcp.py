from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import torch
from rich.console import Console
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

from weightlens.cli import run_cli
from weightlens.sources.dcp import read_metadata


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


def test_dcp_error_diagnostic_reports_findings_without_failing(tmp_path: Path) -> None:
    values = torch.full((10_000,), 1e-4, dtype=torch.float32)
    values[0] = 0.1
    _save_dcp(tmp_path / "ckpt", {"model.weight": values})
    console = _make_console()

    exit_code = run_cli(["analyze", str(tmp_path / "ckpt")], console=console)

    output = console.export_text()
    assert "extreme-spike" in output
    assert "FINDINGS" in output
    assert exit_code == 0


def test_dcp_empty_tensor_is_warning_without_corruption_flag(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ckpt"
    _save_dcp(
        checkpoint,
        {"empty.weight": torch.empty(0), "ok.weight": torch.ones(2)},
    )
    console = _make_console()

    exit_code = run_cli(["analyze", str(checkpoint)], console=console)
    output = " ".join(console.export_text().split())

    assert exit_code == 0
    assert "Inspection: PARTIAL" in output
    assert "empty.weight" in output
    assert "empty-tensor" in output
    assert "1 warning" in output
    assert "corruption_flags: none" in output


def test_dcp_rejects_checkpoint_with_only_empty_tensors(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt"
    _save_dcp(checkpoint, {"empty.weight": torch.empty(0)})
    console = _make_console()

    exit_code = run_cli(["analyze", str(checkpoint)], console=console)
    output = " ".join(console.export_text().replace("│", " ").split())

    assert exit_code == 1
    assert "Checkpoint Health" in output
    assert "is_empty True" in output
    assert "tensor_count 1" in output
    assert "total_params 0" in output
    assert "contains tensors but no values to analyze" in output


def test_missing_storage_location_explains_partial_inspection(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ckpt"
    _save_dcp(
        checkpoint,
        {
            "present.weight": torch.ones(4),
            "missing.weight": torch.ones(4),
        },
    )
    metadata = read_metadata(checkpoint)
    metadata.storage_data = {
        index: info
        for index, info in metadata.storage_data.items()
        if index.fqn != "missing.weight"
    }
    with (checkpoint / ".metadata").open("wb") as handle:
        pickle.dump(metadata, handle)

    console = _make_console()
    exit_code = run_cli(["analyze", str(checkpoint)], console=console)
    output = " ".join(console.export_text().split())

    assert exit_code == 0
    assert "Inspection: PARTIAL" in output
    assert "Coverage: 1 of 2 eligible tensors analyzed" in output
    assert "missing.weight" in output
    assert "DCP metadata has no usable storage location for this tensor." in output
    assert "Its values" in output
    assert "were not inspected." in output


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
