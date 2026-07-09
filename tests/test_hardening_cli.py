"""CLI robustness tests: exit codes, error clarity, Rich markup safety.

Tests rendering of user-controlled strings (paths, URIs, hints) with
bracketed text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from rich.console import Console

from tests.fixtures_safetensors import write_sharded, write_single
from weightlens.cli import run_cli


def _make_console(width: int = 120) -> Console:
    """Recording, non-terminal Console for testing."""
    return Console(
        record=True,
        force_terminal=False,
        color_system=None,
        width=width,
    )


def _save_pth(tmp_path: Path) -> Path:
    state = {
        "layer1.weight": torch.randn(4, 4, dtype=torch.float32),
        "layer1.bias": torch.randn(4, dtype=torch.float32),
    }
    checkpoint_path = tmp_path / "model.pth"
    torch.save(state, checkpoint_path)
    return checkpoint_path




def test_cli_single_safetensors_exits_zero(tmp_path: Path) -> None:
    tensors: dict[str, np.ndarray] = {
        "w": np.random.randn(6, 6).astype(np.float32),
    }
    path = tmp_path / "model.safetensors"
    write_single(str(path), tensors)
    console = _make_console()

    exit_code = run_cli(["analyze", str(path)], console=console)

    output = console.export_text()
    assert exit_code == 0
    assert "Statistics for" in output
    assert "corruption_flags:   none" in output


def test_cli_sharded_index_exits_zero(tmp_path: Path) -> None:
    shards: list[dict[str, np.ndarray]] = [
        {"model.a.weight": np.random.randn(5, 5).astype(np.float32)},
        {"model.b.weight": np.random.randn(5, 5).astype(np.float32)},
    ]
    index_path = write_sharded(str(tmp_path / "shards"), shards)
    console = _make_console()

    exit_code = run_cli(["analyze", index_path], console=console)

    output = console.export_text()
    assert exit_code == 0
    assert "Statistics for" in output
    assert "total_params:       50" in output


def test_cli_pth_checkpoint_exits_zero(tmp_path: Path) -> None:
    checkpoint_path = _save_pth(tmp_path)
    console = _make_console()

    exit_code = run_cli(["analyze", str(checkpoint_path)], console=console)

    output = console.export_text()
    assert exit_code == 0
    assert "Statistics for" in output
    assert "Global Stats" in output




def test_cli_explicit_format_safetensors_override(tmp_path: Path) -> None:
    tensors: dict[str, np.ndarray] = {
        "w": np.random.randn(3, 3).astype(np.float32),
    }
    path = tmp_path / "model.bin"
    write_single(str(path), tensors)
    console = _make_console()

    exit_code = run_cli(
        ["analyze", str(path), "--format", "safetensors"], console=console
    )

    output = console.export_text()
    assert exit_code == 0
    assert "Statistics for" in output
    assert "total_params:       9" in output




def test_cli_nonexistent_path_nonzero_exit_with_message(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.safetensors"
    console = _make_console(width=1000)

    exit_code = run_cli(["analyze", str(missing)], console=console)

    output = console.export_text()
    assert exit_code != 0
    assert "not found" in output.lower()
    assert str(missing) in output


def test_cli_nonexistent_pth_path_nonzero_exit(tmp_path: Path) -> None:
    missing = tmp_path / "ghost.pth"
    console = _make_console()

    exit_code = run_cli(["analyze", str(missing)], console=console)

    output = console.export_text()
    assert exit_code != 0
    assert "not found" in output.lower()


def test_cli_nonexistent_dcp_directory_nonzero_exit(tmp_path: Path) -> None:
    missing = tmp_path / "ghost_ckpt"
    console = _make_console()

    exit_code = run_cli(
        ["analyze", str(missing), "--format", "dcp"], console=console
    )

    assert exit_code != 0




def test_cli_corrupt_safetensors_nonzero_no_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corrupt = tmp_path / "corrupt.safetensors"
    corrupt.write_bytes(b"this is definitely not a safetensors file 0000000")
    console = _make_console()

    exit_code = run_cli(["analyze", str(corrupt)], console=console)

    output = console.export_text()
    captured = capsys.readouterr()

    assert exit_code != 0
    assert "Traceback" not in output
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err
    assert "Analysis aborted" in output


def test_cli_truncated_safetensors_header_nonzero_no_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(b"\x01\x02\x03")
    console = _make_console()

    exit_code = run_cli(["analyze", str(truncated)], console=console)

    output = console.export_text()
    captured = capsys.readouterr()

    assert exit_code != 0
    assert "Traceback" not in output
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


def test_cli_sharded_index_bad_json_nonzero_no_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt_dir = tmp_path / "bad_shard_ckpt"
    ckpt_dir.mkdir()
    index_path = ckpt_dir / "model.safetensors.index.json"
    index_path.write_text("{not valid json::")
    console = _make_console()

    exit_code = run_cli(["analyze", str(index_path)], console=console)

    output = console.export_text()
    captured = capsys.readouterr()

    assert exit_code != 0
    assert "Traceback" not in output
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err




def test_cli_bracketed_path_segment_like_uri_is_not_swallowed(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "x[0].safetensors"
    console = _make_console(width=1000)

    exit_code = run_cli(["analyze", str(missing)], console=console)

    output = console.export_text()
    assert exit_code != 0
    assert "x[0].safetensors" in output


def test_cli_bracketed_install_hint_like_text_is_not_swallowed(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-weightlens[s3]-shard.safetensors"
    console = _make_console(width=1000)

    exit_code = run_cli(["analyze", str(missing)], console=console)

    output = console.export_text()
    assert exit_code != 0
    assert "weightlens[s3]" in output


def test_cli_bracketed_text_in_health_table_is_not_swallowed(
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt.safetensors"
    corrupt.write_bytes(b"\xff\xff\xff\xff\xff\xff\xff\xff")
    console = _make_console()

    exit_code = run_cli(["analyze", str(corrupt)], console=console)
    assert exit_code != 0

    from weightlens.cli import _render_health
    from weightlens.models import CheckpointHealth

    health = CheckpointHealth(
        file_size_bytes=0,
        is_empty=True,
        loadable=False,
        tensor_count=0,
        total_params=0,
        corruption_flags=["unreadable: install weightlens[s3] to continue"],
    )
    table_console = _make_console(width=200)
    _render_health(table_console, health)
    table_output = table_console.export_text()
    assert "weightlens[s3]" in table_output
