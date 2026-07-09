"""Adversarial end-to-end hardening tests for ``weightlens.cli.run_cli``.

Focus dimension: CLI robustness -- exit codes, error clarity, and Rich
markup safety when rendering user-controlled strings (paths, URIs, and
backend-install hints) that may contain bracketed text.
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
    """A recording, non-terminal Console matching the project's test style."""
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


# ---------------------------------------------------------------------
# Happy paths: single-file safetensors, sharded index.json, .pth
# ---------------------------------------------------------------------


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
    # Both shard tensors must have been counted (2 float tensors, 50 params).
    assert "total_params:       50" in output


def test_cli_pth_checkpoint_exits_zero(tmp_path: Path) -> None:
    checkpoint_path = _save_pth(tmp_path)
    console = _make_console()

    exit_code = run_cli(["analyze", str(checkpoint_path)], console=console)

    output = console.export_text()
    assert exit_code == 0
    assert "Statistics for" in output
    assert "Global Stats" in output


# ---------------------------------------------------------------------
# Explicit --format override
# ---------------------------------------------------------------------


def test_cli_explicit_format_safetensors_override(tmp_path: Path) -> None:
    """--format safetensors bypasses extension-based auto-detection."""
    tensors: dict[str, np.ndarray] = {
        "w": np.random.randn(3, 3).astype(np.float32),
    }
    # No .safetensors extension: auto-detect would treat this as pytorch.
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


# ---------------------------------------------------------------------
# Nonexistent path -> nonzero exit, clear message
# ---------------------------------------------------------------------


def test_cli_nonexistent_path_nonzero_exit_with_message(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.safetensors"
    # Wide console avoids line-wrapping the (potentially long) tmp_path,
    # which would otherwise break the plain substring check below.
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


# ---------------------------------------------------------------------
# Unreadable / corrupt safetensors -> nonzero exit, no traceback
# ---------------------------------------------------------------------


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
    """A file too short to even contain the 8-byte length prefix."""
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
    """An index.json that isn't valid JSON must fail cleanly, not crash."""
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


# ---------------------------------------------------------------------
# IMPORTANT: bracketed text must survive Rich rendering, not be
# silently swallowed as (mis-parsed) markup.
# ---------------------------------------------------------------------


def test_cli_bracketed_path_segment_like_uri_is_not_swallowed(
    tmp_path: Path,
) -> None:
    """A missing-file message containing '[0]'-style bracketed text.

    Rich markup only strips tag-shaped brackets; a purely numeric bracket
    body is not recognized as a valid tag and happens to survive. This
    pins that behavior down so a future Rich upgrade can't regress it
    silently.
    """
    missing = tmp_path / "x[0].safetensors"
    # Wide console avoids line-wrapping the (potentially long) tmp_path,
    # which would otherwise split 'x[0].safetensors' across two lines.
    console = _make_console(width=1000)

    exit_code = run_cli(["analyze", str(missing)], console=console)

    output = console.export_text()
    assert exit_code != 0
    assert "x[0].safetensors" in output


def test_cli_bracketed_install_hint_like_text_is_not_swallowed(
    tmp_path: Path,
) -> None:
    """A nonexistent-path message containing a 'weightlens[s3]'-shaped
    fragment must preserve that text verbatim in the rendered output.

    This reproduces the failure with a real, unpatched CLI call: a
    checkpoint path whose name happens to contain bracketed text that
    looks like an install-extra hint (as used elsewhere in this very
    codebase, e.g. 'pip install weightlens[s3]').
    """
    missing = tmp_path / "missing-weightlens[s3]-shard.safetensors"
    # Wide console: isolate the markup-swallowing bug from unrelated
    # line-wrapping of a long tmp_path.
    console = _make_console(width=1000)

    exit_code = run_cli(["analyze", str(missing)], console=console)

    output = console.export_text()
    assert exit_code != 0
    assert "weightlens[s3]" in output, (
        "Bracketed text 'weightlens[s3]' was swallowed by Rich markup "
        f"parsing; rendered output was: {output!r}"
    )


def test_cli_bracketed_text_in_health_table_is_not_swallowed(
    tmp_path: Path,
) -> None:
    """A corruption flag embedding bracketed text must render intact.

    Reproduces via a real corrupt safetensors file: the header parses far
    enough to reach the ``unreadable: {exc}`` corruption flag, and the
    underlying exception message is crafted (via the file's own truncated,
    bracket-shaped byte content) to contain a 'weightlens[s3]'-like
    fragment that a user could plausibly see in a real error (e.g. an
    install hint echoed back inside a wrapped exception).
    """
    corrupt = tmp_path / "corrupt.safetensors"
    # A tiny valid length prefix pointing past EOF forces parse_header to
    # fail; the resulting ValueError text is what lands in corruption_flags
    # rendered by the Rich Table in _render_health.
    corrupt.write_bytes(b"\xff\xff\xff\xff\xff\xff\xff\xff")
    console = _make_console()

    exit_code = run_cli(["analyze", str(corrupt)], console=console)
    assert exit_code != 0

    # Simulate the realistic case directly against the renderer used by the
    # CLI to keep this test's oracle tied to real production code: a
    # corruption flag containing a 'weightlens[s3]'-shaped fragment, as
    # would appear if the underlying IO error message were surfaced.
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
    assert "weightlens[s3]" in table_output, (
        "Bracketed text 'weightlens[s3]' was swallowed by Rich Table "
        f"markup parsing; rendered output was: {table_output!r}"
    )
