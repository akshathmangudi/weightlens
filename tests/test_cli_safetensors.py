from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from tests.fixtures_safetensors import write_sharded, write_single
from weightlens.cli import run_cli


def _make_console() -> Console:
    return Console(
        record=True,
        force_terminal=False,
        color_system=None,
        width=120,
    )


def test_cli_analyzes_single_safetensors(tmp_path: Path) -> None:
    tensors: dict[str, np.ndarray] = {
        "model.layer.weight": np.random.randn(8, 8).astype(np.float32)
    }
    write_single(str(tmp_path / "model.safetensors"), tensors)
    console = _make_console()

    exit_code = run_cli(
        ["analyze", str(tmp_path / "model.safetensors")], console=console
    )

    output = console.export_text()
    assert exit_code == 0
    assert "Statistics for" in output


def test_cli_analyzes_sharded_safetensors(tmp_path: Path) -> None:
    shards: list[dict[str, np.ndarray]] = [
        {"model.a.weight": np.random.randn(8, 8).astype(np.float32)},
        {"model.b.weight": np.random.randn(8, 8).astype(np.float32)},
    ]
    index_path = write_sharded(str(tmp_path / "ck"), shards)
    console = _make_console()

    exit_code = run_cli(["analyze", index_path], console=console)

    output = console.export_text()
    assert exit_code == 0
    assert "Statistics for" in output


def test_cli_missing_backend_returns_3(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate the s3fs backend being absent by making fsspec's url_to_fs raise
    # ImportError. This drives the REAL path (ByteRangeReader -> MissingBackendError
    # -> validate() -> CLI), verifying the error propagates to the exit-3 install
    # hint instead of being swallowed and misreported as a corrupt checkpoint.
    def _no_backend(uri: str) -> object:
        raise ImportError("no s3fs installed")

    monkeypatch.setattr(
        "weightlens.io.byte_range.fsspec.core.url_to_fs", _no_backend
    )
    console = _make_console()

    exit_code = run_cli(["analyze", "s3://bucket/model.safetensors"], console=console)

    output = console.export_text()
    assert exit_code == 3
    assert "weightlens[s3]" in output
