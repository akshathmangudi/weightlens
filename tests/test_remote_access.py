from __future__ import annotations

import pytest
from rich.console import Console

from weightlens.cli import run_cli
from weightlens.io.byte_range import ByteRangeReader


def _console() -> Console:
    return Console(record=True, force_terminal=False, color_system=None, width=120)


def test_byte_range_threads_storage_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_url_to_fs(uri: str, **kwargs: object) -> object:
        captured.update(kwargs)
        raise PermissionError("stop after capture")

    monkeypatch.setattr(
        "weightlens.io.byte_range.fsspec.core.url_to_fs", fake_url_to_fs
    )
    with pytest.raises(PermissionError):
        ByteRangeReader("s3://bucket/key", {"anon": True})
    assert captured == {"anon": True}


def test_cli_anon_flag_threads_anon_option(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_url_to_fs(uri: str, **kwargs: object) -> object:
        captured.update(kwargs)
        raise PermissionError("Forbidden")

    monkeypatch.setattr(
        "weightlens.io.byte_range.fsspec.core.url_to_fs", fake_url_to_fs
    )
    code = run_cli(
        ["analyze", "s3://bucket/model.safetensors", "--anon"], console=_console()
    )
    assert captured.get("anon") is True
    assert code == 4  # auth path still returns the auth exit code


def test_cli_auth_error_returns_4_with_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_url_to_fs(uri: str, **kwargs: object) -> object:
        raise PermissionError("Forbidden")

    monkeypatch.setattr(
        "weightlens.io.byte_range.fsspec.core.url_to_fs", fake_url_to_fs
    )
    console = _console()
    code = run_cli(["analyze", "s3://bucket/model.safetensors"], console=console)
    output = console.export_text()
    assert code == 4
    assert "Authentication failed" in output
    assert "--anon" in output


def test_cli_unexpected_error_returns_1_no_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(target: str) -> str:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("weightlens.cli._detect_format_target", boom)
    console = _console()
    code = run_cli(["analyze", "s3://bucket/model.safetensors"], console=console)
    output = console.export_text()
    assert code == 1
    assert "Unexpected error" in output
    assert "Traceback" not in output


@pytest.mark.integration
def test_real_s3_anon_byte_range() -> None:
    pytest.importorskip("s3fs")
    uri = "s3://registry.opendata.aws/roda/ndjson/index.ndjson"
    reader = ByteRangeReader(uri, {"anon": True})
    assert reader.size() > 0
    assert len(reader.read(0, 16)) == 16


@pytest.mark.integration
def test_real_gcs_anon_byte_range() -> None:
    gcsfs = pytest.importorskip("gcsfs")
    fs = gcsfs.GCSFileSystem(token="anon")
    entries = fs.find("cloud-tpu-checkpoints/bert/keras_bert", maxdepth=4)
    files = [e for e in entries if not e.endswith("/")]
    assert files, "no files found under public GCS prefix"
    reader = ByteRangeReader("gs://" + files[0], {"token": "anon"})
    assert reader.size() > 0
    assert len(reader.read(0, 8)) == 8
