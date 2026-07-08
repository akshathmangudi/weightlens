from __future__ import annotations

from pathlib import Path

import fsspec

from weightlens.io.materialize import materialize


def test_materialize_local_is_identity(tmp_path: Path) -> None:
    p = tmp_path / "x.pth"
    p.write_bytes(b"hello")
    assert materialize(str(p)) == p


def test_materialize_strips_file_scheme(tmp_path: Path) -> None:
    p = tmp_path / "y.pth"
    p.write_bytes(b"hi")
    assert materialize(f"file://{p}") == Path(str(p))


def test_materialize_downloads_remote(tmp_path: Path) -> None:
    with fsspec.open("memory://remote/model.pth", "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    local = materialize("memory://remote/model.pth")
    assert local.exists()
    assert local.read_bytes() == b"\x00\x01\x02\x03"
    assert local.name == "model.pth"
