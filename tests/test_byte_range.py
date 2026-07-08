from __future__ import annotations

import fsspec
import pytest

from weightlens.io.byte_range import ByteRangeReader
from weightlens.io.errors import MissingBackendError
from weightlens.io.uri import is_remote, join_uri, parent_uri, split_protocol


def _write_memory(path: str, data: bytes) -> None:
    with fsspec.open(path, "wb") as f:
        f.write(data)


def test_uri_helpers() -> None:
    assert split_protocol("s3://b/k") == "s3"
    assert split_protocol("/local/x") is None
    assert is_remote("s3://b/k") is True
    assert is_remote("/local/x") is False
    assert is_remote("file:///local/x") is False
    assert join_uri("s3://b/dir", "shard.st") == "s3://b/dir/shard.st"
    assert parent_uri("s3://b/dir/model.index.json") == "s3://b/dir"


def test_byte_range_reads_exact_slice_over_memory() -> None:
    data = bytes(range(256))
    _write_memory("memory://blob.bin", data)
    reader = ByteRangeReader("memory://blob.bin")
    assert reader.size() == 256
    assert reader.read(0, 4) == data[0:4]
    assert reader.read(10, 5) == data[10:15]
    assert reader.read(0, 0) == b""


def test_byte_range_rejects_negative() -> None:
    _write_memory("memory://blob2.bin", b"abc")
    reader = ByteRangeReader("memory://blob2.bin")
    with pytest.raises(ValueError):
        reader.read(-1, 2)


def test_missing_backend_message_names_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate s3fs not installed: url_to_fs raises ImportError for s3://.
    def _boom(uri: str) -> None:
        raise ImportError("no s3fs")

    monkeypatch.setattr("weightlens.io.byte_range.fsspec.core.url_to_fs", _boom)
    with pytest.raises(MissingBackendError) as exc:
        ByteRangeReader("s3://bucket/key")
    assert "weightlens[s3]" in str(exc.value)
