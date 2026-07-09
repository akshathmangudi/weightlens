from __future__ import annotations

from typing import Any

from weightlens.io.errors import MissingBackendError
from weightlens.io.uri import extra_for_uri

try:
    import fsspec
except ImportError:  # pragma: no cover - exercised via monkeypatch/uninstalled env
    fsspec = None


class ByteRangeReader:
    """Read exact byte ranges from any fsspec-supported storage backend."""

    def __init__(
        self, uri: str, storage_options: dict[str, object] | None = None
    ) -> None:
        self.uri = uri
        if fsspec is None:
            raise MissingBackendError(
                "Remote/streaming IO requires fsspec. "
                "Install it with: pip install weightlens[remote]"
            )
        try:
            fs, path = fsspec.core.url_to_fs(uri, **(storage_options or {}))
        except ImportError as exc:  # backend (s3fs/gcsfs) missing
            raise MissingBackendError(
                "Reading this URI needs an extra backend. "
                f"Install it with: pip install weightlens[{extra_for_uri(uri)}]"
            ) from exc
        self._fs: Any = fs
        self._path: str = path

    def read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0:
            raise ValueError("offset and length must be non-negative")
        if length == 0:
            return b""
        data: bytes = self._fs.cat_file(self._path, start=offset, end=offset + length)
        return data

    def size(self) -> int:
        return int(self._fs.info(self._path)["size"])
