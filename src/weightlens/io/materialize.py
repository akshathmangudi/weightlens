from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from weightlens.io.errors import MissingBackendError
from weightlens.io.uri import extra_for_uri, is_remote

try:
    import fsspec
except ImportError:  # pragma: no cover
    fsspec = None

logger = logging.getLogger(__name__)


def materialize(uri: str) -> Path:
    """Return a local path for *uri*, downloading remote inputs to a temp cache.

    Local paths (and ``file://`` URIs) are returned as-is. Remote URIs are
    downloaded in full — the honest fallback for formats (like ``.pth``) whose
    layout is not byte-rangeable.
    """
    if not is_remote(uri):
        return Path(uri[len("file://") :] if uri.startswith("file://") else uri)

    if fsspec is None:
        raise MissingBackendError(
            "Downloading remote checkpoints requires fsspec. "
            "Install it with: pip install weightlens[remote]"
        )
    try:
        fs, path = fsspec.core.url_to_fs(uri)
    except ImportError as exc:
        raise MissingBackendError(
            "Reading this URI needs an extra backend. "
            f"Install it with: pip install weightlens[{extra_for_uri(uri)}]"
        ) from exc

    cache_dir = Path(tempfile.mkdtemp(prefix="weightlens-"))
    local = cache_dir / Path(path).name
    logger.info("Materializing %s -> %s", uri, local)
    fs_any: Any = fs
    fs_any.get_file(path, str(local))
    return local
