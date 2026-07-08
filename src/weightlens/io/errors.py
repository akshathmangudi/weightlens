from __future__ import annotations


class MissingBackendError(RuntimeError):
    """A remote URI needs an fsspec backend that is not installed."""
