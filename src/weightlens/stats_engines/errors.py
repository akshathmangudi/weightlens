from __future__ import annotations


class EmptyTensorError(ValueError):
    """A tensor contains no values for statistical analysis."""


class NonFiniteTensorError(ValueError):
    """A tensor contains NaN or infinite values."""
