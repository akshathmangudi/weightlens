from __future__ import annotations

from typing import Literal

ParameterCategory = Literal[
    "kernel",
    "bias",
    "norm_scale",
    "norm_shift",
    "embedding",
    "buffer",
    "adapter",
    "skip",
]

ALL_DIAGNOSTIC_CATEGORIES: frozenset[ParameterCategory] = frozenset(
    ["kernel", "bias", "norm_scale", "norm_shift", "embedding", "buffer", "adapter"]
)  # everything except "skip"
