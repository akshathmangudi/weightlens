from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


@dataclass
class LayerTensor:
    """Numeric payload for a single tensor layer."""

    name: str
    values: NDArray[np.number]
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not isinstance(cast(Any, self.values), np.ndarray):
            raise TypeError("values must be a numpy.ndarray")
        if not isinstance(cast(Any, self.shape), tuple):
            raise TypeError("shape must be a tuple")


class CheckpointHealth(BaseModel):
    """Integrity summary for a checkpoint."""

    model_config = ConfigDict(extra="forbid", strict=True)

    file_size_bytes: int
    is_empty: bool
    loadable: bool
    tensor_count: int
    total_params: int
    corruption_flags: list[str]


class LayerStats(BaseModel):
    """Per-layer statistical metrics."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    category: str = "kernel"
    mean: float
    std: float
    min: float
    max: float
    l2_norm: float
    sparsity: float
    param_count: int
    p99_abs: float


class GlobalStats(BaseModel):
    """Streaming global statistical metrics."""

    model_config = ConfigDict(extra="forbid", strict=True)

    mean: float
    std: float
    p1: float
    p5: float
    p50: float
    p95: float
    p99: float
    median_layer_variance: float
    median_layer_norm: float
    iqr_layer_norm: float


class DiagnosticFlag(BaseModel):
    """Structured diagnostic signal for a layer."""

    model_config = ConfigDict(extra="forbid", strict=True)

    layer: str
    rule: str
    message: str
    severity: str


class AnalysisResult(BaseModel):
    """Immutable analysis result bundle."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    layer_stats: list[LayerStats]
    global_stats: GlobalStats
    bucket_stats: dict[str, GlobalStats]
    diagnostics: list[DiagnosticFlag]
    health: CheckpointHealth
