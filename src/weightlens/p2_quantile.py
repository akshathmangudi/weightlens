from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _float_list() -> list[float]:
    return []


def _int_list() -> list[int]:
    return []


@dataclass
class P2QuantileEstimator:
    """P² quantile estimator with constant memory."""

    quantile: float
    _count: int = 0
    _initial: list[float] = field(default_factory=_float_list)
    _positions: list[int] = field(default_factory=_int_list)
    _desired: list[float] = field(default_factory=_float_list)
    _heights: list[float] = field(default_factory=_float_list)
    _increments: list[float] = field(default_factory=_float_list)

    def __post_init__(self) -> None:
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        self._increments = [
            0.0,
            self.quantile / 2.0,
            self.quantile,
            (1.0 + self.quantile) / 2.0,
            1.0,
        ]

    def update(self, value: float) -> None:
        if self._count < 5:
            self._initial.append(value)
            self._count += 1
            if self._count == 5:
                self._initialize_markers()
            return

        self._count += 1
        k = self._find_cell(value)
        for index in range(k + 1, 5):
            self._positions[index] += 1
        for index in range(5):
            self._desired[index] += self._increments[index]
        for index in range(1, 4):
            self._adjust_marker(index)

    def value(self) -> float:
        if self._count == 0:
            raise ValueError("No samples provided for quantile estimation.")
        if self._count <= 5:
            return self._exact_quantile(self._initial, self.quantile)
        return float(self._heights[2])

    def _initialize_markers(self) -> None:
        self._initial = sorted(self._initial)
        self._heights = list(self._initial)
        self._positions = [1, 2, 3, 4, 5]
        q = self.quantile
        self._desired = [
            1.0,
            1.0 + 2.0 * q,
            1.0 + 4.0 * q,
            3.0 + 2.0 * q,
            5.0,
        ]

    def _find_cell(self, value: float) -> int:
        if value < self._heights[0]:
            self._heights[0] = value
            return 0
        if value < self._heights[1]:
            return 0
        if value < self._heights[2]:
            return 1
        if value < self._heights[3]:
            return 2
        if value <= self._heights[4]:
            return 3
        self._heights[4] = value
        return 3

    def _adjust_marker(self, index: int) -> None:
        desired_delta = self._desired[index] - self._positions[index]
        if desired_delta >= 1.0:
            step = 1
        elif desired_delta <= -1.0:
            step = -1
        else:
            return

        if step == 1 and self._positions[index + 1] - self._positions[index] <= 1:
            return
        if step == -1 and self._positions[index - 1] - self._positions[index] >= -1:
            return

        proposal = self._parabolic(index, step)
        if self._heights[index - 1] < proposal < self._heights[index + 1]:
            self._heights[index] = proposal
        else:
            self._heights[index] = self._linear(index, step)
        self._positions[index] += step

    def _parabolic(self, index: int, step: int) -> float:
        positions = self._positions
        heights = self._heights
        return heights[index] + step / (positions[index + 1] - positions[index - 1]) * (
            (positions[index] - positions[index - 1] + step)
            * (heights[index + 1] - heights[index])
            / (positions[index + 1] - positions[index])
            + (positions[index + 1] - positions[index] - step)
            * (heights[index] - heights[index - 1])
            / (positions[index] - positions[index - 1])
        )

    def _linear(self, index: int, step: int) -> float:
        positions = self._positions
        heights = self._heights
        return heights[index] + step * (
            (heights[index + step] - heights[index])
            / (positions[index + step] - positions[index])
        )

    @staticmethod
    def _exact_quantile(values: list[float], quantile: float) -> float:
        data = np.asarray(values, dtype=np.float64)
        return float(np.quantile(data, quantile, method="linear"))
