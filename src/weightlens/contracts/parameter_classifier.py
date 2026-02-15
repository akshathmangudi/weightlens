from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weightlens.categories import ParameterCategory


class ParameterClassifier(ABC):
    """Classify a parameter name into a category for bucketed diagnostics."""

    @abstractmethod
    def classify(
        self, name: str, shape: tuple[int, ...], dtype: str
    ) -> ParameterCategory:
        """Return the category for a given parameter.

        Classification is based on name, shape, and dtype only —
        never on tensor values (performance guarantee).
        """
        raise NotImplementedError
