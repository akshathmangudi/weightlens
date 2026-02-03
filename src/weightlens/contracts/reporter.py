from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weightlens.models import AnalysisResult


class Reporter(ABC):
    """Render analysis results for presentation."""

    @abstractmethod
    def render(self, result: AnalysisResult, filename: str) -> None:
        """Render the analysis result to the configured output."""
        raise NotImplementedError
