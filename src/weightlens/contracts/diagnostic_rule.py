from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from weightlens.models import DiagnosticFlag, GlobalStats, LayerStats


class DiagnosticRule(ABC):
    """Interpret layer/global stats into a diagnostic flag."""

    @property
    @abstractmethod
    def severity(self) -> Literal["info", "warn", "error"]:
        """Severity label for the rule."""
        raise NotImplementedError

    @abstractmethod
    def check(
        self, layer: LayerStats, global_stats: GlobalStats
    ) -> DiagnosticFlag | None:
        """Return a DiagnosticFlag if the rule triggers."""
        raise NotImplementedError
