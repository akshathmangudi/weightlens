from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from weightlens.categories import ALL_DIAGNOSTIC_CATEGORIES

if TYPE_CHECKING:
    from weightlens.categories import ParameterCategory
    from weightlens.models import DiagnosticFlag, GlobalStats, LayerStats


class DiagnosticRule(ABC):
    """Interpret layer/global stats into a diagnostic flag."""

    @property
    @abstractmethod
    def severity(self) -> Literal["info", "warn", "error"]:
        """Severity label for the rule."""
        raise NotImplementedError

    @property
    def applicable_categories(self) -> frozenset[ParameterCategory]:
        """Parameter categories this rule should be evaluated against.

        Override in subclasses to restrict scope.  Default: all diagnostic
        categories (everything except ``"skip"``).
        """
        return ALL_DIAGNOSTIC_CATEGORIES

    @abstractmethod
    def check(
        self, layer: LayerStats, global_stats: GlobalStats
    ) -> DiagnosticFlag | None:
        """Return a DiagnosticFlag if the rule triggers."""
        raise NotImplementedError
