from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weightlens.models import CheckpointHealth


class CheckpointValidator(ABC):
    """Validate checkpoint integrity without batch loading."""

    @abstractmethod
    def validate(self) -> "CheckpointHealth":
        """Return a CheckpointHealth summary for the checkpoint."""
        raise NotImplementedError
