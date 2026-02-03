import logging
import sys

from .contracts import (
    CheckpointValidator,
    DiagnosticRule,
    GlobalAggregator,
    StatsEngine,
    WeightSource,
)

__all__ = [
    "CheckpointValidator",
    "DiagnosticRule",
    "GlobalAggregator",
    "StatsEngine",
    "WeightSource",
]

# NOTE: switch to INFO after verification passes.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
