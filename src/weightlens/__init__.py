import logging
import sys

from .contracts import (
    CheckpointValidator,
    DiagnosticRule,
    GlobalAggregator,
    Reporter,
    StatsEngine,
    WeightSource,
)

__all__ = [
    "CheckpointValidator",
    "DiagnosticRule",
    "GlobalAggregator",
    "Reporter",
    "StatsEngine",
    "WeightSource",
]

# NOTE: switch to INFO after verification passes.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
