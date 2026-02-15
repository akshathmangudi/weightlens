from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from weightlens.aggregators import StreamingGlobalAggregator
from weightlens.analyzer import Analyzer
from weightlens.classifiers import PyTorchParameterClassifier
from weightlens.contracts import CheckpointValidator
from weightlens.diagnostics import (
    AbnormalNormRule,
    DeadLayerRule,
    ExplodingVarianceRule,
    ExtremeSpikeRule,
)
from weightlens.models import CheckpointHealth
from weightlens.reporters import RichReporter
from weightlens.sources import PyTorchWeightSource
from weightlens.stats_engines import BasicStatsEngine
from weightlens.validators import PyTorchCheckpointValidator


class StaticCheckpointValidator(CheckpointValidator):
    """CheckpointValidator that reuses a precomputed health result."""

    def __init__(self, health: CheckpointHealth) -> None:
        self._health = health

    def validate(self) -> CheckpointHealth:
        return self._health


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lens", description="WeightLens CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser(
        "analyze", help="Analyze a PyTorch .pth checkpoint"
    )
    analyze.add_argument(
        "checkpoint",
        type=str,
        help="Path to the checkpoint (.pth) file",
    )
    return parser


def _render_health(console: Console, health: CheckpointHealth) -> None:
    table = Table(title="Checkpoint Health")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("file_size_bytes", str(health.file_size_bytes))
    table.add_row("loadable", str(health.loadable))
    table.add_row("is_empty", str(health.is_empty))
    table.add_row("tensor_count", str(health.tensor_count))
    table.add_row("total_params", str(health.total_params))
    flags = ", ".join(health.corruption_flags) if health.corruption_flags else "none"
    table.add_row("corruption_flags", flags)
    console.print(table)


def _run_analyze(checkpoint_path: Path, *, console: Console | None) -> int:
    out_console = console or Console()
    validator = PyTorchCheckpointValidator(checkpoint_path)
    try:
        health = validator.validate()
    except FileNotFoundError:
        out_console.print(f"Checkpoint not found: {checkpoint_path}")
        return 2

    if not health.loadable or health.is_empty:
        _render_health(out_console, health)
        out_console.print("Analysis aborted: checkpoint not loadable or empty.")
        return 1

    analyzer = Analyzer(
        source=PyTorchWeightSource(checkpoint_path),
        validator=StaticCheckpointValidator(health),
        stats_engine=BasicStatsEngine(),
        aggregator=StreamingGlobalAggregator(),
        rules=[
            DeadLayerRule(),
            ExplodingVarianceRule(),
            ExtremeSpikeRule(),
            AbnormalNormRule(),
        ],
        classifier=PyTorchParameterClassifier(),
    )
    result = analyzer.analyze()

    reporter = RichReporter(out_console)
    reporter.render(result, checkpoint_path.name)
    return 0


def run_cli(
    argv: Sequence[str] | None = None, *, console: Console | None = None
) -> int:
    logging.getLogger().setLevel(logging.ERROR)
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        return _run_analyze(Path(args.checkpoint), console=console)
    parser.error("Unknown command.")
    return 2


def main() -> None:
    raise SystemExit(run_cli())
