from __future__ import annotations

import argparse
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from weightlens.aggregators import StreamingGlobalAggregator
from weightlens.analyzer import Analyzer
from weightlens.classifiers import DCPParameterClassifier, PyTorchParameterClassifier
from weightlens.contracts import CheckpointValidator
from weightlens.diagnostics import (
    AbnormalNormRule,
    DeadLayerRule,
    ExplodingVarianceRule,
    ExtremeSpikeRule,
)
from weightlens.io import materialize
from weightlens.io.errors import MissingBackendError
from weightlens.io.uri import anon_storage_options, is_remote
from weightlens.models import CheckpointHealth
from weightlens.reporters import RichReporter
from weightlens.sources import PyTorchWeightSource, SafetensorsWeightSource
from weightlens.stats_engines import BasicStatsEngine
from weightlens.validators import (
    PyTorchCheckpointValidator,
    SafetensorsCheckpointValidator,
)

logger = logging.getLogger(__name__)


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
        "analyze",
        help=(
            "Analyze a checkpoint "
            "(.pth/.safetensors file, .index.json, or DCP directory)"
        ),
    )
    analyze.add_argument(
        "checkpoint",
        type=str,
        help=(
            "Path or URI to the checkpoint (e.g., model.pth, model.safetensors, "
            "model.safetensors.index.json, s3://bucket/model.safetensors, "
            "or DCP directory)"
        ),
    )
    analyze.add_argument(
        "--format",
        type=str,
        choices=["pytorch", "dcp", "safetensors", "auto"],
        default="auto",
        help="Checkpoint format (default: auto-detect)",
    )
    analyze.add_argument(
        "--include-optimizer",
        action="store_true",
        default=False,
        help="Include optimizer state tensors in DCP analysis (default: skip them)",
    )
    analyze.add_argument(
        "--num-workers",
        type=_parse_num_workers,
        default="1",
        help="Number of parallel stats workers ('auto' or integer, default: 1)",
    )
    analyze.add_argument(
        "--anon",
        action="store_true",
        default=False,
        help="Use anonymous access for remote reads (public buckets)",
    )
    return parser


def _detect_format(path: Path) -> str:
    """Auto-detect checkpoint format from *path*.

    Returns ``'pytorch'`` for files and ``'dcp'`` for directories that
    contain a recognised metadata file.
    """
    if path.is_file():
        return "pytorch"
    if path.is_dir():
        from weightlens.sources.dcp import find_metadata_path

        try:
            find_metadata_path(path)
            return "dcp"
        except FileNotFoundError:
            raise ValueError(
                f"Directory {path} does not contain a metadata file. "
                "Not a recognised DCP checkpoint."
            ) from None
    raise FileNotFoundError(f"Path does not exist: {path}")


def _detect_format_target(target: str) -> str:
    """Detect format from a path or URI string (extension-based for remote)."""
    if target.endswith(".index.json") or target.endswith(".safetensors"):
        return "safetensors"
    if target.endswith((".pth", ".pt")):
        return "pytorch"
    if is_remote(target):
        raise ValueError(
            f"Cannot infer format for remote target {target!r}. "
            "Pass --format safetensors or a .safetensors/.index.json/.pth URI."
        )
    return _detect_format(Path(target))


def _render_health(console: Console, health: CheckpointHealth) -> None:
    table = Table(title="Checkpoint Health")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("file_size_bytes", str(health.file_size_bytes))
    table.add_row("loadable", str(health.loadable))
    table.add_row("is_empty", str(health.is_empty))
    table.add_row("tensor_count", str(health.tensor_count))
    table.add_row("total_params", str(health.total_params))
    flags = (
        ", ".join(escape(f) for f in health.corruption_flags)
        if health.corruption_flags
        else "none"
    )
    table.add_row("corruption_flags", flags)
    console.print(table)


def _parse_num_workers(value: str) -> int | None:
    if value == "auto":
        return None
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--num-workers must be 'auto' or an integer, got {value!r}"
        ) from None
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"--num-workers must be >= 1, got {n}"
        )
    return n


def _run_analyze_pytorch(
    checkpoint_path: Path,
    *,
    console: Console,
    num_workers: int | None = None,
) -> int:
    validator = PyTorchCheckpointValidator(checkpoint_path)
    try:
        health = validator.validate()
    except FileNotFoundError:
        console.print(f"Checkpoint not found: {escape(str(checkpoint_path))}")
        return 2

    if not health.loadable or health.is_empty:
        _render_health(console, health)
        console.print("Analysis aborted: checkpoint not loadable or empty.")
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
        num_workers=num_workers,
    )
    result = analyzer.analyze()

    reporter = RichReporter(console)
    reporter.render(result, checkpoint_path.name)
    return 0


def _run_analyze_dcp(
    checkpoint_path: Path,
    *,
    console: Console,
    include_optimizer: bool = False,
    num_workers: int | None = None,
) -> int:
    from weightlens.sources.dcp import DCPWeightSource
    from weightlens.validators.dcp_checkpoint import DCPCheckpointValidator

    console.print(
        "[yellow]EXPERIMENTAL[/yellow]: DCP checkpoint support is alpha. "
        "Tensors are streamed one at a time via byte-offset reads."
    )

    validator = DCPCheckpointValidator(checkpoint_path)
    try:
        health = validator.validate()
    except FileNotFoundError:
        console.print(f"Checkpoint directory not found: {escape(str(checkpoint_path))}")
        return 2

    if not health.loadable or health.is_empty:
        _render_health(console, health)
        console.print("Analysis aborted: checkpoint not loadable or empty.")
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyzer = Analyzer(
            source=DCPWeightSource(checkpoint_path),
            validator=StaticCheckpointValidator(health),
            stats_engine=BasicStatsEngine(),
            aggregator=StreamingGlobalAggregator(),
            rules=[
                DeadLayerRule(),
                ExplodingVarianceRule(),
                ExtremeSpikeRule(),
                AbnormalNormRule(),
            ],
            classifier=DCPParameterClassifier(include_optimizer=include_optimizer),
            num_workers=num_workers,
        )
        result = analyzer.analyze()

    reporter = RichReporter(console)
    reporter.render(result, checkpoint_path.name)
    return 0


def _run_analyze_safetensors(
    uri: str,
    *,
    console: Console,
    num_workers: int | None = None,
    storage_options: dict[str, object] | None = None,
) -> int:
    validator = SafetensorsCheckpointValidator(uri, storage_options)
    try:
        health = validator.validate()
    except FileNotFoundError:
        console.print(f"Checkpoint not found: {escape(uri)}")
        return 2

    if not health.loadable or health.is_empty:
        _render_health(console, health)
        console.print("Analysis aborted: checkpoint not loadable or empty.")
        return 1

    analyzer = Analyzer(
        source=SafetensorsWeightSource(uri, storage_options),
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
        num_workers=num_workers,
    )
    result = analyzer.analyze()
    RichReporter(console).render(result, uri.rsplit("/", 1)[-1])
    return 0


def _run_analyze(
    target: str,
    *,
    fmt: str = "auto",
    include_optimizer: bool = False,
    num_workers: int | None = None,
    anon: bool = False,
    console: Console | None,
) -> int:
    out_console = console or Console()
    storage_options = anon_storage_options(target) if anon else None
    try:
        resolved_fmt = _detect_format_target(target) if fmt == "auto" else fmt

        if resolved_fmt == "safetensors":
            return _run_analyze_safetensors(
                target,
                console=out_console,
                num_workers=num_workers,
                storage_options=storage_options,
            )
        if resolved_fmt == "dcp":
            return _run_analyze_dcp(
                Path(target),
                console=out_console,
                include_optimizer=include_optimizer,
                num_workers=num_workers,
            )
        local_path = (
            materialize(target, storage_options) if is_remote(target)
            else Path(target)
        )
        return _run_analyze_pytorch(
            local_path, console=out_console, num_workers=num_workers
        )
    except MissingBackendError as exc:
        out_console.print(f"[red]{escape(str(exc))}[/red]")
        return 3
    except PermissionError:
        out_console.print(
            f"[red]Authentication failed for {escape(target)}.[/red]\n"
            "Configure credentials the standard way (AWS: aws configure, "
            "AWS_* env vars, or an IAM role; GCS: gcloud auth "
            "application-default login or GOOGLE_APPLICATION_CREDENTIALS), "
            "or pass --anon for public buckets."
        )
        return 4
    except (FileNotFoundError, ValueError) as exc:
        out_console.print(escape(str(exc)))
        return 2
    except Exception as exc:  # never crash the CLI with a traceback
        logger.debug("Unexpected error analyzing %s", target, exc_info=True)
        out_console.print(
            f"[red]Unexpected error analyzing {escape(target)}: "
            f"{escape(str(exc))}[/red]"
        )
        return 1


def run_cli(
    argv: Sequence[str] | None = None, *, console: Console | None = None
) -> int:
    logging.getLogger().setLevel(logging.ERROR)
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        return _run_analyze(
            args.checkpoint,  # keep as str; URIs must not be wrapped in Path
            fmt=args.format,
            include_optimizer=args.include_optimizer,
            num_workers=args.num_workers,
            anon=args.anon,
            console=console,
        )
    parser.error("Unknown command.")
    return 2


def main() -> None:
    raise SystemExit(run_cli())
