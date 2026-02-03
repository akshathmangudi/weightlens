from __future__ import annotations

from collections.abc import Iterable

from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from weightlens.contracts import Reporter
from weightlens.models import (
    AnalysisResult,
    CheckpointHealth,
    DiagnosticFlag,
    GlobalStats,
)


class RichReporter(Reporter):
    """Render analysis results using Rich tables."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def render(self, result: AnalysisResult, filename: str) -> None:
        self._console.print()
        self._console.print(f"Statistics for {filename}", style="bold underline")
        self._console.print(Rule(style="dim"))

        self._console.print(self._build_health_section(result.health))
        self._console.print()

        self._console.print("Global Stats", style="bold")
        self._console.print(Rule(style="dim"))
        self._console.print(self._build_global_stats_section(result.global_stats))
        self._console.print()

        self._render_diagnostics(result.diagnostics)

    @staticmethod
    def _build_health_section(health: CheckpointHealth) -> Table:
        table = Table.grid(padding=(0, 3))
        table.add_column("Field", style="bold cyan")
        table.add_column("Value")

        table.add_row("file_size_bytes:", f"{health.file_size_bytes:,}")
        table.add_row("loadable:", str(health.loadable).lower())
        table.add_row("is_empty:", str(health.is_empty).lower())
        table.add_row("tensor_count:", f"{health.tensor_count:,}")
        table.add_row("total_params:", f"{health.total_params:,}")

        flags_color = "green" if not health.corruption_flags else "red bold"
        flags = (
            ", ".join(health.corruption_flags)
            if health.corruption_flags
            else "none"
        )
        table.add_row("corruption_flags:", Text(flags, style=flags_color))
        return table

    @staticmethod
    def _build_global_stats_section(global_stats: GlobalStats) -> Table:
        table = Table.grid(padding=(0, 3))
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value")

        table.add_row("mean:", f"{global_stats.mean:.6f}")
        table.add_row("std:", f"{global_stats.std:.6f}")
        table.add_row("p99:", f"{global_stats.p99:.6f}")
        table.add_row(
            "median_layer_variance:",
            f"{global_stats.median_layer_variance:.6f}",
        )
        table.add_row(
            "median_layer_norm:",
            f"{global_stats.median_layer_norm:.6f}",
        )
        table.add_row("iqr_layer_norm:", f"{global_stats.iqr_layer_norm:.6f}")
        return table

    def _render_diagnostics(self, diagnostics: Iterable[DiagnosticFlag]) -> None:
        diagnostics_list = list(diagnostics)
        count = len(diagnostics_list)

        header = f"Diagnostics ({count})"
        self._console.print(header, style="bold")
        self._console.print(Rule(style="dim"))

        if not diagnostics_list:
            self._console.print("[dim]None[/dim]")
            return

        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold",
            expand=True,
            padding=(0, 2),
        )
        table.add_column("Severity", ratio=1)
        table.add_column("Rule", ratio=2)
        table.add_column("Layer", ratio=4)
        table.add_column("Message", ratio=4)

        for flag in diagnostics_list:
            severity_style = "red bold" if flag.severity == "error" else "yellow"
            table.add_row(
                Text(flag.severity, style=severity_style),
                flag.rule,
                flag.layer,
                flag.message,
            )
        self._console.print(table)
