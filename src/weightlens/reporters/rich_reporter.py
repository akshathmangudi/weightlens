from __future__ import annotations

from collections.abc import Iterable, Mapping

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from weightlens.contracts import Reporter
from weightlens.models import (
    AnalysisCoverage,
    AnalysisResult,
    CheckpointHealth,
    DiagnosticFlag,
    GlobalStats,
    TensorOmission,
)


class RichReporter(Reporter):
    """Render analysis results using Rich tables."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        thresholds: Mapping[str, float | str] | None = None,
    ) -> None:
        self._console = console or Console()
        self._thresholds = thresholds

    def render(self, result: AnalysisResult, filename: str) -> None:
        self._console.print()
        self._console.print(
            f"Statistics for {escape(filename)}", style="bold underline"
        )
        self._console.print(Rule(style="dim"))

        self._console.print(self._build_health_section(result.health))
        self._console.print()

        self._render_coverage(result.coverage)

        self._console.print("Global Stats (analyzed tensors only)", style="bold")
        self._console.print(Rule(style="dim"))
        self._console.print(self._build_global_stats_section(result.global_stats))
        self._console.print()

        if self._thresholds is not None:
            self._console.print("Diagnostic thresholds", style="bold")
            self._console.print(Rule(style="dim"))
            self._console.print(self._build_thresholds_section(self._thresholds))
            self._console.print()

        self._render_diagnostics(result.diagnostics, partial=result.coverage.partial)

    def _render_coverage(self, coverage: AnalysisCoverage) -> None:
        self._console.print(
            "Inspection: PARTIAL" if coverage.partial else "Inspection: COMPLETE",
            style="yellow bold" if coverage.partial else "green",
        )
        excluded = sum(item.kind == "excluded" for item in coverage.omissions)
        skipped = sum(item.kind == "skipped" for item in coverage.omissions)
        self._console.print(
            f"Coverage: {coverage.analyzed} of {coverage.eligible} eligible "
            f"tensors analyzed; {excluded} excluded; {skipped} skipped."
        )
        if coverage.omissions:
            self._console.print(self._build_omissions_section(coverage.omissions))
        self._console.print()

    @staticmethod
    def _build_omissions_section(omissions: list[TensorOmission]) -> Table:
        table = Table(box=box.SIMPLE_HEAD, show_header=True, expand=True)
        table.add_column("Status")
        table.add_column("Tensor")
        table.add_column("Reason")
        table.add_column("Dtype")
        for item in omissions:
            table.add_row(
                Text(item.kind), Text(item.name), Text(item.reason),
                Text(item.dtype or "unknown"),
            )
        return table

    @staticmethod
    def _build_thresholds_section(thresholds: Mapping[str, float | str]) -> Table:
        table = Table.grid(padding=(0, 3))
        table.add_column("Flag", style="bold cyan")
        table.add_column("Value")
        for name, value in thresholds.items():
            table.add_row(name, str(value))
        return table

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

        table.add_row("mean:", f"{global_stats.mean:.6g}")
        table.add_row("std:", f"{global_stats.std:.6g}")
        table.add_row("p99:", f"{global_stats.p99:.6g}")
        table.add_row(
            "median_layer_variance:",
            f"{global_stats.median_layer_variance:.6g}",
        )
        table.add_row(
            "median_layer_norm:",
            f"{global_stats.median_layer_norm:.6g}",
        )
        table.add_row("iqr_layer_norm:", f"{global_stats.iqr_layer_norm:.6g}")
        return table

    def _render_diagnostics(
        self, diagnostics: Iterable[DiagnosticFlag], *, partial: bool = False
    ) -> None:
        diagnostics_list = list(diagnostics)
        count = len(diagnostics_list)

        header = f"Diagnostics ({count})"
        self._console.print(header, style="bold")
        self._console.print(Rule(style="dim"))

        if not diagnostics_list:
            self._console.print("[dim]None[/dim]")
        else:
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
                severity_style = (
                    "red bold" if flag.severity == "error" else "yellow"
                )
                table.add_row(
                    Text(flag.severity, style=severity_style),
                    Text(flag.rule),
                    Text(flag.layer),
                    Text(flag.message),
                )
            self._console.print(table)

        self._console.print()
        self._console.print(self._build_summary(diagnostics_list, partial=partial))

    @staticmethod
    def _build_summary(
        diagnostics: list[DiagnosticFlag], *, partial: bool = False
    ) -> Panel:
        errors = sum(1 for d in diagnostics if d.severity == "error")
        warns = sum(1 for d in diagnostics if d.severity == "warn")
        infos = sum(1 for d in diagnostics if d.severity == "info")

        if diagnostics:
            parts = []
            if errors > 0:
                parts.append(
                    f"{errors} error-level finding{'s' if errors != 1 else ''}"
                )
            if warns > 0:
                parts.append(
                    f"{warns} warning{'s' if warns != 1 else ''}"
                )
            if infos > 0:
                parts.append(f"{infos} info finding{'s' if infos != 1 else ''}")
            return Panel(
                ", ".join(parts),
                style="red bold" if errors else "yellow",
                title="FINDINGS",
                title_align="left",
            )
        return Panel(
            "No findings in analyzed tensors" if partial else "No diagnostic findings",
            style="green",
            title="NO FINDINGS",
            title_align="left",
        )
