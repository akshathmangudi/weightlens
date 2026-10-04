from __future__ import annotations

import pytest
from rich.console import Console

from weightlens.models import DiagnosticFlag, GlobalStats
from weightlens.reporters.rich_reporter import RichReporter


def _console() -> Console:
    return Console(record=True, force_terminal=False, color_system=None, width=300)


def test_diagnostics_table_does_not_swallow_bracketed_layer_name() -> None:
    # A tensor name is checkpoint-controlled and can contain '[...]'. Rendered
    # as bare Rich markup those segments are silently dropped; the reporter must
    # wrap cells in Text() so they survive verbatim.
    console = _console()
    reporter = RichReporter(console)
    flag = DiagnosticFlag(
        layer="model.layers[0].attn[q_proj].weight",
        rule="dead-layer",
        message="layer is all zeros",
        severity="error",
    )

    reporter._render_diagnostics([flag])

    output = console.export_text()
    assert "[0]" in output
    assert "[q_proj]" in output
    assert "dead-layer" in output


def test_global_stats_report_keeps_small_nonzero_values() -> None:
    console = _console()
    stats = GlobalStats(
        mean=1e-8,
        std=2e-8,
        p1=1e-8,
        p5=1e-8,
        p50=1e-8,
        p95=2e-8,
        p99=2e-8,
        median_layer_variance=1e-16,
        median_layer_norm=1e-8,
        iqr_layer_norm=1e-9,
    )

    console.print(RichReporter._build_global_stats_section(stats))

    displayed = {
        name: float(value.strip())
        for name, value in (
            line.split(":", 1) for line in console.export_text().splitlines()
        )
    }
    assert displayed["mean"] == pytest.approx(stats.mean, rel=1e-5)
    assert displayed["p99"] == pytest.approx(stats.p99, rel=1e-5)
    assert displayed["median_layer_variance"] == pytest.approx(
        stats.median_layer_variance, rel=1e-5
    )
