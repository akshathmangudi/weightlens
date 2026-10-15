from __future__ import annotations

import pytest
from rich.console import Console

from weightlens.models import (
    AnalysisCoverage,
    DiagnosticFlag,
    GlobalStats,
    TensorOmission,
)
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


def test_error_finding_does_not_label_completed_analysis_failed() -> None:
    console = _console()
    flag = DiagnosticFlag(
        layer="model.weight",
        rule="extreme-spike",
        message="spike_ratio=101 >= 100",
        severity="error",
    )

    RichReporter(console)._render_diagnostics([flag])

    output = console.export_text()
    assert "FINDINGS" in output
    assert "1 error-level finding" in output
    assert "FAILED" not in output


def test_no_diagnostics_do_not_claim_model_passed() -> None:
    console = _console()

    RichReporter(console)._render_diagnostics([])

    output = console.export_text()
    assert "NO FINDINGS" in output
    assert "PASSED" not in output


def test_partial_coverage_names_skipped_and_excluded_tensors() -> None:
    console = _console()
    reporter = RichReporter(console)
    coverage = AnalysisCoverage(
        analyzed=1,
        eligible=2,
        omissions=[
            TensorOmission(
                name="block[0].weight", kind="skipped",
                reason="NaN or Inf values", dtype="float32",
            ),
            TensorOmission(
                name="step", kind="excluded",
                reason="non-floating tensor", dtype="int64",
            ),
        ],
    )

    reporter._render_coverage(coverage)
    reporter._render_diagnostics([], partial=coverage.partial)

    output = console.export_text()
    assert "Inspection: PARTIAL" in output
    assert "1 of 2 eligible tensors analyzed" in output
    assert "block[0].weight" in output
    assert "NaN or Inf values" in output
    assert "step" in output
    assert "No findings in analyzed tensors" in output


def test_warning_only_report_still_shows_findings() -> None:
    console = _console()
    flag = DiagnosticFlag(
        layer="model.weight",
        rule="exploding-variance",
        message="variance_ratio=11 >= 10",
        severity="warn",
    )

    RichReporter(console)._render_diagnostics([flag])

    output = console.export_text()
    assert "FINDINGS" in output
    assert "1 warning" in output


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
