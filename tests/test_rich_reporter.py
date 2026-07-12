from __future__ import annotations

from rich.console import Console

from weightlens.models import DiagnosticFlag
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
