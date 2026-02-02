import pytest
from pydantic import ValidationError

from weightlens.models import (
    AnalysisResult,
    CheckpointHealth,
    DiagnosticFlag,
    GlobalStats,
    LayerStats,
)


def make_checkpoint_health() -> CheckpointHealth:
    return CheckpointHealth(
        file_size_bytes=1024,
        is_empty=False,
        loadable=True,
        tensor_count=3,
        total_params=42,
        corruption_flags=["none"],
    )


def make_layer_stats() -> LayerStats:
    return LayerStats(
        name="encoder.weight",
        mean=0.01,
        std=0.1,
        min=-0.5,
        max=0.6,
        l2_norm=12.5,
        sparsity=0.0,
        param_count=128,
    )


def make_global_stats() -> GlobalStats:
    return GlobalStats(
        mean=0.02,
        std=0.2,
        p1=-0.4,
        p5=-0.2,
        p50=0.0,
        p95=0.3,
        p99=0.5,
    )


def make_diagnostic_flag() -> DiagnosticFlag:
    return DiagnosticFlag(
        layer="encoder.weight",
        rule="dead-layer",
        message="near-zero variance",
        severity="warn",
    )


def test_checkpoint_health_accepts_valid_payload() -> None:
    health = make_checkpoint_health()

    assert health.file_size_bytes == 1024
    assert health.tensor_count == 3
    assert health.corruption_flags == ["none"]


def test_checkpoint_health_rejects_missing_fields() -> None:
    with pytest.raises(ValidationError):
        CheckpointHealth.model_validate(
            {
                "file_size_bytes": 1024,
                "is_empty": False,
                "tensor_count": 3,
                "total_params": 42,
                "corruption_flags": ["none"],
            }
        )


def test_checkpoint_health_rejects_invalid_types() -> None:
    with pytest.raises(ValidationError):
        CheckpointHealth(
            file_size_bytes="1024",  # type: ignore[arg-type]
            is_empty=False,
            loadable=True,
            tensor_count=3,
            total_params=42,
            corruption_flags=["none"],
        )


def test_layer_stats_accepts_valid_payload() -> None:
    stats = make_layer_stats()

    assert stats.name == "encoder.weight"
    assert stats.param_count == 128


def test_layer_stats_rejects_invalid_types() -> None:
    with pytest.raises(ValidationError):
        LayerStats(
            name="encoder.weight",
            mean="0.01",  # type: ignore[arg-type]
            std=0.1,
            min=-0.5,
            max=0.6,
            l2_norm=12.5,
            sparsity=0.0,
            param_count=128,
        )


def test_global_stats_accepts_valid_payload() -> None:
    stats = make_global_stats()

    assert stats.p50 == 0.0
    assert stats.p99 == 0.5


def test_global_stats_rejects_missing_fields() -> None:
    with pytest.raises(ValidationError):
        GlobalStats.model_validate(
            {
                "mean": 0.02,
                "std": 0.2,
                "p1": -0.4,
                "p5": -0.2,
                "p50": 0.0,
                "p95": 0.3,
            }
        )


def test_diagnostic_flag_accepts_valid_payload() -> None:
    flag = make_diagnostic_flag()

    assert flag.rule == "dead-layer"
    assert flag.severity == "warn"


def test_diagnostic_flag_rejects_invalid_types() -> None:
    with pytest.raises(ValidationError):
        DiagnosticFlag(
            layer="encoder.weight",
            rule="dead-layer",
            message="near-zero variance",
            severity=2,  # type: ignore[arg-type]
        )


def test_analysis_result_accepts_valid_payload() -> None:
    result = AnalysisResult(
        layer_stats=[make_layer_stats()],
        global_stats=make_global_stats(),
        diagnostics=[make_diagnostic_flag()],
        health=make_checkpoint_health(),
    )

    assert result.global_stats.mean == 0.02
    assert result.layer_stats[0].name == "encoder.weight"
