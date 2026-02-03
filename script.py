from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weightlens.contracts import DiagnosticRule
    from weightlens.models import DiagnosticFlag, LayerStats

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"


def _ensure_src_path() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


def analyze(checkpoint_path: Path) -> None:
    _ensure_src_path()
    from weightlens.aggregators import StreamingGlobalAggregator
    from weightlens.diagnostics import (
        AbnormalNormRule,
        DeadLayerRule,
        ExplodingVarianceRule,
        ExtremeSpikeRule,
    )
    from weightlens.sources import PyTorchWeightSource
    from weightlens.stats_engines import BasicStatsEngine
    from weightlens.validators import PyTorchCheckpointValidator

    validator = PyTorchCheckpointValidator(checkpoint_path)
    health = validator.validate()
    print(f"\n== {checkpoint_path} ==")
    print("health:", health.model_dump())

    if not health.loadable or health.is_empty:
        print("skip: checkpoint not loadable or empty")
        return

    source = PyTorchWeightSource(checkpoint_path)
    engine = BasicStatsEngine()
    aggregator = StreamingGlobalAggregator()

    layer_stats: list[LayerStats] = []
    for layer in source.iter_layers():
        stats = engine.compute_layer(layer)
        layer_stats.append(stats)
        aggregator.update(layer.values)
        aggregator.update_layer_stats(stats)

    global_stats = aggregator.finalize()
    print("global_stats:", global_stats.model_dump())

    rules: list[DiagnosticRule] = [
        DeadLayerRule(),
        ExplodingVarianceRule(),
        ExtremeSpikeRule(),
        AbnormalNormRule(),
    ]
    diagnostics: list[DiagnosticFlag] = []
    for stats in layer_stats:
        for rule in rules:
            flag = rule.check(stats, global_stats)
            if flag is not None:
                diagnostics.append(flag)

    print(f"diagnostics: {len(diagnostics)}")
    for flag in diagnostics:
        print(flag.model_dump())


def main() -> None:
    assets_dir = REPO_ROOT / "assets"
    checkpoints = list(assets_dir.glob("*.pth"))
    if not checkpoints:
        print(f"No .pth files found under {assets_dir}")
        return

    for path in checkpoints:
        analyze(path)


if __name__ == "__main__":
    main()