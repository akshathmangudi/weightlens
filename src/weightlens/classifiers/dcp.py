from __future__ import annotations

import re

from weightlens.categories import ParameterCategory
from weightlens.classifiers.pytorch import PyTorchParameterClassifier
from weightlens.contracts import ParameterClassifier

_OPTIMIZER_RE = re.compile(r"(^|\.)optimizer\.state\.")


class DCPParameterClassifier(ParameterClassifier):
    """Classify DCP (Megatron-LM) parameter names.

    DCP checkpoints bundle model weights and optimizer state together.
    This classifier detects optimizer tensors and handles Megatron-style
    norm naming before delegating standard names to
    :class:`PyTorchParameterClassifier`.

    Parameters
    ----------
    include_optimizer:
        When *True*, optimizer tensors are categorised as ``"optimizer"``
        (a separate bucket with no diagnostic rules).  When *False*
        (default) they are mapped to ``"skip"`` so they are excluded
        entirely from analysis.
    """

    def __init__(self, *, include_optimizer: bool = False) -> None:
        self._include_optimizer = include_optimizer
        self._pytorch = PyTorchParameterClassifier()

    def classify(
        self, name: str, shape: tuple[int, ...], dtype: str
    ) -> ParameterCategory:
        # --- optimizer state detection ---
        if _OPTIMIZER_RE.search(name):
            return "optimizer" if self._include_optimizer else "skip"

        # --- Megatron-LM norm naming (e.g. layer_norm_weight) ---
        if name.endswith("_norm_weight"):
            return "norm_scale"
        if name.endswith("_norm_bias"):
            return "norm_shift"

        # --- delegate everything else ---
        return self._pytorch.classify(name, shape, dtype)
