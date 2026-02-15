from __future__ import annotations

from weightlens.categories import ParameterCategory
from weightlens.contracts import ParameterClassifier


class PyTorchParameterClassifier(ParameterClassifier):
    """Classify PyTorch parameter names into categories via string matching."""

    def classify(
        self, name: str, shape: tuple[int, ...], dtype: str
    ) -> ParameterCategory:
        _ = shape, dtype  # classification is name-only for PyTorch

        # --- specific non-learnable bookkeeping ---
        if "num_batches_tracked" in name:
            return "skip"

        # --- running statistics (batch-norm buffers) ---
        if "running_mean" in name or "running_var" in name:
            return "buffer"

        # --- adapter / LoRA parameters ---
        if "lora_" in name or "adapter" in name:
            return "adapter"

        ends_with_weight = name.endswith(".weight")
        ends_with_bias = name.endswith(".bias")

        # --- embedding ---
        if ends_with_weight and "embed" in name:
            return "embedding"

        # --- normalisation layers (covers layernorm, groupnorm, batchnorm/bn*) ---
        is_norm = "norm" in name or self._is_bn_param(name)
        if ends_with_weight and is_norm:
            return "norm_scale"
        if ends_with_bias and is_norm:
            return "norm_shift"

        # --- generic weight / bias ---
        if ends_with_weight:
            return "kernel"
        if ends_with_bias:
            return "bias"

        # fallback: treat unknown parameters as kernels
        return "kernel"

    @staticmethod
    def _is_bn_param(name: str) -> bool:
        """Check if the parameter belongs to a batch-norm layer (e.g. bn1.weight)."""
        # Extract the module name (part before the last dot)
        dot_idx = name.rfind(".")
        if dot_idx < 0:
            return False
        module = name[:dot_idx]
        # Get the last component of the module path
        last_dot = module.rfind(".")
        leaf = module[last_dot + 1 :] if last_dot >= 0 else module
        # Match "bn", "bn1", "bn2", "batchnorm", etc.
        return leaf.startswith("bn") or leaf.startswith("batchnorm")
