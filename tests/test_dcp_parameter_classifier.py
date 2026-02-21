from __future__ import annotations

from weightlens.classifiers.dcp import DCPParameterClassifier


def _classify(name: str, *, include_optimizer: bool = False) -> str:
    return DCPParameterClassifier(include_optimizer=include_optimizer).classify(
        name, shape=(1,), dtype="float32"
    )


# ------------------------------------------------------------------
# Optimizer state tensors
# ------------------------------------------------------------------


class TestOptimizerState:
    def test_exp_avg_sq_skipped_by_default(self) -> None:
        name = "chained_0.optimizer.state.exp_avg_sq.decoder.layers.0.weight"
        assert _classify(name) == "skip"

    def test_exp_avg_skipped_by_default(self) -> None:
        name = "chained_0.optimizer.state.exp_avg.decoder.layers.0.weight"
        assert _classify(name) == "skip"

    def test_step_skipped_by_default(self) -> None:
        name = "chained_0.optimizer.state.step.decoder.layers.0.weight"
        assert _classify(name) == "skip"

    def test_exp_avg_sq_included_when_flag_set(self) -> None:
        name = "chained_0.optimizer.state.exp_avg_sq.decoder.layers.0.weight"
        assert _classify(name, include_optimizer=True) == "optimizer"

    def test_exp_avg_included_when_flag_set(self) -> None:
        name = "chained_0.optimizer.state.exp_avg.decoder.layers.0.weight"
        assert _classify(name, include_optimizer=True) == "optimizer"

    def test_bare_prefix_optimizer_state(self) -> None:
        name = "optimizer.state.exp_avg.layer.weight"
        assert _classify(name) == "skip"

    def test_bare_prefix_included(self) -> None:
        name = "optimizer.state.exp_avg.layer.weight"
        assert _classify(name, include_optimizer=True) == "optimizer"


# ------------------------------------------------------------------
# False positive guard: model params with "optimizer" in the name
# ------------------------------------------------------------------


class TestOptimizerFalsePositive:
    def test_optimizer_head_weight_is_kernel(self) -> None:
        assert _classify("model.optimizer_head.weight") == "kernel"

    def test_optimizer_head_bias_is_bias(self) -> None:
        assert _classify("model.optimizer_head.bias") == "bias"


# ------------------------------------------------------------------
# Megatron-LM norm naming
# ------------------------------------------------------------------


class TestMegatronNorm:
    def test_layer_norm_weight(self) -> None:
        assert _classify("decoder.layers.0.layer_norm_weight") == "norm_scale"

    def test_input_layer_norm_weight(self) -> None:
        assert _classify("decoder.layers.0.input_layer_norm_weight") == "norm_scale"

    def test_layer_norm_bias(self) -> None:
        assert _classify("decoder.layers.0.layer_norm_bias") == "norm_shift"

    def test_post_attention_norm_weight(self) -> None:
        assert (
            _classify("decoder.layers.0.post_attention_norm_weight") == "norm_scale"
        )


# ------------------------------------------------------------------
# Delegation to PyTorchParameterClassifier
# ------------------------------------------------------------------


class TestDelegation:
    def test_linear_weight(self) -> None:
        assert _classify("encoder.linear.weight") == "kernel"

    def test_linear_bias(self) -> None:
        assert _classify("encoder.linear.bias") == "bias"

    def test_embedding(self) -> None:
        assert _classify("model.embed_tokens.weight") == "embedding"

    def test_layernorm_weight(self) -> None:
        assert _classify("model.layers.0.input_layernorm.weight") == "norm_scale"

    def test_layernorm_bias(self) -> None:
        assert _classify("model.layers.0.input_layernorm.bias") == "norm_shift"

    def test_adapter(self) -> None:
        assert _classify("base_model.model.lora_A.weight") == "adapter"

    def test_running_mean_buffer(self) -> None:
        assert _classify("bn1.running_mean") == "buffer"

    def test_num_batches_tracked_skip(self) -> None:
        assert _classify("bn1.num_batches_tracked") == "skip"
