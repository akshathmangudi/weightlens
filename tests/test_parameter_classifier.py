from __future__ import annotations

from weightlens.classifiers.pytorch import PyTorchParameterClassifier


def _classify(name: str) -> str:
    return PyTorchParameterClassifier().classify(name, shape=(1,), dtype="float32")


class TestKernel:
    def test_linear_weight(self) -> None:
        assert _classify("encoder.linear.weight") == "kernel"

    def test_conv_weight(self) -> None:
        assert _classify("features.0.weight") == "kernel"

    def test_projection_weight(self) -> None:
        assert _classify("model.layers.0.self_attn.q_proj.weight") == "kernel"

    def test_fallback_unknown(self) -> None:
        assert _classify("some_random_param") == "kernel"


class TestBias:
    def test_linear_bias(self) -> None:
        assert _classify("encoder.linear.bias") == "bias"

    def test_conv_bias(self) -> None:
        assert _classify("features.0.bias") == "bias"


class TestNormScale:
    def test_batchnorm_weight(self) -> None:
        assert _classify("bn1.weight") == "norm_scale"

    def test_layernorm_weight(self) -> None:
        assert _classify("model.layers.0.input_layernorm.weight") == "norm_scale"

    def test_groupnorm_weight(self) -> None:
        assert _classify("group_norm.weight") == "norm_scale"


class TestNormShift:
    def test_batchnorm_bias(self) -> None:
        assert _classify("bn1.bias") == "norm_shift"

    def test_layernorm_bias(self) -> None:
        assert _classify("model.layers.0.input_layernorm.bias") == "norm_shift"


class TestEmbedding:
    def test_embed_tokens(self) -> None:
        assert _classify("encoder.embed_tokens.weight") == "embedding"

    def test_word_embedding(self) -> None:
        assert _classify("model.embedding.weight") == "embedding"

    def test_position_embedding(self) -> None:
        assert _classify("positional_embed.weight") == "embedding"


class TestBuffer:
    def test_running_mean(self) -> None:
        assert _classify("bn1.running_mean") == "buffer"

    def test_running_var(self) -> None:
        assert _classify("bn1.running_var") == "buffer"


class TestAdapter:
    def test_lora_a(self) -> None:
        assert _classify("base_model.model.lora_A.weight") == "adapter"

    def test_lora_b(self) -> None:
        assert _classify("base_model.model.lora_B.weight") == "adapter"

    def test_adapter_module(self) -> None:
        assert _classify("encoder.adapter.weight") == "adapter"


class TestSkip:
    def test_num_batches_tracked(self) -> None:
        assert _classify("bn1.num_batches_tracked") == "skip"
