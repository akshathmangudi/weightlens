import numpy as np
import pytest

from weightlens.models import LayerTensor


def test_layer_tensor_accepts_valid_payload() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    layer = LayerTensor(
        name="encoder.weight",
        values=values,
        shape=(3,),
        dtype=str(values.dtype),
    )

    assert layer.name == "encoder.weight"
    assert layer.values is values
    assert layer.shape == (3,)
    assert layer.dtype == "float32"


def test_layer_tensor_rejects_non_ndarray_values() -> None:
    with pytest.raises(TypeError, match="values must be a numpy.ndarray"):
        LayerTensor(
            name="bad.values",
            values=[1.0, 2.0],  # type: ignore[arg-type]
            shape=(2,),
            dtype="float32",
        )


def test_layer_tensor_rejects_non_tuple_shape() -> None:
    values = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(TypeError, match="shape must be a tuple"):
        LayerTensor(
            name="bad.shape",
            values=values,
            shape=[2],  # type: ignore[arg-type]
            dtype="float32",
        )
