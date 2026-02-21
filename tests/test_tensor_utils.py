from __future__ import annotations

import numpy as np
import torch

from weightlens.tensor_utils import tensor_to_numpy


def test_zero_copy_float32_contiguous_cpu() -> None:
    """float32 contiguous CPU tensor should share memory (zero-copy)."""
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    arr = tensor_to_numpy(t)
    assert arr.dtype == np.float32
    assert np.shares_memory(arr, t.numpy())


def test_float16_produces_float32_copy() -> None:
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    arr = tensor_to_numpy(t)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0], atol=1e-3)


def test_bfloat16_produces_float32_copy() -> None:
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    arr = tensor_to_numpy(t)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0], atol=1e-2)


def test_multidimensional_is_flattened() -> None:
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    arr = tensor_to_numpy(t)
    assert arr.shape == (4,)
    np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0, 4.0])


def test_non_contiguous_tensor() -> None:
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).t()
    assert not t.is_contiguous()
    arr = tensor_to_numpy(t)
    assert arr.dtype == np.float32
    np.testing.assert_array_equal(arr, [1.0, 3.0, 2.0, 4.0])
