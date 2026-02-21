from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


def tensor_to_numpy(tensor: torch.Tensor) -> NDArray[np.floating]:
    """Convert a PyTorch tensor to a flat float32 numpy array.

    Avoids unnecessary copies: when the tensor is already float32,
    contiguous, and on CPU, ``numpy()`` is zero-copy.
    """
    t = tensor.detach()
    if not t.is_cpu:
        t = t.cpu()
    if not t.is_contiguous():
        t = t.contiguous()
    if t.dtype != torch.float32:
        t = t.to(torch.float32)
    return t.reshape(-1).numpy()
