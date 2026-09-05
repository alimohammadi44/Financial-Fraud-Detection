from __future__ import annotations

import math
import numpy as np
import torch


def haar_approximation_level2(x: np.ndarray) -> np.ndarray:
    """Return level-2 Haar approximation coefficients row-wise.

    The paper specifies a level-2 Haar DWT and retains the approximation
    coefficients, but it does not state the boundary-extension mode. This
    reproduction uses symmetric end padding by repeating the last value when a
    level has odd length, then applies the orthonormal Haar low-pass operation.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape={x.shape}")
    out = np.asarray(x, dtype=np.float64)
    for _ in range(2):
        if out.shape[1] % 2:
            out = np.concatenate([out, out[:, -1:]], axis=1)
        out = (out[:, 0::2] + out[:, 1::2]) / math.sqrt(2.0)
    return out.astype(np.float32)


def sinusoidal_time_encoding(time_step: torch.Tensor, dim: int = 8) -> torch.Tensor:
    """Fixed sinusoidal encoding before the paper's learnable time projection."""
    if dim <= 0 or dim % 2 != 0:
        raise ValueError("time encoding dimension must be a positive even integer")
    t = time_step.float().reshape(-1, 1)
    k = torch.arange(0, dim, 2, device=t.device, dtype=t.dtype)
    denom = torch.pow(torch.tensor(10000.0, device=t.device, dtype=t.dtype), k / dim)
    angles = t / denom
    enc = torch.empty((t.shape[0], dim), device=t.device, dtype=t.dtype)
    enc[:, 0::2] = torch.sin(angles)
    enc[:, 1::2] = torch.cos(angles)
    return enc
