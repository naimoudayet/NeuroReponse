from __future__ import annotations

import numpy as np


def sliding_windows(x: np.ndarray, window: int, hop: int) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError(f"sliding_windows expects a 1-D signal, got shape {x.shape}")
    if window <= 0 or hop <= 0:
        raise ValueError("window and hop must be positive")
    n = (len(x) - window) // hop + 1
    if n <= 0:
        return np.empty((0, window), dtype=x.dtype)
    indices = np.arange(window)[None, :] + hop * np.arange(n)[:, None]
    return x[indices]
