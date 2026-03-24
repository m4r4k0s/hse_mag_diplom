from __future__ import annotations

import numpy as np


def ber(u_true: np.ndarray, u_hat: np.ndarray) -> float:
    return float(np.mean(u_true != u_hat))


def fer(u_true: np.ndarray, u_hat: np.ndarray) -> float:
    if u_true.ndim == 1:
        return float(np.any(u_true != u_hat))
    block_errors = np.any(u_true != u_hat, axis=1)
    return float(np.mean(block_errors))
