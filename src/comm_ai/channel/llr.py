from __future__ import annotations

import numpy as np


def bpsk_awgn_llr(y: np.ndarray, sigma2: float) -> np.ndarray:
    return 2.0 * y / sigma2
