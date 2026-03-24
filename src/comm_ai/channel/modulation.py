from __future__ import annotations

import numpy as np


def bpsk_modulate(coded_bits: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * coded_bits.astype(np.float64)
