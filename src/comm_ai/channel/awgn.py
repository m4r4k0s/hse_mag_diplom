from __future__ import annotations

import numpy as np


def sigma2_from_snr_db(snr_db: float, rate: float = 0.5) -> float:
    ebn0 = 10 ** (snr_db / 10.0)
    return 1.0 / (2.0 * rate * ebn0)


def awgn_channel(x: np.ndarray, sigma2: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    noise = rng.normal(0.0, np.sqrt(sigma2), size=x.shape)
    y = x + noise
    return y, noise
